/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/type_helpers.hpp"

#include "jit_uni_reduction_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace Xbyak_loongarch64;
#define GET_OFF(field) offsetof(jit_reduction_call_s, field)

template <typename Vmm>
jit_uni_reduction_kernel_t<Vmm>::jit_uni_reduction_kernel_t(
        const jit_reduction_conf_t &conf, const memory_desc_t *dst_md)
    : jit_uni_reduction_kernel_base_t(conf)
    , tail_size_(conf.reduce_size % simd_w_)
    , io_load_(this, conf_.isa, conf_.src_type, {false},
              io::io_tail_conf_t {simd_w_, tail_size_, k_tail_load_mask_,
                      vmm_tail_load_mask_.getIdx(), reg_tmp_},
             /* io::io_emu_bf16_conf_t {vmm_bf16_emu_1_, vmm_bf16_emu_2_,
                      vmm_bf16_emu_3_, reg_tmp_, vmm_bf16_emu_4_},*/
              io::io_saturation_conf_t {vmm_zero_saturation_.getIdx(),
                      vmm_saturation_ubound_.getIdx(), reg_tmp_})
    , io_store_(this, conf_.isa, conf_.dst_type, {false},
              io::io_tail_conf_t {simd_w_, 1, k_tail_store_mask_,
                      vmm_tail_store_mask_.getIdx(), reg_tmp_},
            /*  io::io_emu_bf16_conf_t {vmm_bf16_emu_1_, vmm_bf16_emu_2_,
                      vmm_bf16_emu_3_, reg_tmp_, vmm_bf16_emu_4_},*/
              io::io_saturation_conf_t {vmm_zero_saturation_.getIdx(),
                      vmm_saturation_ubound_.getIdx(), reg_tmp_}) {
    init_compute_op();
    init_compute_scalar_op();
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::init_acc() {
    using namespace alg_kind;
    using namespace nstl;

    const VReg xmm_tmp_(vmm_tmp1_.getIdx());
    float starting_val = 0;

    switch (conf_.alg) {
        case reduction_max:
            starting_val = numeric_limits<float>::lowest();
            break;
        case reduction_min: starting_val = numeric_limits<float>::max(); break;
        case reduction_mean:
        case reduction_sum: starting_val = 0.f; break;
        case reduction_mul: starting_val = 1.f; break;
        default: assert(!"unknown alg");
    }

    mov_imm(reg_tmp_, float2int(starting_val));
    uni_replgr2vr_w(vmm_acc_, reg_tmp_);
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::init_compute_op() {
    using namespace alg_kind;
    switch (conf_.alg) {
        case reduction_max:
            compute_op_ = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                uni_fmax_s(acc, acc, to_acc);
            };
            break;
        case reduction_min:
            compute_op_ = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                uni_fmin_s(acc, acc, to_acc);
            };
            break;
        case reduction_mean:
        case reduction_sum:
            compute_op_ = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                uni_fadd_s(acc, acc, to_acc);
            };
            break;
        case reduction_mul:
            compute_op_ = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                uni_fmul_s(acc, acc, to_acc);
            };
            break;
        default: assert(!"unsupported alg.");
    }
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::init_compute_scalar_op() {
    using namespace alg_kind;

    switch (conf_.alg) {
        case reduction_max:
            compute_scalar_op_
                    = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                        uni_fmax_s(VReg(31), acc, to_acc);
                        vextrins_w(acc, VReg(31), 0);
                      };
            break;
        case reduction_min:
            compute_scalar_op_
                    = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                        uni_fmin_s(VReg(31), acc, to_acc);
                        vextrins_w(acc, VReg(31), 0);
                      };
            break;
        case reduction_mean:
        case reduction_sum:
            compute_scalar_op_
                    = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                        uni_fadd_s(VReg(31), acc, to_acc);
                        vextrins_w(acc, VReg(31), 0);
                      };
            break;
        case reduction_mul:
            compute_scalar_op_
                    = [&](const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc) {
                        uni_fmul_s(VReg(31), acc, to_acc);
                        vextrins_w(acc, VReg(31), 0);
                      };
            break;
        default: assert(!"unsupported alg.");
    }
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::reduce_ymm_to_xmm(
        const VReg &acc, const VReg &tmp) {
    const XVReg ymm_acc(acc.getIdx());
    const VReg xmm_acc(acc.getIdx());
    const XVReg xmm_to_acc(tmp.getIdx());
    xvpermi_q(xmm_to_acc, ymm_acc, 0x31);
    compute_op_(xmm_acc, VReg(xmm_to_acc.getIdx()));
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::reduce_xmm_to_scalar(const VReg &acc,
        const VReg &tmp, const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_xmm_);

    const VReg xmm_acc(acc.getIdx());
    const VReg ymm_to_acc(tmp.getIdx());

    static constexpr int number_of_f32_to_move = number_of_f32_in_xmm_ - 1;
    static constexpr uint8_t insertps_configuration[number_of_f32_to_move]
            //= {0b01001110, 0b10001110, 0b11001110};     //数组 {7:6==>1,2,3}
            ={0b00000001, 0b00000010, 0b00000011};

    for (std::size_t i = 0; i < number_of_values_to_reduce - 1; i++) {
        //high不变，低32位改变为xmm的32-63;64-95;96-127
        vextrins_w(ymm_to_acc, xmm_acc, insertps_configuration[i]);
        compute_scalar_op_(xmm_acc, ymm_to_acc);
    }
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::reduce_ymm_to_scalar(
        const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &tmp1,
        const Xbyak_loongarch64::VReg &tmp2,
        const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_ymm_);

    const XVReg ymm_acc(acc.getIdx());
    const VReg xmm_acc(acc.getIdx());
    const VReg xmm_tmp(tmp1.getIdx());
    const VReg xmm_acc_upper_half(tmp2.getIdx());

    if (number_of_values_to_reduce == number_of_f32_in_ymm_) {
        reduce_ymm_to_xmm(VReg(ymm_acc.getIdx()), xmm_tmp);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp);
    } else if (number_of_values_to_reduce > number_of_f32_in_xmm_) {
        xvpermi_q(XVReg(xmm_acc_upper_half.getIdx()), ymm_acc, 0x31);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp);
        reduce_xmm_to_scalar(xmm_acc_upper_half, xmm_tmp,
                number_of_values_to_reduce - number_of_f32_in_xmm_);
        compute_scalar_op_(xmm_acc, xmm_acc_upper_half);
    } else if (number_of_values_to_reduce <= number_of_f32_in_xmm_) {
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp, number_of_values_to_reduce);
    }
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::reduce_vmm_to_scalar(
        const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &tmp1,
        const Xbyak_loongarch64::VReg &tmp2, const Xbyak_loongarch64::VReg &tmp3,
        const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_ymm_);

    const XVReg ymm_acc(acc.getIdx());
    const VReg xmm_tmp1(tmp2.getIdx());
    const VReg xmm_tmp2(tmp3.getIdx());

    if (number_of_values_to_reduce <= number_of_f32_in_ymm_) {
        reduce_ymm_to_scalar(
                VReg(ymm_acc.getIdx()), xmm_tmp1, xmm_tmp2, number_of_values_to_reduce);
    }
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::reduce() {
    Label label_work_begin, label_work_end;

    L(label_work_begin);
    {
        beqz(reg_work_, label_work_end);
        io_load_.load(reg_src_, 0, vmm_tmp1_,false);
        compute_op_(VReg(vmm_acc_.getIdx()), VReg(vmm_tmp1_.getIdx()));

        addi_d(reg_src_, reg_src_, simd_w_ * conf_.src_dt_size);

        addi_d(reg_work_, reg_work_, -1);
        b(label_work_begin);
    }
    L(label_work_end);

    if (tail_size_) {
        io_load_.load(reg_src_, 0, vmm_tmp1_,true);
        reduce_vmm_to_scalar(
                VReg(vmm_tmp1_.getIdx()), VReg(vmm_tmp2_.getIdx()), VReg(vmm_tmp3_.getIdx()), VReg(vmm_tmp4_.getIdx()), tail_size_);
        compute_scalar_op_(VReg(vmm_acc_.getIdx()), VReg(vmm_tmp1_.getIdx()));
    }
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::load_params() {
    ld_d(reg_src_, reg_param_, GET_OFF(src));
    ld_d(reg_dst_, reg_param_, GET_OFF(dst));
    mov_imm(reg_work_, conf_.reduce_size / simd_w_);
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::finalize() {
    if (static_cast<std::size_t>(conf_.reduce_size) > tail_size_) {
        reduce_vmm_to_scalar(
                VReg(vmm_acc_.getIdx()), VReg(vmm_tmp1_.getIdx()), VReg(vmm_tmp2_.getIdx()), VReg(vmm_tmp3_.getIdx()), simd_w_);
    }

    if (conf_.alg == alg_kind::reduction_mean) {
        const VReg xmm_acc(vmm_acc_.getIdx());
        const VReg xmm_reduce_size(vmm_tmp1_.getIdx());
        mov_imm(reg_tmp_, float2int(static_cast<float>(conf_.reduce_size)));
        vinsgr2vr_w(xmm_reduce_size, reg_tmp_, 0);
        vfdiv_s(VReg(31), xmm_acc, xmm_reduce_size);
        vextrins_w(xmm_acc, VReg(31), 0);
    }

    io_store_.store(vmm_acc_, reg_dst_, 0, true);
}

template <typename Vmm>
void jit_uni_reduction_kernel_t<Vmm>::generate() {
    preamble();

    //io_store_.init_bf16();
    if (conf_.is_saturation_needed) io_store_.init_saturate_f32();

    if (tail_size_ > 0) io_load_.prepare_tail_mask();
    io_store_.prepare_tail_mask();

    load_params();
    init_acc();
    reduce();
    finalize();

    postamble();
}

template struct jit_uni_reduction_kernel_t<Xbyak_loongarch64::VReg>;
template struct jit_uni_reduction_kernel_t<Xbyak_loongarch64::XVReg>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
