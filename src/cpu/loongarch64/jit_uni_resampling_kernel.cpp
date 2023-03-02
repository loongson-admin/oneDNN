/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <bitset>

#include "common/c_types_map.hpp"

#include "cpu/loongarch64/jit_generator.hpp"
#include "cpu/loongarch64/jit_uni_resampling_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace Xbyak_loongarch64;
using namespace format_tag;
using tag_kind = jit_memory_tag_kind_t;

#define GET_OFF(field) offsetof(jit_resampling_call_s, field)

template <cpu_isa_t isa, typename Vmm>
jit_uni_resampling_kernel_t<isa, Vmm>::jit_uni_resampling_kernel_t(
        const jit_resampling_conf_t &conf, const memory_desc_t *dst_md)
    : jit_uni_resampling_kernel_base_t(conf)
    , tail_size_(calculate_tail_size())
    , io_(this, conf_.isa, {conf_.src_data_type, conf_.dst_data_type},
              {can_movntps_be_used()},
              io::io_tail_conf_t {simd_w_, tail_size_, k_tail_mask_,
                      vmm_tail_mask_.getIdx(), reg_tmp_},
              /*io::io_emu_bf16_conf_t {vmm_bf16_emu_1_, vmm_bf16_emu_2_,
                      vmm_bf16_emu_3_, reg_tmp_, vmm_bf16_emu_4_},*/
              create_saturation_vmm_map(),
              io::io_gather_conf_t {simd_w_, k_full_mask_,
                      vmm_full_mask_.getIdx(), reg_tmp_, reg_tmp1_,
                      vmm_tmp_gather_.getIdx()}) {
    if (conf_.with_postops) {
        memory_desc_wrapper dst_d = memory_desc_wrapper(*dst_md);

        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr bool use_exact_tail_scalar_bcast = true;

        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<size_t>(vmm_post_op_helper_.getIdx()), reg_src_,
                reg_tmp_, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), dst_d, tail_size_,
                k_tail_mask_, use_exact_tail_scalar_bcast};

        const bcast_set_t accepted_broadcasts
                = {broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial};
        const binary_injector::static_params_t bsp {
                reg_param, accepted_broadcasts, rhs_sp};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<isa, Vmm>>(
                this, conf_.post_ops, bsp);

        std::tie(any_binary_postop_is_per_oc_bcast_type_,
                any_binary_postop_is_per_oc_sp_bcast_type_)
                = binary_injector_utils::bcast_strategies_present_tup(
                        conf_.post_ops.entry_, dst_d,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial);
    }
}

template <cpu_isa_t isa, typename Vmm>
bool jit_uni_resampling_kernel_t<isa, Vmm>::can_movntps_be_used() const {
    const std::size_t alignment = simd_w_ * conf_.dst_dt_size;

    assert(alignment > 0 && conf_.dst_dt_size > 0
            && "Incorrect output data type size.");

    bool are_data_filling_register_fully = false;
    switch (conf_.dst_data_type) {
        case data_type::f32:
        case data_type::s32: are_data_filling_register_fully = true; break;
        /*case data_type::bf16:
            are_data_filling_register_fully = is_xmm_ ? false : true;
            break;*/
        case data_type::s8:
        case data_type::u8:
            are_data_filling_register_fully = false;
            break;
        default: assert(!"Unsupported data type."); break;
    }

    // When movntps can be used:
    // 1) There is no tail size because movntps has no possibility to store
    // data with a mask. The blocked format is an exception from this rule
    // because there is a padded area and for io operation, there is no use of masks.
    // 2) Data are filling the register fully. Example: Zmm register can hold sixteen
    // f32 values, so there is a possibility to calculate 16 values at the same time,
    // but during store operation of i8 data the same sixteen values can hold only xmm
    // register. If ymm will be used then eight values of f32 can be hold, but neither
    // zmm nor ymm nor xmm can hold i8 data fully because data size is 64 bits only.
    // 3) The memory operand must be aligned on a 16-byte (128-bit version),
    // 32-byte (VEX.256 encoded version) or 64-byte (EVEX.512 encoded version)
    // boundary otherwise a general-protection exception (#GP) will be generated.
    // 4) Instruction is supported and the register is fully filled with data.
    // 5) Data is big enough to see profit from using non-temporal stores.
    bool can_use_movntps = false;
    if (conf_.dst_dt_size % 4 == 0)
        can_use_movntps = conf_.is_data_size_bigger_than_L3
                && are_data_filling_register_fully
                && conf_.output_data_size % alignment == 0
                && (tail_size_ == 0 || conf_.tag_kind == tag_kind::blocked);

    return can_use_movntps;
}

template <cpu_isa_t isa, typename Vmm>
std::size_t jit_uni_resampling_kernel_t<isa, Vmm>::calculate_tail_size() const {
    std::size_t tail_size = 0;

    if (conf_.tag_kind == tag_kind::nspc
            || conf_.tag_kind == tag_kind::blocked) {
        tail_size = conf_.c % simd_w_;
    } else if (conf_.tag_kind == tag_kind::ncsp) {
        if (conf_.alg == alg_kind::resampling_nearest)
            tail_size = conf_.ow % simd_w_;
        else
            tail_size = (conf_.od * conf_.oh * conf_.ow) % simd_w_;
    } else
        assert(!"Incorrect memory tag passed to resampling primitive.");

    return tail_size;
}

template <cpu_isa_t isa, typename Vmm>
int jit_uni_resampling_kernel_t<isa, Vmm>::get_channels_to_compute_without_tail(
        const bool is_tail_in_blocked_format) const {
    assert(utils::one_of(conf_.tag_kind, tag_kind::blocked, tag_kind::nspc)
            && "Incorrect memory tag.");

    int c_to_compute_without_tail = 0;

    if (conf_.tag_kind == tag_kind::blocked && is_tail_in_blocked_format) {
        // Example:
        // c = 27
        // c_block = 16
        // simd_w = 4
        // result = ((27 % 16) / 4) * 4 = (11 / 4) * 4 = 2 * 4 = 8
        c_to_compute_without_tail
                = ((conf_.c % conf_.inner_stride) / simd_w_) * simd_w_;
    } else
        c_to_compute_without_tail = (conf_.inner_stride / simd_w_) * simd_w_;

    return c_to_compute_without_tail;
}

template <cpu_isa_t isa, typename Vmm>
std::map<data_type_t, io::io_saturation_conf_t>
jit_uni_resampling_kernel_t<isa, Vmm>::create_saturation_vmm_map() const {

    std::map<data_type_t, io::io_saturation_conf_t> saturation_map {};

    if (conf_.is_saturation_needed)
        saturation_map.emplace(conf_.dst_data_type,
                io::io_saturation_conf_t {vmm_zero_saturation_.getIdx(),
                        vmm_saturation_ubound_.getIdx(), reg_tmp_});

    return saturation_map;
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa,
        Vmm>::get_params_for_linear_in_c_oriented_format() {
    uni_ld_d(reg_src_ftl_, reg_param, GET_OFF(src));
    uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_front));
    add_d(reg_src_ftl_, reg_src_ftl_, X_TMP_0);
    uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_top));
    add_d(reg_src_ftl_, reg_src_ftl_, X_TMP_0);
    add_d(reg_src_ftr_, reg_src_ftl_, zero);

    if (conf_.ndims == 4 || conf_.ndims == 5) {
        uni_xvldrepl_w(weight_top_, reg_param, GET_OFF(weight_top));
        uni_xvldrepl_w(weight_bottom_, reg_param, GET_OFF(weight_bottom));
        ld_d(reg_src_fbl_, reg_param, GET_OFF(src));
        uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_front));
        add_d(reg_src_fbl_, reg_src_fbl_, X_TMP_0);
        uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_bottom));
        add_d(reg_src_fbl_, reg_src_fbl_, X_TMP_0);
        add_d(reg_src_fbr_, reg_src_fbl_, zero);
    }
    if (conf_.ndims == 5) {
        uni_xvldrepl_w(weight_front_, reg_param, GET_OFF(weight_front));
        uni_xvldrepl_w(weight_back_, reg_param, GET_OFF(weight_back));
        uni_ld_d(reg_src_btl_, reg_param, GET_OFF(src));
        uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_back));
        add_d(reg_src_btl_, reg_src_btl_, X_TMP_0);
        uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_top));
        add_d(reg_src_btl_, reg_src_btl_, X_TMP_0);
        add_d(reg_src_btr_, reg_src_btl_, zero);

        uni_ld_d(reg_src_bbl_, reg_param, GET_OFF(src));
        uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_back));
        add_d(reg_src_bbl_, reg_src_bbl_, X_TMP_0);
        uni_ld_d(X_TMP_0, reg_param, GET_OFF(src_offset_bottom));
        add_d(reg_src_bbl_, reg_src_bbl_, X_TMP_0);
        add_d(reg_src_bbr_, reg_src_bbl_, zero);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::preserve_zero_padding_in_post_ops(
        const int data_idx) {
    Vmm vmm_data(data_idx);
    const Vmm vmm_zeros(vmm_tmp_.getIdx());

    uni_vpxor(vmm_zeros, vmm_zeros, vmm_zeros);

    static const uint32_t tail_mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                    0, 0, 0, 0, 0, 0, 0};
    mov_imm(X_TMP_0,reinterpret_cast<size_t>(&tail_mask[8 - tail_size_]));
    uni_xvld(Vmm(15), X_TMP_0, 0);
    xvbitsel_v(vmm_data, vmm_data, vmm_zeros, Vmm(15));
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::apply_sum(
        const int data_idx, const bool is_tail) {
    if (conf_.with_sum) {
        assert(!conf_.sum_scales.empty()
                && "No scales for sum post operation.");
        const auto sum_injector = [this, data_idx, is_tail]() {
            const Vmm vmm_prev_dst(vmm_tmp_.getIdx());
            const Vmm vmm_dst(data_idx);

            // Zeroing previous dst is needed to preserve zero padding.
            if (is_tail && conf_.tag_kind == tag_kind::blocked)
                uni_vpxor(vmm_prev_dst, vmm_prev_dst, vmm_prev_dst);

            io_.at(conf_.dst_data_type)
                    ->load(reg_dst_, 0, vmm_prev_dst, is_tail);
            const float sum_scale = sum_scales_.front();
            if (sum_scale == 1.f)
                uni_fadd_s(vmm_dst, vmm_dst, vmm_prev_dst);
            else {
                // If the algorithm used is the linear algorithm, and the shape
                // has 5 dimensions, then we have not enough gpr registers to use
                // tmp registers. Therefore, if there is a need to use them it is
                // needed to save their state and restore it after execution of all
                // needed operations.
                if (conf_.alg == alg_kind::resampling_linear
                        && conf_.ndims == 5)
                    push_xreg(reg_tmp1_);
                mov_imm(reg_tmp1_, float2int(sum_scale));
                if (conf_.alg == alg_kind::resampling_linear
                        && conf_.ndims == 5)
                    pop_xreg(reg_tmp1_);
                uni_replgr2vr_w(vmm_sum_scale_, reg_tmp1_);
                uni_fmadd_s(vmm_dst, vmm_prev_dst, vmm_sum_scale_, vmm_dst);
            }
            sum_scales_.push(sum_scale);
            sum_scales_.pop();
        };
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::apply_postops(
        const int data_idx, const bool is_tail, const Reg64 *reg_c) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    const bool is_preserving_zero_padding_needed = is_tail && conf_.with_eltwise
            && conf_.tag_kind == tag_kind::blocked;
    bool update_c_offset = false;

    if (conf_.with_sum) apply_sum(data_idx, is_tail);

    if (conf_.with_binary) {
        if (any_binary_postop_is_per_oc_bcast_type_
                || any_binary_postop_is_per_oc_sp_bcast_type_) {
            if (conf_.tag_kind == tag_kind::blocked) {
                // If simd width is lower than block size then we
                // need to add the current offset of the processed
                // block to global c_offset. However, it is important
                // to reverse this operation after compute_vector
                // because this can mess up the next post_op operations.
                update_c_offset = conf_.inner_stride > simd_w_;
                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        data_idx, reg_c_offset);
            } else if (conf_.tag_kind == tag_kind::ncsp) {
                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        data_idx, reg_c_offset);
            } else {
                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        data_idx, *reg_c);
            }
        }
        if (is_tail) { rhs_arg_params.vmm_tail_idx_.emplace(data_idx); }
    }

    if (update_c_offset) add_d(reg_c_offset, reg_c_offset, *reg_c);
    postops_injector_->compute_vector(data_idx, rhs_arg_params);
    if (is_preserving_zero_padding_needed)
        preserve_zero_padding_in_post_ops(data_idx);
    if (update_c_offset) sub_d(reg_c_offset, reg_c_offset, *reg_c);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::preserve_zero_padding(
        const int c_to_compute_without_tail, const bool is_tail) {
    const int c_to_compute_with_tail
            = is_tail ? utils::rnd_up(tail_size_, simd_w_) : 0;
    const int c_to_zeroing = conf_.inner_stride - c_to_compute_without_tail
            - c_to_compute_with_tail;

    if (c_to_zeroing > 0) {
        assert(c_to_zeroing % simd_w_ == 0);
        const Vmm vmm_zeros(vmm_tmp_.getIdx());

        for (int c = 0; c < c_to_zeroing; c += simd_w_) {
            uni_vpxor(vmm_zeros, vmm_zeros, vmm_zeros);
            io_.at(conf_.dst_data_type)
                ->store(vmm_zeros, reg_dst_, c * conf_.dst_dt_size, false);
        }

        add_imm(reg_dst_, reg_dst_, c_to_zeroing * conf_.dst_dt_size, X_TMP_0);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::interpolate_c_oriented_format(
        const c_oriented_generation_fn_t &generation_fn) {
    const unsigned c_with_padding = utils::rnd_up(conf_.c, conf_.inner_stride);
    const unsigned padding_size_to_preserve = c_with_padding - conf_.c;

    if (padding_size_to_preserve > 0 && conf_.tag_kind == tag_kind::blocked) {
        Label tail_label;
        Label end_label;
        mov_imm(X_TMP_1, utils::rnd_dn(conf_.c, conf_.inner_stride));
        beq(reg_c_offset, X_TMP_1, tail_label);
        generation_fn(false /*is_tail_in_blocked_format*/);
        b(end_label);
        L(tail_label);
        generation_fn(true /*is_tail_in_blocked_format*/);
        L(end_label);
    } else {
        generation_fn(false /*is_tail_in_blocked_format*/);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::nearest_ncsp_format() {
    const Reg64 &reg_indices_h = reg_aux_src_0_;
    const Reg64 &reg_indices_w = reg_aux_src_1_;
    const Reg64 &reg_src_shifted = reg_aux_src_2_;
    const Reg64 &reg_oh = reg_tmp1_;

    auto nearest_interpolation = ([&](bool is_tail) {
        uni_xvld(vmm_indices_, reg_indices_w, 0);
        io_.at(conf_.src_data_type)
                ->gather(reg_src_shifted, vmm_indices_, vmm_src_, is_tail);
        if (conf_.with_postops) apply_postops(vmm_src_.getIdx(), is_tail);
        io_.at(conf_.dst_data_type)->store(vmm_src_, reg_dst_, 0, is_tail);
    });

    add_d(reg_indices_h, reg_indices_, zero);
    add_d(reg_indices_w, reg_indices_, zero);
    add_imm(reg_indices_w, reg_indices_w, conf_.oh * conf_.el_size_of_indices, X_TMP_0);

    Label oh_loop_begin, oh_loop_end;
    Label ow_loop_begin, ow_loop_end;
    xor_(reg_oh, reg_oh, reg_oh);

    L(oh_loop_begin);
    {
        mov_imm(X_TMP_1, conf_.oh);
        bge(reg_oh, X_TMP_1, oh_loop_end);
        push_xreg(reg_oh);

        mov_imm(reg_work_, conf_.ow);
        add_d(reg_src_shifted, reg_src_, zero);
        xor_(reg_tmp_, reg_tmp_, reg_tmp_);
        uni_ld_w(reg_tmp_, reg_indices_h, 0);
        add_d(reg_src_shifted, reg_src_shifted, reg_tmp_);

        push_xreg(reg_indices_w);

        L(ow_loop_begin);
        {
            mov_imm(X_TMP_1, simd_w_);
            blt(reg_work_, X_TMP_1, ow_loop_end);

            nearest_interpolation(false);

            add_imm(reg_dst_, reg_dst_, simd_w_ * conf_.dst_dt_size, X_TMP_0);
            add_imm(reg_indices_w, reg_indices_w, simd_w_ * conf_.el_size_of_indices, X_TMP_0);
            add_imm(reg_work_, reg_work_, -1 * simd_w_, X_TMP_0);

            b(ow_loop_begin);
        }
        L(ow_loop_end);

        if (tail_size_ > 0) {
            nearest_interpolation(true);
            add_imm(reg_dst_, reg_dst_, tail_size_ * conf_.dst_dt_size, X_TMP_0);
        }

        add_imm(reg_indices_h, reg_indices_h, conf_.el_size_of_indices, X_TMP_0);
        pop_xreg(reg_indices_w);
        pop_xreg(reg_oh);
        addi_d(reg_oh, reg_oh, 1);
        b(oh_loop_begin);
    }
    L(oh_loop_end);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::nearest_c_oriented_format(
        const bool is_tail_in_blocked_format) {
    const int c_to_compute_without_tail
            = get_channels_to_compute_without_tail(is_tail_in_blocked_format);
    const bool insert_tail_processsing_code
            = (conf_.tag_kind == tag_kind::nspc && tail_size_ > 0)
            || is_tail_in_blocked_format;

    const Reg64 &reg_c = reg_tmp_;
    const Reg64 &reg_src_shifted = reg_aux_src_0_;

    auto nearest_interpolation = [&](const bool is_tail) {
        const bool load_and_store_with_tail
                = is_tail && conf_.tag_kind == tag_kind::nspc;

        io_.at(conf_.src_data_type)
                ->load(reg_src_shifted, 0, vmm_src_,
                        load_and_store_with_tail);
        if (conf_.with_postops)
            apply_postops(vmm_src_.getIdx(), is_tail, &reg_c);
        io_.at(conf_.dst_data_type)
                 ->store(vmm_src_, reg_dst_, 0, load_and_store_with_tail);
    };

    Label loop_begin, loop_end;

    L(loop_begin);
    {
        mov_imm(X_TMP_1, 1);
        blt(reg_work_, X_TMP_1, loop_end);

        add_d(reg_src_shifted, reg_src_, zero);
        uni_ld_w(reg_tmp1_, reg_indices_, 0);
        add_d(reg_src_shifted, reg_src_shifted, reg_tmp1_);

        Label c_loop_begin, c_loop_end;
        xor_(reg_c, reg_c, reg_c);
        L(c_loop_begin);
        {
            mov_imm(X_TMP_1, c_to_compute_without_tail);
            beq(reg_c, X_TMP_1, c_loop_end);

            nearest_interpolation(false);

            add_imm(reg_src_shifted, reg_src_shifted, simd_w_ * conf_.src_dt_size, X_TMP_0);
            add_imm(reg_dst_, reg_dst_, simd_w_ * conf_.dst_dt_size, X_TMP_0);

            add_imm(reg_c, reg_c, simd_w_, X_TMP_0);
            b(c_loop_begin);
        }
        L(c_loop_end);

        if (insert_tail_processsing_code) {
            if (tail_size_ > 0) {
                nearest_interpolation(true);
                if (conf_.tag_kind == tag_kind::nspc)
                    add_imm(reg_dst_, reg_dst_, tail_size_ * conf_.dst_dt_size, X_TMP_0);
                else if (conf_.tag_kind == tag_kind::blocked) {
                    add_imm(reg_dst_, reg_dst_, simd_w_ * conf_.dst_dt_size, X_TMP_0);
                }
            }

            if (conf_.tag_kind == tag_kind::blocked)
                preserve_zero_padding(
                        c_to_compute_without_tail, is_tail_in_blocked_format);
        }

        add_imm(reg_indices_, reg_indices_, conf_.el_size_of_indices, X_TMP_0);
        addi_d(reg_work_, reg_work_, -1);
        b(loop_begin);
    }
    L(loop_end);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::linear_ncsp_format() {
    const unsigned indices_stride
            = conf_.ow * conf_.oh * conf_.od * conf_.el_size_of_indices;
    const unsigned weights_stride
            = conf_.ow * conf_.oh * conf_.od * sizeof(float);

    auto linear_interpolation = [&](const bool is_tail) {
        const Vmm vmm_dst(vmm_idx(0));

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            uni_xvld(vmm_indices_, reg_indices_, i * indices_stride);
            io_.at(conf_.src_data_type)
                    ->gather(reg_src_, vmm_indices_, Vmm(vmm_idx(i)), is_tail);
        }

        uni_xvld(vmm_weights_, reg_weights, 0);
        uni_fmul_s(vmm_dst, vmm_dst, vmm_weights_);
        for (unsigned i = 1; i < conf_.number_of_corners; i++) {
            uni_xvld(vmm_weights_, reg_weights, i * weights_stride);
            uni_fmadd_s(vmm_dst, Vmm(vmm_idx(i)), vmm_weights_, vmm_dst);
        }

        if (conf_.with_postops) apply_postops(vmm_idx(0), is_tail);

        if (conf_.is_saturation_needed && conf_.ndims == 5 ) {
            // When saturation is needed, and the shape has
            // 5 dimensions, and we have only 16 Vmm registers,
            // we have no space for holding information for saturation
            // in registers. That is why we need to repeat saturation
            // initialization before every store operation.
            io_.init_saturate_f32({conf_.dst_data_type});
        }

        io_.at(conf_.dst_data_type)->store(vmm_dst, reg_dst_, 0, is_tail);
    };

    Label loop_begin, loop_end;

    L(loop_begin);
    {
        mov_imm(X_TMP_1, simd_w_);
        blt(reg_work_, X_TMP_1, loop_end);

        linear_interpolation(false);

        add_imm(reg_dst_, reg_dst_, simd_w_ * conf_.dst_dt_size, X_TMP_0);
        add_imm(reg_weights, reg_weights, simd_w_ * sizeof(float), X_TMP_0);
        add_imm(reg_indices_, reg_indices_, simd_w_ * conf_.el_size_of_indices, X_TMP_0);
        add_imm(reg_work_, reg_work_, -1 * simd_w_, X_TMP_0);

        b(loop_begin);
    }
    L(loop_end);

    if (tail_size_ > 0) linear_interpolation(true);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::linear_c_oriented_format(
        const bool is_tail_in_blocked_format) {
    const int c_to_compute_without_tail
            = get_channels_to_compute_without_tail(is_tail_in_blocked_format);
    const bool insert_tail_processsing_code
            = (conf_.tag_kind == tag_kind::nspc && tail_size_ > 0)
            || is_tail_in_blocked_format;

    const Reg64 &reg_c = reg_tmp_;
    const Reg64 &reg_index_left = reg_tmp_;
    const Reg64 &reg_index_right = reg_tmp_;

    const std::vector<std::reference_wrapper<const Reg64>> src_regs
            = {reg_src_ftl_, reg_src_ftr_, reg_src_fbl_, reg_src_fbr_,
                    reg_src_btl_, reg_src_btr_, reg_src_bbl_, reg_src_bbr_};
    const std::vector<std::reference_wrapper<const Vmm>> src_vmms
            = {src_ftl_, src_ftr_, src_fbl_, src_fbr_, src_btl_, src_btr_,
                    src_bbl_, src_bbr_};

    assert(src_regs.size() >= conf_.number_of_corners
            && src_vmms.size() >= conf_.number_of_corners);

    auto linear_interpolation = [&](const Reg64 &reg_c, const bool is_tail) {
        const bool load_and_store_with_tail
                = is_tail && conf_.tag_kind == tag_kind::nspc;

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            io_.at(conf_.src_data_type)
                    ->load(src_regs[i].get(), 0, src_vmms[i].get(),
                            load_and_store_with_tail);
        }

        // w_d[0]*(w_h[0]*(src[0][0][0]*w_w[0] + src[0][0][1]*w_w[1]) +
        //         w_h[1]*(src[0][1][0]*w_w[0] + src[0][1][1]*w_w[1]))
        // +
        // w_d[1]*(w_h[0]*(src[1][0][0]*w_w[0] + src[1][0][1]*w_w[1]) +
        //         w_h[1]*(src[1][1][0]*w_w[0] + src[1][1][1]*w_w[1]))
        uni_fmul_s(src_ftl_, src_ftl_, weight_left_);
        uni_fmadd_s(src_ftl_, src_ftr_, weight_right_, src_ftl_);
        if (conf_.ndims == 4 || conf_.ndims == 5) {
            uni_fmul_s(src_fbl_, src_fbl_, weight_left_);
            uni_fmadd_s(src_fbl_, src_fbr_, weight_right_, src_fbl_);
            uni_fmul_s(src_ftl_, src_ftl_, weight_top_);
            uni_fmadd_s(src_ftl_, src_fbl_, weight_bottom_, src_ftl_);
        }
        if (conf_.ndims == 5) {
            uni_fmul_s(src_btl_, src_btl_, weight_left_);
            uni_fmadd_s(src_btl_, src_btr_, weight_right_, src_btl_);
            uni_fmul_s(src_bbl_, src_bbl_, weight_left_);
            uni_fmadd_s(src_bbl_, src_bbr_, weight_right_, src_bbl_);
            uni_fmul_s(src_btl_, src_btl_, weight_top_);
            uni_fmadd_s(src_btl_, src_bbl_, weight_bottom_, src_btl_);
            uni_fmul_s(src_ftl_, src_ftl_, weight_front_);
            uni_fmadd_s(src_ftl_, src_btl_, weight_back_, src_ftl_);
        }

        if (conf_.with_postops)
            apply_postops(src_ftl_.getIdx(), is_tail, &reg_c);

        if (conf_.is_saturation_needed && conf_.ndims == 5 ) {
            // When saturation is needed, and the shape has
            // 5 dimensions, and we have only 16 Vmm registers,
            // we have no space for holding information for saturation
            // in registers. That is why we need to repeat saturation
            // initialization before every store operation.
            push_xreg(reg_tmp_);
            io_.init_saturate_f32({conf_.dst_data_type});
            pop_xreg(reg_tmp_);
        }

        io_.at(conf_.dst_data_type)
                ->store(src_ftl_, reg_dst_, 0, load_and_store_with_tail);
    };

    xor_(reg_index_left, reg_index_left, reg_index_left);

    Label loop_begin, loop_end;
    L(loop_begin);
    {
        mov_imm(X_TMP_1, 1);
        blt(reg_work_, X_TMP_1, loop_end);

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            push_xreg(src_regs[i]);
        }

        uni_ld_w(reg_index_left, reg_indices_, 0);
        for (unsigned i = 0; i < conf_.number_of_corners / 2; i++) {
            add_d(src_regs[2 * i], src_regs[2 * i], reg_index_left);
        }
        uni_ld_w(reg_index_right, reg_indices_, conf_.el_size_of_indices);
        for (unsigned i = 0; i < conf_.number_of_corners / 2; i++) {
            add_d(src_regs[2 * i + 1], src_regs[2 * i + 1], reg_index_right);
        }

        uni_xvldrepl_w(weight_left_, reg_weights, 0);
        uni_xvldrepl_w(weight_right_, reg_weights, sizeof(float));


        Label c_loop_begin, c_loop_end;
        xor_(reg_c, reg_c, reg_c);
        L(c_loop_begin);
        {
            mov_imm(X_TMP_1, c_to_compute_without_tail);
            beq(reg_c, X_TMP_1, c_loop_end);

            linear_interpolation(reg_c, false);
            add_imm(reg_dst_, reg_dst_, simd_w_ * conf_.dst_dt_size, X_TMP_0);

            for (unsigned i = 0; i < conf_.number_of_corners; i++)
                add_imm(src_regs[i], src_regs[i], simd_w_ * conf_.src_dt_size, X_TMP_0);


            add_imm(reg_c, reg_c, simd_w_, X_TMP_0);
            b(c_loop_begin);
        }
        L(c_loop_end);

        if (insert_tail_processsing_code) {
            if (tail_size_ > 0) {
                linear_interpolation(reg_c, true);
                if (conf_.tag_kind == tag_kind::nspc)
                    add_imm(reg_dst_, reg_dst_, tail_size_ * conf_.dst_dt_size, X_TMP_0);
                else if (conf_.tag_kind == tag_kind::blocked) {
                    add_imm(reg_dst_, reg_dst_, simd_w_ * conf_.dst_dt_size, X_TMP_0);
                }
            }

            if (conf_.tag_kind == tag_kind::blocked)
                preserve_zero_padding(
                        c_to_compute_without_tail, is_tail_in_blocked_format);
        }

        // During one loop cycle are read two values for left and
        // right corners from both the weights and indices tables.
        // These two values occurs one after the other in memory,
        // so the address should be shifted by two elements.
        add_imm(reg_indices_, reg_indices_, 2 * conf_.el_size_of_indices, X_TMP_0);
        add_imm(reg_weights, reg_weights, 2 * sizeof(float), X_TMP_0);

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            pop_xreg(src_regs[(conf_.number_of_corners - 1) - i]);
        }

        addi_d(reg_work_, reg_work_, -1);
        b(loop_begin);
    }
    L(loop_end);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_resampling_kernel_t<isa, Vmm>::generate() {
    preamble();

    //io_.init_bf16();
    if (conf_.is_saturation_needed)
        io_.init_saturate_f32({conf_.dst_data_type});
    // Preparing tail is needed for blocked format, because
    // there is chance that padding will not be preserved when user use
    // post-ops.
    if (tail_size_ > 0
            && (conf_.tag_kind != tag_kind::blocked || conf_.with_postops))
        io_.prepare_tail_mask();
    if ((conf_.isa == lasx) && conf_.tag_kind == tag_kind::ncsp) {
        io_.init_full_mask();
        io_.prepare_full_mask();
    }

    ld_d(reg_dst_, reg_param, GET_OFF(dst));
    ld_d(reg_work_, reg_param, GET_OFF(batch_of_sp_points_to_process));
    ld_d(reg_indices_, reg_param, GET_OFF(indices));
    ld_d(reg_c_offset, reg_param, GET_OFF(c_offset));

    if (conf_.alg == alg_kind::resampling_nearest) {
        ld_d(reg_src_, reg_param, GET_OFF(src));
        if (conf_.tag_kind == tag_kind::ncsp) {
            nearest_ncsp_format();
        } else if (conf_.tag_kind == tag_kind::nspc
                || conf_.tag_kind == tag_kind::blocked) {
            interpolate_c_oriented_format(
                    [&](const bool is_tail_in_blocked_format) {
                        nearest_c_oriented_format(is_tail_in_blocked_format);
                    });
        }
    } else if (conf_.alg == alg_kind::resampling_linear) {
        ld_d(reg_weights, reg_param, GET_OFF(weights));
        if (conf_.tag_kind == tag_kind::ncsp) {
            ld_d(reg_src_, reg_param, GET_OFF(src));
            linear_ncsp_format();
        } else if (conf_.tag_kind == tag_kind::nspc
                || conf_.tag_kind == tag_kind::blocked) {
            get_params_for_linear_in_c_oriented_format();
            interpolate_c_oriented_format(
                    [&](const bool is_tail_in_blocked_format) {
                        linear_c_oriented_format(is_tail_in_blocked_format);
                    });
        }
    }

    postamble();

    if (conf_.with_eltwise && postops_injector_)
        postops_injector_->prepare_table();
}


template struct jit_uni_resampling_kernel_t<lasx, XVReg>;
//template struct jit_uni_resampling_kernel_t<lasx, VReg>;
//template struct jit_uni_resampling_kernel_t<lsx, VReg>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
