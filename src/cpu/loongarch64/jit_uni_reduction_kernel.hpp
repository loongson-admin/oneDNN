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

#ifndef CPU_LOONGARCH64_UNI_REDUCTION_KERNEL_HPP
#define CPU_LOONGARCH64_UNI_REDUCTION_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_resampling_pd.hpp"

#include "cpu/loongarch64/jit_generator.hpp"
#include "cpu/loongarch64/jit_primitive_conf.hpp"
#include "cpu/loongarch64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

struct jit_uni_reduction_kernel_base_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_resampling)

    jit_uni_reduction_kernel_base_t(const jit_reduction_conf_t &conf)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, conf.isa), conf_(conf) {}
    virtual ~jit_uni_reduction_kernel_base_t() = default;

    virtual std::size_t get_simd_w() = 0;

protected:
    const jit_reduction_conf_t &conf_;
};

template <typename Vmm>
struct jit_uni_reduction_kernel_t : public jit_uni_reduction_kernel_base_t {
    jit_uni_reduction_kernel_t(
            const jit_reduction_conf_t &conf, const memory_desc_t *dst_md);

    virtual ~jit_uni_reduction_kernel_t() = default;

    std::size_t get_simd_w() override { return simd_w_; }

private:
    using compute_fn_t = std::function<void(
            const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &to_acc)>;

    void init_acc();
    void init_compute_op();
    void init_compute_scalar_op();

    void reduce_ymm_to_xmm(const Xbyak_loongarch64::VReg &acc, const Xbyak_loongarch64::VReg &tmp);

    void reduce_xmm_to_scalar(const Xbyak_loongarch64::VReg &acc,
            const Xbyak_loongarch64::VReg &tmp, const std::size_t number_of_values_to_reduce
            = number_of_f32_in_xmm_);

    void reduce_ymm_to_scalar(const Xbyak_loongarch64::VReg &acc,
            const Xbyak_loongarch64::VReg &tmp1,
            const Xbyak_loongarch64::VReg &tmp2,
            const std::size_t number_of_values_to_reduce
            = number_of_f32_in_ymm_);
    void reduce_vmm_to_scalar(const Xbyak_loongarch64::VReg &acc,
            const Xbyak_loongarch64::VReg &tmp1, const Xbyak_loongarch64::VReg &tmp2,
            const Xbyak_loongarch64::VReg &tmp3, const std::size_t number_of_values_to_reduce
            = number_of_f32_in_ymm_);


    void reduce();

    void load_params();
    void finalize();
    void generate() override;

    const Vmm vmm_tail_load_mask_ = Vmm(0);
    const Vmm vmm_tail_store_mask_ = Vmm(1);
    const Vmm vmm_zero_saturation_ = Vmm(2);
    const Vmm vmm_saturation_ubound_ = Vmm(3);
    const Vmm vmm_acc_ = Vmm(4);
    const Vmm vmm_tmp1_ = Vmm(5);
    const Vmm vmm_tmp2_ = Vmm(6);
    const Vmm vmm_tmp3_ = Vmm(7);
    const Vmm vmm_tmp4_ = Vmm(8);

    const Xbyak_loongarch64::XReg &k_tail_load_mask_ = t7;
    const Xbyak_loongarch64::XReg &k_tail_store_mask_ = t8;

    const Xbyak_loongarch64::XReg &reg_work_ = t0;//rax;
    const Xbyak_loongarch64::XReg &reg_src_ = t1;//rbx;
    const Xbyak_loongarch64::XReg &reg_dst_ = t4;//rdx;
    const Xbyak_loongarch64::XReg &reg_param_ = abi_param1;
    const Xbyak_loongarch64::XReg &reg_tmp_ = abi_not_param1;

    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak_loongarch64::XVReg>::value;
    static constexpr std::size_t vlen_ = is_ymm_ ? 32 : 16;
    static constexpr std::size_t simd_w_ = vlen_ / sizeof(float);
    static constexpr std::size_t number_of_f32_in_xmm_ = 4;
    static constexpr std::size_t number_of_f32_in_ymm_ = 8;

    const std::size_t tail_size_;

    io::jit_io_helper_t<Vmm> io_load_;
    io::jit_io_helper_t<Vmm> io_store_;

    compute_fn_t compute_op_;
    compute_fn_t compute_scalar_op_;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
