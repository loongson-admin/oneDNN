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

#ifndef CPU_LOONGARCH64_UNI_RESAMPLING_KERNEL_HPP
#define CPU_LOONGARCH64_UNI_RESAMPLING_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_resampling_pd.hpp"

#include "cpu/loongarch64/cpu_isa_traits.hpp"
#include "cpu/loongarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/loongarch64/jit_generator.hpp"
#include "cpu/loongarch64/jit_primitive_conf.hpp"
#include "cpu/loongarch64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

struct jit_uni_resampling_kernel_base_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_resampling)

    jit_uni_resampling_kernel_base_t(const jit_resampling_conf_t &conf)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, conf.isa)
        , conf_(conf)
        , sum_scales_(conf_.sum_scales) {}

    virtual ~jit_uni_resampling_kernel_base_t() = default;

    virtual std::size_t get_simd_w() = 0;

protected:
    const jit_resampling_conf_t &conf_;
    std::queue<float> sum_scales_;
};

template <cpu_isa_t isa, typename Vmm>
struct jit_uni_resampling_kernel_t : public jit_uni_resampling_kernel_base_t {

    jit_uni_resampling_kernel_t(
            const jit_resampling_conf_t &conf, const memory_desc_t *dst_md);

    virtual ~jit_uni_resampling_kernel_t() = default;

    std::size_t get_simd_w() override { return simd_w_; }

private:
    using VReg = Xbyak_loongarch64::VReg;
    using XVReg = Xbyak_loongarch64::XVReg;
    using Opmask = Xbyak_loongarch64::XReg;
    using Reg64 = Xbyak_loongarch64::XReg;
    using c_oriented_generation_fn_t = std::function<void(const bool)>;

    constexpr int vmm_idx(int idx) const {
        return (cpu_isa_traits<isa>::n_vregs - 1) - idx;
    }

    bool can_movntps_be_used() const;
    std::size_t calculate_tail_size() const;
    int get_channels_to_compute_without_tail(
            bool is_tail_in_blocked_format) const;

    std::map<data_type_t, io::io_saturation_conf_t>
    create_saturation_vmm_map() const;

    void get_params_for_linear_in_c_oriented_format();

    void preserve_zero_padding_in_post_ops(int data_idx);
    void apply_sum(const int data_idx, const bool is_tail);
    void apply_postops(const int data_idx, const bool is_tail,
            const Reg64 *reg_c = nullptr);

    void preserve_zero_padding(
            int c_to_compute_without_tail, const bool is_tail);

    void interpolate_c_oriented_format(
            const c_oriented_generation_fn_t &generation_fn);
    void nearest_ncsp_format();
    void nearest_c_oriented_format(const bool is_tail_in_blocked_format);
    void linear_ncsp_format();
    void linear_c_oriented_format(const bool is_tail_in_blocked_format);

    void generate() override;

    const Vmm vmm_tail_mask_ = Vmm(0);
    // Vgatherdps always gets data using a conditional mask.
    // This register contains all bits set to 1, allowing
    // to get the maximum number of values available to the register
    const Vmm vmm_full_mask_ = Vmm(1);
    const Vmm vmm_src_ = Vmm(2);
    const Vmm vmm_weights_ = Vmm(3);
    const Vmm vmm_indices_ = Vmm(4);
    const Vmm vmm_tmp_gather_ = Vmm(5);
    const Vmm vmm_sum_scale_ = Vmm(7);
    const Vmm vmm_tmp_ = Vmm(8);
    const Vmm vmm_post_op_helper_ = Vmm(9);
    const Vmm vmm_zero_saturation_ = Vmm(10);
    const Vmm vmm_saturation_ubound_ = Vmm(11);

    const Opmask &k_tail_mask_ = s3;
    const Opmask &k_full_mask_ = s4;

    const Reg64 &reg_tmp_ = t0;
    const Reg64 &reg_dst_ = a4;
    const Reg64 &reg_work_ = a5;
    const Reg64 &reg_indices_ = a6;
    const Reg64 &reg_c_offset = a7;
    const Reg64 &reg_param = abi_param1;
    const Reg64 &reg_weights = abi_not_param1;
    const Reg64 &reg_src_ = a3;
    const Reg64 &reg_aux_src_0_ = t1;
    const Reg64 &reg_aux_src_1_ = t2;
    const Reg64 &reg_aux_src_2_ = t3;
    const Reg64 &reg_tmp1_ = t7;

    // Registers which are used only for linear algorithm
    // and for channel oriented formats.
    // Meaning of shortcuts:
    // f - front, b - back
    // t - top, b - bottom
    // l - left, r - right
    // Example:
    // src_ftl_ - source tensor data for the front top left corner
    // reg_src_ftl_ - register which contains address of source
    //                tensor data for the front top left corner
    const Vmm weight_left_ = Vmm(1);
    const Vmm weight_right_ = Vmm(2);
    const Vmm weight_top_ = Vmm(3);
    const Vmm weight_bottom_ = Vmm(4);
    const Vmm weight_front_ = Vmm(5);
    const Vmm weight_back_ = Vmm(6);
    const Vmm src_ftl_ = Vmm(vmm_idx(0));
    const Vmm src_ftr_ = Vmm(vmm_idx(1));
    const Vmm src_fbl_ = Vmm(vmm_idx(2));
    const Vmm src_fbr_ = Vmm(vmm_idx(3));
    const Vmm src_btl_ = Vmm(vmm_idx(4));
    const Vmm src_btr_ = Vmm(vmm_idx(5));
    const Vmm src_bbl_ = Vmm(vmm_idx(6));
    const Vmm src_bbr_ = Vmm(vmm_idx(7));

    const Reg64 &reg_src_ftl_ = reg_src_;
    const Reg64 &reg_src_ftr_ = reg_aux_src_0_;
    const Reg64 &reg_src_fbl_ = reg_aux_src_1_;
    const Reg64 &reg_src_fbr_ = reg_aux_src_2_;
    const Reg64 &reg_src_btl_ = t4;
    const Reg64 &reg_src_btr_ = t5;
    const Reg64 &reg_src_bbl_ = t6;
    const Reg64 &reg_src_bbr_ = t7;

    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak_loongarch64::XVReg>::value;
    static constexpr bool is_xmm_ = std::is_same<Vmm, Xbyak_loongarch64::VReg>::value;
    static constexpr std::size_t vlen_ = is_ymm_ ? 32 : 16;
    static constexpr std::size_t simd_w_ = vlen_ / sizeof(float);
    const std::size_t tail_size_;

    bool any_binary_postop_is_per_oc_bcast_type_ = false;
    bool any_binary_postop_is_per_oc_sp_bcast_type_ = false;

    io::jit_io_multi_dt_helper_t<Vmm> io_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa, Vmm>>
            postops_injector_;
};
} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
