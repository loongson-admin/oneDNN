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

#ifndef CPU_LOONGARCH64_UNI_BINARY_KERNEL_HPP
#define CPU_LOONGARCH64_UNI_BINARY_KERNEL_HPP

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/cpu_isa_traits.hpp"
#include "cpu/loongarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/loongarch64/jit_generator.hpp"
#include "cpu/loongarch64/jit_primitive_conf.hpp"
#include "cpu/loongarch64/utils/jit_io_helper.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace Xbyak_loongarch64;

struct binary_kernel_t : public jit_generator {
    using op_t = binary_op_t;
    using bcast_t = binary_bcast_t;

    binary_kernel_t(const size_t vlen, const binary_pd_t *pd,
            const jit_binary_conf_t conf, bool tail_kernel = false);
    ~binary_kernel_t() override = default;

    void operator()(jit_binary_call_s *p) { jit_generator::operator()(p); }

    size_t simd_w() const noexcept { return simd_w_; }
    size_t vlen() const noexcept { return vlen_; }

protected:
    size_t get_tail_size() const;

    const size_t vlen_;
    const size_t simd_w_;
    constexpr static int vmm_start_idx_ = 1;
    const binary_pd_t *pd_;
    const jit_binary_conf_t conf_;
    const bool is_tail_kernel_;
    const bool is_src1_outer_dims_tail_;
    const size_t tail_size_;
    const size_t padding_tail_size_;
};

template <cpu_isa_t isa>
struct jit_uni_binary_kernel_t : public binary_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    const XReg &reg_param_ = abi_param1;
    const XReg &reg_src0_ = t0;
    const XReg &reg_src1_ = t1;
    const XReg &reg_dst_ = t2;
    const XReg &reg_offt_src0_ = t3;
    const XReg &reg_outer_dims_range_ = t4;
    const XReg &reg_offt_src1_ = t5;
    const XReg &reg_src1_stride_range_ = t6;
    const XReg &reg_reverse_src1_stride_range_ = t5;
    const XReg &reg_reverse_spat_offt_ = t7;
    const XReg &reg_tmp_ = a7;
    const XReg &reg_tmp1_ = abi_not_param1;
    const XReg &reg_elt_inj_table_ = t7;
    const XReg &reg_off_rhs_postops_ = a2;
    const XReg &reg_scales_src0_ = a3;
    const XReg &reg_scales_src1_ = a4;
    const XReg &reg_offt_dst_ = a2;

    const XReg &tail_opmask_ = a5;
    const XReg &cmp_mask = a6;
    const XReg &full_mask_ = a7;

    const Vmm vmm_tail_vmask_ = Vmm(0);
    const Vmm vreg_sum_scale_ = Vmm(9);
    const VReg xreg_sum_scale_ = VReg(9);
    const Vmm vreg_zero_ = Vmm(10);
    const Vmm vreg_one_ = Vmm(11);
    const Vmm vreg_saturation_ubound_ = Vmm(12);
    const Vmm vreg_bcast_src1_ = Vmm(13);
    const VReg xreg_bcast_src1_ = VReg(13);
    const Vmm vreg_scales_src0_ = Vmm(14);
    const Vmm vreg_scales_src1_ = Vmm(15);

    const Vmm vmm_full_mask_ = Vmm(5);
    const Vmm vmm_tmp_gather_ = Vmm(6);
    const Vmm vmm_indices_ = Vmm(7);
    const Vmm vmm_gathered_src_ = Vmm(8);

    const size_t unroll_regs_ = 4;
    const size_t offt_src0_;
    const size_t offt_src1_;

    static constexpr cpu_isa_t inject_isa = isa;
    io::jit_io_multi_dt_helper_t<Vmm> io_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<inject_isa>>
            postops_injector_;
    const Vmm &elt_inj_opmask_ = Vmm(16);  //TODO

    void init();
    void init_post_ops_injector();
    void apply_postops(int unroll, bool tail);
    void load_kernel_params();
    XReg src0_ptr(size_t offt = 0);
    XReg src1_ptr(size_t offt = 0);
    XReg dst_ptr(size_t offt = 0);
    XReg xreg_addr(const XReg &base, const XReg &off = XReg(DUMMY_IDX),
        const int disp = 0);
    unsigned int cmp_predicate(alg_kind_t alg);
    void compute_cmp_mask(const Vmm &p_mask, const Vmm &vmm_src,
                      const Vmm &compare_operand, int cmp_predicate);
    void perform_op(
            const Vmm &v0, const Vmm &v1, const Vmm &s_src0, const Vmm &s_src1);
    void prepare_isa_kernel();
    void compute_bcast(bool tail);
    void load_src1(const Vmm &vreg_src1, const int offt, bool tail);
    void compute_dst(int unroll, bool tail);
    void forward();
    void forward_over_outer_dims();
    void generate() override;

    jit_uni_binary_kernel_t(const binary_pd_t *pd, const jit_binary_conf_t conf,
            bool tail_kernel = false);
    ~jit_uni_binary_kernel_t() override = default;

    std::map<data_type_t, io::io_saturation_conf_t>
    create_saturation_vmm_map() const;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
