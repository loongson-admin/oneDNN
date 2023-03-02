/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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
#include "cpu/loongarch64/jit_uni_i8i8_pooling.hpp"
#include <math.h>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/loongarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

static bcast_set_t get_supported_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc};
}

static inline dim_t get_offset(
        const memory_desc_wrapper &mdw, int n, int c, int d, int h, int w) {
    switch (mdw.ndims()) {
        case 3: return mdw.blk_off(n, c, w);
        case 4: return mdw.blk_off(n, c, h, w);
        case 5: return mdw.blk_off(n, c, d, h, w);
        default: assert(!"Invalid tensor dimension in pooling");
    }
    return 0;
}

using namespace Xbyak_loongarch64;

using namespace dnnl::impl::utils;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;
using namespace alg_kind;

#define GET_OFF(field) offsetof(call_params_t, field)

struct call_params_t {
    const char *src_i8;
    const char *dst_i8;
    const void *post_ops_binary_rhs_arg_vec;
    size_t kd_range;
    size_t kh_range;
    size_t kw_range;
    float idivider;
    const char *src_safe_access;
    const char *dst_safe_access;
};

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_ker_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_i8i8_pooling_fwd_ker_t)

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    VReg xreg(int idx) const { return VReg(idx); }
    XVReg yreg(int idx) const { return XVReg(idx); }
    Vmm vreg(int idx) const { return Vmm(xreg(idx).getIdx()); }

    XReg reg_param = a0; // Our "unified abi_param1"
    XReg reg_ptr_src_i8 = a2;
    XReg reg_ptr_dst_i8 = a3;
    XReg reg_ptr_maskmovdqu_dst = t2; // store destination - must be rdi

    XReg reg_kd_index
            = t2; // shared with reg_ptr_maskmovdqu_dst; only used before store
    XReg reg_kh_index = a4;
    XReg reg_kw_index = a5;
    XReg reg_kd = a6;
    XReg reg_kh = a7;
    XReg reg_kw = t7;
    XReg c_iter = t0; // shared with reg_mask; only used after mask init

    XReg aux_reg_src_d
            = t1; // shared with reg_tmp; loaded before each accum loop, unused during store
    XReg aux_reg_src_h = t3;
    XReg aux_reg_src_w = t4;

    XReg reg_tmp = t2; // only used during mask init and store
    XReg reg_src_safe_access = t5;
    XReg reg_dst_safe_access = t6;

    // ref to any of XYZ-regs via xreg/yreg/vreg functions
    VReg xmm_tmp = xreg(0); // temp to init vreg_tmp
    Vmm vreg_tmp = vreg(0); // max pooling : holds minimum values for data_type
    Vmm vreg_zeros = vreg(1);
    Vmm vreg_tail = vreg(4);

    int post_op_tail_opmask_idx_ = -1;
    jit_pool_conf_t jpp;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

    enum : int { max_vidx_base = utils::one_of(isa, lsx, lasx) ? 7 : 2 };
    //"avg" pool uses more registers for unrolling.
    enum : int { avg_vidx_base = utils::one_of(isa, lsx, lasx) ? 4 : 2 };

    Vmm max_base_vr(int idx) const { return vreg(max_vidx_base + idx); }
    Vmm avg_base_vr(int idx) const { return vreg(avg_vidx_base + idx); }

    size_t sizeof_src_dt() const { return data_type_size(jpp.src_dt); }
    size_t sizeof_dst_dt() const { return data_type_size(jpp.dst_dt); }

    /* max pooling */
    Vmm vreg_src(int idx) const { return max_base_vr(idx); } // [0    .. ur_c-1]
    Vmm vreg_dst(int idx) const {
        return max_base_vr(jpp.ur_c + idx);
    } // [ur_c .. 2*ur_c-1]

    /* avg pooling */
    // s32 used for processing of s8/u8 data
    // thus we need to take into account ratio of sizes s32/i8 = 4
    static constexpr data_type_t avg_proc_dt = data_type::s32;
    enum : int {
        s32_to_i8_ratio = sizeof(typename prec_traits<avg_proc_dt>::type)
                / sizeof(typename prec_traits<data_type::u8>::type),
        max_num_ll = s32_to_i8_ratio,
        mmx_msk_base_reg = 3
    };

    Vmm vreg_src_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 0 * max_num_ll);
    } // ll: 0..4 [0..3]

    Vmm vreg_dst_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 1 * max_num_ll);
    } // ll: 0..4 [4..7]

    Vmm vreg_dst_f32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 2 * max_num_ll);
    } // ll: 0..4 [8..11]

    VReg mmx_mask(int ll) {
        return VReg(mmx_msk_base_reg + ll);
    }; // ll: 0..4 [Mmx(2)...Mmx(5)]

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    void init_tmp_reg();
    void init_mask();

    //void load_vreg_mask_q(int ll) {};

    void load_src_max_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src_avg_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src(int jj, int ll, int c_tail);

    void store_dst_max_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst_avg_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst(int jj, int ll, int c_tail);

    void compute_avg_step(int ur_c, int c_tail);
    void compute_max_op(const int jj);
    void compute_max_step(int ur_c, int c_tail);
    void compute_step(int ur_c, int c_tail);

    void compute_c_block();
    void generate() override;

    static status_t init_conf(jit_pool_conf_t &jpp, const pooling_pd_t *ppd);

    jit_uni_i8i8_pooling_fwd_ker_t(
            const jit_pool_conf_t &jpp_, const memory_desc_t *dst_md)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
        , jpp(jpp_)
        , postops_injector_(nullptr) {

        if (jpp.with_postops) {

            const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
            const std::size_t c_tail_elems = jpp.c % simd_w;
            post_op_tail_opmask_idx_ = 0;
            if (c_tail_elems) {
                for (int ll = max_num_ll - 1; ll >= 0; ll--) {
                    if (jpp.tail[ll] != 0) {
                        post_op_tail_opmask_idx_ = ll;
                        break;
                    }
                }
            };

            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            static constexpr std::size_t tmp_vmm_injector = 0u;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    tmp_vmm_injector, t8, t7, preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec),
                    memory_desc_wrapper(*dst_md), c_tail_elems,
                    //mask(post_op_tail_opmask_idx_),
                    t2,
                    use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    reg_param, get_supported_bcast_strategies(), rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<isa>>(
                    this, jpp.post_ops, bsp);
        }
    }
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lsx>::load_src_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32)
            load_bytes(vreg_src(jj), aux_reg_src_w, offset, jpp.c_tail * data_type_size(s32));
        else
            load_bytes(vreg_src(jj), aux_reg_src_w, offset, jpp.c_tail);
    } else
        uni_xvld(vreg_src(jj), aux_reg_src_w, offset);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lasx>::load_src_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32)
            load_bytes(vreg_src(jj), aux_reg_src_w, offset, jpp.c_tail * data_type_size(s32));
        else
            load_bytes(vreg_src(jj), aux_reg_src_w, offset, jpp.c_tail);
    } else
        uni_xvld(vreg_src(jj), aux_reg_src_w, offset);
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lsx>::load_src_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    const Vmm &vr_src = vreg_src_s32(jj, ll);
    const XVReg &xvr_src = XVReg(vr_src.getIdx());

    if (jpp.src_dt == s32) {
        if (masked)
            for (int64_t i = 0; i < jpp.c_tail; i++)
                load_bytes(vr_src, aux_reg_src_w, offset, jpp.c_tail * data_type_size(s32));
        else
            uni_xvld(vr_src, aux_reg_src_w, offset);
    } else if (utils::one_of(jpp.src_dt, s8, u8)) {
        if (masked) {
            const int copy_range = math::ilog2q(jpp.tail[ll] + 1);
            load_bytes(vr_src, aux_reg_src_w, offset, copy_range);

            if (jpp.src_dt == s8)
                vext2xv_w_b(xvr_src, xvr_src);
            else
                vext2xv_wu_bu(xvr_src, xvr_src);
        } else {
            if (jpp.src_dt == s8) {
                uni_xvld(vr_src, aux_reg_src_w, offset);
                vext2xv_w_b(xvr_src, xvr_src);
            }
            else {
                uni_xvld(vr_src, aux_reg_src_w, offset);
                vext2xv_wu_bu(xvr_src, xvr_src);
            }
        }
    } else
        assert(!"unsupported src data type");
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lasx>::load_src_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    const Vmm &vr_src = vreg_src_s32(jj, ll);
    if (jpp.src_dt == s32) {
        if (masked)
            for (int64_t i = 0; i < jpp.c_tail; i++)
                load_bytes(vr_src, aux_reg_src_w, offset, jpp.c_tail * data_type_size(s32));
        else
            uni_xvld(vr_src, aux_reg_src_w, offset);
    } else if (utils::one_of(jpp.src_dt, s8, u8)) {
        if (masked) {
            const int copy_range = math::ilog2q(jpp.tail[ll] + 1);
            load_bytes(vr_src, aux_reg_src_w, offset, copy_range);

            if (jpp.src_dt == s8)
                vext2xv_w_b(vr_src, vr_src);
            else
                vext2xv_wu_bu(vr_src, vr_src);
        } else {
            if (jpp.src_dt == s8) {
                uni_xvld(vr_src, aux_reg_src_w, offset);
                vext2xv_w_b(vr_src, vr_src);
            }
            else {
                uni_xvld(vr_src, aux_reg_src_w, offset);
                vext2xv_wu_bu(vr_src, vr_src);
            }
        }
    } else
        assert(!"unsupported src data type");
};


template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::load_src(int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_max_op(jj, ll, offset, masked, jpp.tail[0]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lsx>::store_dst_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32)
            for (int i = 0; i < jpp.c_tail; i++)
                store_bytes(vreg_dst(jj), reg_ptr_dst_i8, offset, jpp.c_tail * data_type_size(s32));
        else if (utils::one_of(jpp.src_dt, u8, s8))
            store_bytes(vreg_dst(jj), reg_ptr_dst_i8, offset, jpp.c_tail);
        else
            assert(!"unsupported src data type");
    } else
        uni_xvst(vreg_dst(jj), reg_ptr_dst_i8, offset);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lasx>::store_dst_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32)
            for (int i = 0; i < jpp.c_tail; i++)
                store_bytes(vreg_dst(jj), reg_ptr_dst_i8, offset, jpp.c_tail * data_type_size(s32));
        else if (utils::one_of(jpp.src_dt, u8, s8))
            store_bytes(vreg_dst(jj), reg_ptr_dst_i8, offset, jpp.c_tail);
        else
            assert(!"unsupported src data type");
    } else
        uni_xvst(vreg_dst(jj), reg_ptr_dst_i8, offset);
}


template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lsx>::store_dst_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk) return;

    const Vmm &vr_dst = vreg_dst_s32(jj, ll);

    if (jpp.src_dt == s32) {
        if (masked)
            store_bytes(vr_dst, reg_ptr_dst_i8, offset, jpp.c_tail * data_type_size(s32));
        else
            uni_xvst(vr_dst, reg_ptr_dst_i8, offset);
    } else if (utils::one_of(jpp.src_dt, s8, u8)) {
        vpickev_h(vr_dst, vr_dst, vr_dst);
        vpickev_b(vr_dst, vr_dst, vr_dst);

        const int copy_range = masked
                ? math::ilog2q(jpp.tail[ll] + 1)
                : cpu_isa_traits<lsx>::vlen / data_type_size(avg_proc_dt);

        store_bytes(vr_dst, reg_ptr_dst_i8, offset, copy_range);
    } else
        assert(!"unsupported src data type");
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lasx>::store_dst_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk) return;

    const Vmm &vr_dst = vreg_dst_s32(jj, ll);
    if (jpp.src_dt == s32) {
        if (masked)
            store_bytes(vr_dst, reg_ptr_dst_i8, offset, jpp.c_tail * data_type_size(s32));
        else
            uni_xvst(vr_dst, reg_ptr_dst_i8, offset);
    } else if (utils::one_of(jpp.src_dt, s8, u8)) {
        xvpickev_h(vr_dst, vr_dst, vr_dst);
        xvpermi_d(vr_dst, vr_dst, 0x58);
        xvpickev_b(vr_dst, vr_dst, vr_dst);

        const int copy_range = masked
                ? math::ilog2q(jpp.tail[ll] + 1)
                : cpu_isa_traits<lasx>::vlen / data_type_size(avg_proc_dt);
        store_bytes(vr_dst, reg_ptr_dst_i8, offset, copy_range);
    } else
        assert(!"unsupported src data type");
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::store_dst(
        int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_max_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lsx>::compute_max_op(const int jj) {
    using namespace data_type;
    switch (jpp.src_dt) {
        case s32: vmax_w(vreg_dst(jj), vreg_src(jj), vreg_src(jj)); break;
        case s8: vmax_b(vreg_dst(jj), vreg_src(jj), vreg_src(jj)); break;
        case u8: vmax_bu(vreg_dst(jj), vreg_src(jj), vreg_src(jj)); break;
        default: assert(!"unsupported src data type");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lasx>::compute_max_op(const int jj) {
    using namespace data_type;
    switch (jpp.src_dt) {
        case s32: xvmax_w(vreg_dst(jj), vreg_dst(jj), vreg_src(jj)); break;
        case s8: xvmax_b(vreg_dst(jj), vreg_dst(jj), vreg_src(jj)); break;
        case u8: xvmax_bu(vreg_dst(jj), vreg_dst(jj), vreg_src(jj)); break;
        default: assert(!"unsupported src data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_max_step(
        int ur_c, int c_tail) {
    Label l_kd, l_kh, l_kw;

    int ih = jpp.ih;
    int iw = jpp.iw;
    int c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++)
        uni_bsll_v(vreg_dst(jj), vreg_tmp, 0);

    add_d(aux_reg_src_d, reg_ptr_src_i8, zero);
    xor_(reg_kd_index, reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        add_d(aux_reg_src_h, aux_reg_src_d, zero);
        xor_(reg_kh_index, reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            add_d(aux_reg_src_w, aux_reg_src_h, zero);
            xor_(reg_kw_index, reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    load_src(jj, 0, c_tail);
                    compute_max_op(jj);
                }
                add_imm(aux_reg_src_w, aux_reg_src_w, c * sizeof_src_dt(), X_TMP_0);
                addi_d(reg_kw_index, reg_kw_index, 1);
                blt(reg_kw_index, reg_kw, l_kw);
            }
            add_imm(aux_reg_src_h, aux_reg_src_h, iw * c * sizeof_src_dt(), X_TMP_0);
            addi_d(reg_kh_index, reg_kh_index, 1);
            blt(reg_kh_index, reg_kh, l_kh);
        }
        add_imm(aux_reg_src_d, aux_reg_src_d, ih * iw * c * sizeof_src_dt(), X_TMP_0);
        addi_d(reg_kd_index, reg_kd_index, 1);
        blt(reg_kd_index, reg_kd, l_kd);
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, 0, c_tail);
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_avg_step(
        int ur_c, int c_tail) {
    using namespace data_type;

    Label l_kd, l_kh, l_kw;

    int ih = jpp.ih;
    int iw = jpp.iw;
    int c = jpp.c;

    const int num_ll = data_type_size(avg_proc_dt) / data_type_size(jpp.src_dt);

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            bool masked = jj == ur_c - 1 && c_tail;
            size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                // Clearing of src reg is not needed as they are written before read
                uni_vpxor(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll),
                        vreg_dst_s32(jj, ll));
            }
        }
    }

    add_d(aux_reg_src_d, reg_ptr_src_i8, zero);
    xor_(reg_kd_index, reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        add_d(aux_reg_src_h, aux_reg_src_d, zero);
        xor_(reg_kh_index, reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            add_d(aux_reg_src_w, aux_reg_src_h, zero);
            xor_(reg_kw_index, reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    for (int ll = 0; ll < num_ll; ll++) {
                        bool masked = jj == ur_c - 1 && c_tail;
                        size_t msk = jpp.tail[ll];
                        if (!(masked && !msk)) {
                            load_src(jj, ll, c_tail);
                            uni_add_w(vreg_dst_s32(jj, ll),
                                    vreg_dst_s32(jj, ll), vreg_src_s32(jj, ll));
                        }
                    }
                }
                add_imm(aux_reg_src_w, aux_reg_src_w, c * sizeof_src_dt(), X_TMP_0);
                addi_d(reg_kw_index, reg_kw_index, 1);
                blt(reg_kw_index, reg_kw, l_kw);
            }
            add_imm(aux_reg_src_h, aux_reg_src_h, iw * c * sizeof_src_dt(), X_TMP_0);
            addi_d(reg_kh_index, reg_kh_index, 1);
            blt(reg_kh_index, reg_kh, l_kh);
        }
        add_imm(aux_reg_src_d, aux_reg_src_d, ih * iw * c * sizeof_src_dt(), X_TMP_0);
        addi_d(reg_kd_index, reg_kd_index, 1);
        blt(reg_kd_index, reg_kd, l_kd);
    }

    static constexpr int vlen_size_elem
            = cpu_isa_traits<isa>::vlen / sizeof(float);
    const auto reg_tmp_postops = t0;//r15;
    const injector_utils::register_preserve_guard_t reg_guard(this,
            jpp.with_binary
                    ? std::initializer_list<Xbyak_loongarch64::XReg> {reg_tmp_postops}
                    : std::initializer_list<Xbyak_loongarch64::XReg> {},
            {});
    if (jpp.with_binary) {
        mov_imm(X_TMP_1, ur_c * num_ll * vlen_size_elem);
        mul_d(reg_tmp_postops, c_iter, X_TMP_1);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            const bool masked = jj == ur_c - 1 && c_tail;
            const size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                const auto &reg_dst_f32 = vreg_dst_f32(jj, ll);
                const auto &reg_dst_s32 = vreg_dst_s32(jj, ll);
                uni_xvffint_s_w(reg_dst_f32, reg_dst_s32);
                uni_fmadd_s(reg_dst_f32, reg_dst_f32, vreg_tmp, vreg_zeros);

                if (jpp.with_postops) {
                    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
                    if (jpp.with_binary) {
                        rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                                reg_dst_f32.getIdx(), reg_tmp_postops);
                        rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                                reg_dst_f32.getIdx(),
                                ll * vlen_size_elem + jj * vlen_size_elem);
                        rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                                reg_dst_f32.getIdx(), reg_tmp_postops);
                        rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                                reg_dst_f32.getIdx(),
                                ll * vlen_size_elem + jj * vlen_size_elem);
                        const bool tail = ll == post_op_tail_opmask_idx_;
                        if (tail && masked)
                            rhs_arg_params.vmm_tail_idx_.emplace(
                                    reg_dst_f32.getIdx());
                    }
                    postops_injector_->compute_vector(
                            reg_dst_f32.getIdx(), rhs_arg_params);
                }

                uni_xvftint_w_s(reg_dst_s32, reg_dst_f32);

                if (jpp.with_postops)
                    if (jpp.dst_dt == u8) {
                        uni_max_w(reg_dst_s32, reg_dst_s32, vreg_zeros);
                    }
                store_dst(jj, ll, c_tail);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_step(int ur_c, int c_tail) {
    switch (jpp.alg) {
        case pooling_max: compute_max_step(ur_c, c_tail); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: compute_avg_step(ur_c, c_tail); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_c_block() {
    Label l_main_loop;

    int nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    int c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    xor_(c_iter, c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop);
        {
            compute_step(ur_c, 0);
            add_imm(reg_ptr_src_i8, reg_ptr_src_i8, ur_c * c_block * sizeof_src_dt(), X_TMP_0);
            add_imm(reg_ptr_dst_i8, reg_ptr_dst_i8, ur_c * c_block * sizeof_dst_dt(), X_TMP_0);
            addi_d(c_iter, c_iter, 1);
            mov_imm(X_TMP_1, c_steps);
            blt(c_iter, X_TMP_1, l_main_loop);
        }
    }

    if (ur_c_tail != 0) { compute_step(ur_c_tail, c_tail); }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lsx>::init_mask() {}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<lasx>::init_mask() {
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_tmp_reg() {
    using namespace data_type;

    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            uni_ld_d(reg_tmp, reg_param, offsetof(call_params_t, idivider));
            uni_replgr2vr_w(vreg_tmp, reg_tmp);
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case s32:
                    mov_imm(reg_tmp, nstl::numeric_limits<int32_t>::lowest());
                    break;
                case s8:
                    mov_imm(reg_tmp, nstl::numeric_limits<int8_t>::lowest());
                    break;
                case u8:
                    mov_imm(reg_tmp, nstl::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            if (jpp.src_dt == s32)
                uni_replgr2vr_w(vreg_tmp, reg_tmp);
            else if (mayiuse(lasx))
                uni_replgr2vr_b(vreg_tmp, reg_tmp);
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::generate() {
    preamble();


#define READ_PARAM(reg, field) \
    ld_d(reg, reg_param, offsetof(call_params_t, field));
    READ_PARAM(reg_ptr_src_i8, src_i8);
    READ_PARAM(reg_ptr_dst_i8, dst_i8);
    READ_PARAM(reg_kd, kd_range);
    READ_PARAM(reg_kh, kh_range);
    READ_PARAM(reg_kw, kw_range);
    READ_PARAM(reg_src_safe_access, src_safe_access);
    READ_PARAM(reg_dst_safe_access, dst_safe_access);

#undef READ_PARAM

    uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

    init_mask();

    init_tmp_reg();

    compute_c_block();

    //emms();
    postamble();

    if (jpp.with_eltwise && postops_injector_)
        postops_injector_->prepare_table();
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(
        jit_pool_conf_t &jpp, const pooling_pd_t *ppd) {
    if (!mayiuse(isa)) return status::unimplemented;

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(ppd->src_md());
    const memory_desc_wrapper dst_d(ppd->dst_md());
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];

    jpp.id = is_3d ? src_d.dims()[ndims - 3] : 1;
    jpp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];

    jpp.od = is_3d ? dst_d.dims()[ndims - 3] : 1;
    jpp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jpp.ow = dst_d.dims()[ndims - 1];

    jpp.stride_d = is_3d ? pd.strides[ndims - 5] : 1;
    jpp.stride_h = is_1d ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];

    jpp.kd = is_3d ? pd.kernel[ndims - 5] : 1;
    jpp.kh = is_1d ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = is_3d ? pd.padding[0][ndims - 5] : 0;
    jpp.t_pad = is_1d ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.alg = pd.alg_kind;

    jpp.src_dt = pd.src_desc.data_type;
    jpp.dst_dt = pd.dst_desc.data_type;

    // data_type items per one vreg on the <isa>
    //     isa == lsx   : 16 bytes -> 16 for s8/u8, 4 for s32
    //     isa == lasx    : 32 bytes -> 32 for s8/u8, 8 for s32
    int simd_w = cpu_isa_traits<isa>::vlen / data_type_size(jpp.src_dt);

    /* Verify that vlen-sized memory access happens within the tensor's
     * size, otherwise load/store will always spill outside the memory
     * boundary.*/
    bool safe_load_n_store = IMPLICATION(utils::one_of(isa, lasx, lsx),
            jpp.mb * jpp.c * nstl::min(jpp.id, jpp.od)
                            * nstl::min(jpp.ih, jpp.oh)
                            * nstl::min(jpp.iw, jpp.ow)
                    >= simd_w);
    if (!safe_load_n_store) return status::unimplemented;

    jpp.c_block = simd_w;
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.c_tail != 0;

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    /* If channel_size is bigger than vlen, we can safely assume there is no
     * underflow of memory boundary, so always perform c_tail and save
     * a couple of compute cycles*/
    jpp.safe_c_tail = jpp.c_tail > 0 && jpp.c >= simd_w;

    switch (jpp.alg) {
        case pooling_max:
            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            // avg_proc_dt (s32) defines granularity (because u8/s8 processed as s32)
            const size_t msk_gran
                    = cpu_isa_traits<isa>::vlen / data_type_size(avg_proc_dt);
            const size_t msk_msk = (1ULL << msk_gran) - 1;
            size_t m = tail_mask;
            for (size_t ll = 0; ll < max_num_ll; ll++) {
                jpp.tail[ll] = m & msk_msk;
                m = m >> msk_gran;
            }
            break;
        }
        default: return status::unimplemented;
    }

    if (!post_ops_ok(jpp, *ppd->attr(), dst_d)) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
bool jit_uni_i8i8_pooling_fwd_ker_t<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    const auto &entries = post_ops.entry_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    if (entries.empty()) return true;

    for (const auto &entry : entries) {
        if (entry.is_eltwise()) {
            const auto alg = entry.eltwise.alg;
            jpp.with_eltwise = eltwise_injector::is_supported(isa, alg);
        } else if (entry.is_binary()) {
            if (entry.binary.src1_desc.data_type == data_type::bf16)
                return false;
            jpp.with_binary = true;
        } else
            return false;
    }

    jpp.with_postops = jpp.with_eltwise || jpp.with_binary;
    jpp.post_ops = post_ops;

    /*
     * TODO Currently eltwise/binary injectors assumes that data in vmm has f32 dt.
     * In max pooling data remains in i8 data type.
     */
    return IMPLICATION(jpp.with_postops, jpp.alg != pooling_max)
            && binary_injector::binary_args_broadcast_supported(
                    post_ops, dst_d, get_supported_bcast_strategies());
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::pd_t::jit_conf() {
    return jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(jpp_, this);
}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::jit_uni_i8i8_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), ker_(nullptr) {}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::~jit_uni_i8i8_pooling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            new jit_uni_i8i8_pooling_fwd_ker_t<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));
    return ker_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_i8 = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst_i8 = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto &jpp = pd()->jpp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);
    /* Calculate when the memory-access will happen outisde of the memory
     * boundary, if so, compute a safe memory access. */
    const auto src_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(src_i8 + src_d.size() - 1)
            - (cpu_isa_traits<isa>::vlen - 1));

    const auto dst_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(dst_i8 + dst_d.size() - 1)
            - (cpu_isa_traits<isa>::vlen - 1));

    parallel_nd(
            jpp.mb, jpp.od, jpp.oh, jpp.ow, [&](int n, int od, int oh, int ow) {
                const int id = nstl::max(od * jpp.stride_d - jpp.f_pad, 0);
                const int ih = nstl::max(oh * jpp.stride_h - jpp.t_pad, 0);
                const int iw = nstl::max(ow * jpp.stride_w - jpp.l_pad, 0);

                const int kd_start
                        = nstl::max(0, jpp.f_pad - od * jpp.stride_d);
                const int kd_end = nstl::min(
                        jpp.kd, jpp.id + jpp.f_pad - od * jpp.stride_d);
                const int kh_start
                        = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
                const int kh_end = nstl::min(
                        jpp.kh, jpp.ih + jpp.t_pad - oh * jpp.stride_h);
                const int kw_start
                        = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
                const int kw_end = nstl::min(
                        jpp.kw, jpp.iw + jpp.l_pad - ow * jpp.stride_w);

                auto p = call_params_t();
                p.src_i8 = &src_i8[get_offset(src_d, n, 0, id, ih, iw)
                        * src_d.data_type_size()];
                p.dst_i8 = &dst_i8[get_offset(dst_d, n, 0, od, oh, ow)
                        * dst_d.data_type_size()];
                p.kd_range = (size_t)(kd_end - kd_start);
                p.kh_range = (size_t)(kh_end - kh_start);
                p.kw_range = (size_t)(kw_end - kw_start);
                p.idivider = 1.0f
                        / ((jpp.alg == pooling_avg_exclude_padding)
                                        ? p.kd_range * p.kh_range * p.kw_range
                                        : jpp.kd * jpp.kh * jpp.kw);
                p.src_safe_access = src_safe_access;
                p.dst_safe_access = dst_safe_access;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*ker_)(&p);
            });
    return status::success;
}

// Explicit instantiation only for supported <isa> values.
template struct jit_uni_i8i8_pooling_fwd_ker_t<lasx>;
template struct jit_uni_i8i8_pooling_fwd_t<lasx>;
//template struct jit_uni_i8i8_pooling_fwd_ker_t<lsx>;
//template struct jit_uni_i8i8_pooling_fwd_t<lsx>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
