/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/injectors/jit_uni_eltwise_injector.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

namespace eltwise_injector {

bool is_isa_supported(cpu_isa_t isa) {
    return mayiuse(isa, lsx);
}

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
            eltwise_logsigmoid, eltwise_mish, eltwise_exp, eltwise_gelu_tanh,
            eltwise_hardswish, eltwise_swish, eltwise_log, eltwise_clip,
            eltwise_clip_v2, eltwise_pow, eltwise_gelu_erf, eltwise_round,
            eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
            eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
            eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd,
            eltwise_clip_v2_use_dst_for_bwd);
}

bool is_supported(cpu_isa_t isa, alg_kind_t alg) {
    return is_isa_supported(isa) && is_alg_supported(alg);
}

} // namespace eltwise_injector

using namespace Xbyak_loongarch64;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    using namespace alg_kind;
    using namespace Xbyak_loongarch64::util;
    preserved_vecs_count = 0;
    vecs_to_preserve = aux_vecs_count();
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin()) + 1;
    start_idx_tail = vmm_idxs.begin();

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = *start_idx_tail;
        ++start_idx_tail;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    // Same logic but to allocate gprs
    size_t preserved_gprs_count = 0;
    for (size_t gpr_idx = 0; gpr_idx <= 30; ++gpr_idx) {
        int _idx = 30 - gpr_idx; // we allocate from the end
        if (preserved_gprs_count < aux_gprs_count()
                && (((unsigned)_idx) != x_table.getIdx()))
            preserved_gpr_idxs[preserved_gprs_count++] = _idx;
    }
    assert(preserved_gprs_count == aux_gprs_count());
    if (save_state_) {
	int sp_step = 0;
	h->st_d(x_table, h->X_SP, -8*(++sp_step));

        for (size_t i = 0; i < preserved_gprs_count; ++i) {
            /* This route has not been tested */
            h->st_d(XReg(preserved_gpr_idxs[i]), h->X_SP, -8*(++sp_step));
        }

	if (sp_step)
            h->sub_imm(h->X_SP, h->X_SP, sp_step * 8, h->X_TMP_0);

        if (preserved_vecs_count)
            h->sub_imm(
                    h->X_SP, h->X_SP, preserved_vecs_count * vlen, h->X_TMP_0);

        size_t i = 0;

        while (i < preserved_vecs_count) {
            int count = 0;
            int ii = i;
            do {
                h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                        h->X_DEFAULT_ADDR);
                i++;
            } while (i < preserved_vecs_count && count < h->x_tmp_vec_size);

            for (int j = 0; j < count; j++)
                h->xvst(XVReg(preserved_vec_idxs[ii++]), h->x_tmp_vec[j], 0);
        }
        load_table_addr();
    }

    assign_regs();
    set_coef_to_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(
        const injector_utils::vmm_index_set_iterator_t start_idx_it) {
    size_t tail_vecs_to_preserve = std::distance(start_idx_it, start_idx_tail);
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        /* This route has not been tested */
        if (idx_off) h->add_imm(h->X_SP, h->X_SP, idx_off * vlen, h->X_TMP_0);

        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
                h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                        h->X_DEFAULT_ADDR);
                i++;
            } while (i < tail_vecs_to_preserve && count < h->x_tmp_vec_size);

            for (int j = 0; j < count; j++)
                h->xvld(XVReg(preserved_vec_idxs[idx_off + ii++]), h->x_tmp_vec[j], 0);
        }
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_) {
        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
                h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                        h->X_DEFAULT_ADDR);
                i++;
            } while (i < tail_vecs_to_preserve && count < h->x_tmp_vec_size);

            for (int j = 0; j < count; j++)
                h->xvst(XVReg(preserved_vec_idxs[idx_off + ii++]), h->x_tmp_vec[j], 0);
        }

        if (idx_off) {
            h->sub_imm(h->X_SP, h->X_SP, idx_off * vlen, h->X_TMP_0);
        }
    }

    assign_regs();
    set_coef_to_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
    using namespace Xbyak_loongarch64::util;
    if (!save_state_) return;

    size_t i = 0;

    while (i < preserved_vecs_count) {
        int count = 0;
        int ii = i;
        do {
            h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                    h->X_DEFAULT_ADDR);
            i++;
        } while (i < preserved_vecs_count && count < h->x_tmp_vec_size);

        for (int j = 0; j < count; j++)
            h->xvld(XVReg(preserved_vec_idxs[ii++]), h->x_tmp_vec[j], 0);
    }

    if (preserved_vecs_count)
        h->add_imm(h->X_SP, h->X_SP, preserved_vecs_count * vlen, h->X_TMP_0);

    int sp_step = 0;
    for (int i = aux_gprs_count() - 1; i >= 0; --i)
        h->ld_d(XReg(preserved_gpr_idxs[i]), h->X_SP, 8*(sp_step++));
    h->ld_d(x_table, h->X_SP, 8*(sp_step++));

    if(sp_step)
	h->add_imm(h->X_SP, h->X_SP, 8*sp_step, h->X_TMP_0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    /* For translation of x64's memory operand instructions */
    z_tmp = Vmm(static_cast<uint32_t>(preserved_vec_idxs[0]));

    vmm_mask = Vmm(preserved_vec_idxs[1]);
    vmm_aux0 = Vmm(preserved_vec_idxs[1]);
    vmm_aux1 = Vmm(preserved_vec_idxs[2]);
    vmm_aux2 = Vmm(preserved_vec_idxs[3]);
    vmm_aux3 = Vmm(preserved_vec_idxs[4]);
    vmm_aux4 = Vmm(preserved_vec_idxs[5]);
    vmm_aux5 = Vmm(preserved_vec_idxs[6]);
    vmm_aux6 = Vmm(preserved_vec_idxs[7]);
    vmm_aux7 = Vmm(preserved_vec_idxs[8]);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::set_coef_to_regs() {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu:
                if (alpha_ != 0.f) table_val(alpha, z_tmp);
                break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: table_val(alpha, vmm_aux4); break;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
            case eltwise_square:
            case eltwise_abs:
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt:
            case eltwise_swish: break;
            case eltwise_linear:
                table_val(alpha, z_tmp);
                table_val(beta, vmm_aux0);
                break;
            case eltwise_bounded_relu: table_val(alpha, z_tmp); break;
            case eltwise_soft_relu:
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
            case eltwise_gelu_tanh:
            case eltwise_log: break;
            case eltwise_clip:
                table_val(alpha, z_tmp);
                table_val(beta, vmm_aux0);
                break;
            case eltwise_pow:
            case eltwise_gelu_erf:
            case eltwise_round: break;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: table_val(alpha, z_tmp); break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu:
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
            case eltwise_square:
            case eltwise_abs:
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt:
            case eltwise_linear:
            case eltwise_bounded_relu:
            case eltwise_soft_relu:
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
            case eltwise_gelu_tanh:
            case eltwise_swish:
            case eltwise_log: break;
            case eltwise_clip:
                table_val(beta, z_tmp);
                table_val(alpha, vmm_aux0);
                break;
            case eltwise_pow:
            case eltwise_gelu_erf: break;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(
        const Vmm &vmm_src, const Vmm &compare_operand, int cmp_predicate) {

    enum {
        EQ_OQ = 0,
        LT_OS = 1,
        LE_OS = 2,
        UNORD_Q = 3,
        NEQ_UQ = 4,
        NLT_US = 5,
        NLE_US = 6,
        ORD_Q = 7,
        EQ_UQ = 8,
        NGE_US = 9,
        NGT_US = 10,
        FALSE_OQ = 11,
        NEQ_OQ = 12,
        GE_OS = 13,
        GT_OS = 14,
        TRUE_UQ = 15,
        EQ_OS = 16,
        LT_OQ = 17,
        LE_OQ = 18,
        UNORD_S = 19,
        NEQ_US = 20,
        NLT_UQ = 21,
        NLE_UQ = 22,
        ORD_S = 23,
        EQ_US = 24,
        NGE_UQ = 25,
        NGT_UQ = 26,
        FALSE_OS = 27,
        NEQ_OS = 28,
        GE_OQ = 29,
        GT_OQ = 30,
        TRUE_US = 31,
    };

    switch (cmp_predicate) {
        case EQ_OQ:
            h->xvfcmp_ceq_s(p_mask, vmm_src, compare_operand);
            break;
        case LT_OS:
            h->xvfcmp_clt_s(p_mask, vmm_src, compare_operand);
            break;
        case LE_OS:
            h->xvfcmp_cle_s(p_mask, vmm_src, compare_operand);
            break;
        case NEQ_UQ:
            h->xvfcmp_cne_s(p_mask, vmm_src, compare_operand);
            break;
        case NLT_US:
            h->xvfcmp_cle_s(p_mask, compare_operand, vmm_src);
            break;
        case NLE_US:
            h->xvfcmp_clt_s(p_mask, compare_operand, vmm_src);
            break;
        case EQ_UQ:
            break;
        case NGE_US:
        case NGT_US:
        case NEQ_OQ:
        case GE_OS:
        case GT_OS:
        case EQ_OS:
        case LT_OQ:
        case LE_OQ:
        case NEQ_US:
        case NLT_UQ:
        case NLE_UQ:
        case EQ_US:
        case NGE_UQ:
        case NGT_UQ:
        case NEQ_OS:
        case GE_OQ:
        case GT_OQ:
        case UNORD_Q:
        case ORD_Q:
        case FALSE_OQ:
        case TRUE_UQ:
        case UNORD_S:
        case ORD_S:
        case FALSE_OS:
        case TRUE_US:
        default: assert(!"Unsupported compare mode"); break;
    }
}

// Uses injector masks objects: p_mask
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
        const Vmm &vmm_dst, const Vmm &src) {
    h->xvbitsel_v(vmm_dst, vmm_dst, src, p_mask);  //p152
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(
        const Vmm &vmm_src) {

    /* Use old algotithm */
    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    compute_cmp_mask(vmm_src, table_val(exp_ln_flt_min_f, z_tmp), _cmp_lt_os);

    h->xvfmin_s(vmm_src, table_val(exp_ln_flt_max_f, z_tmp), vmm_src);
    h->xvfmax_s(vmm_src, table_val(exp_ln_flt_min_f, z_tmp), vmm_src);

    h->xvbsll_v(vmm_aux1, vmm_src, 0);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->xvfmadd_s(vmm_src, vmm_src, table_val(exp_log2ef, z_tmp), table_val(half, z_tmp2));

    h->xvfrintrm_s(vmm_aux2, vmm_src);

    // keep vmm_src = fx for further computations
    h->xvbsll_v(vmm_src, vmm_aux2, 0);

    // x = x - fx * ln2
    h->xvfnmsub_s(vmm_aux1, vmm_aux2, table_val(ln2f, z_tmp), vmm_aux1);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->xvfsub_s(vmm_src, vmm_src, table_val(one, z_tmp));
    h->xvfrintrne_s(vmm_aux2, vmm_src);  //TODO: FPCR default is RNE ?
    h->xvftintrz_w_s(vmm_aux2, vmm_aux2);
    h->xvadd_w(vmm_aux2, vmm_aux2, table_val(exponent_bias, z_tmp));
    h->xvslli_w(vmm_aux2, vmm_aux2, n_mantissa_bits); //Vmm(6) = 2^-fx

    // use vmm_src as tmp vmm_zero when applying mask
    h->xvxor_v(vmm_src, vmm_src, vmm_src);
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux2, vmm_src);

    // compute polynomial
    table_val(exp_pol, vmm_src, 4);
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, table_val(exp_pol, z_tmp, 3));
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, table_val(exp_pol, z_tmp, 2));
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, table_val(exp_pol, z_tmp, 1));
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, table_val(exp_pol, z_tmp, 0));
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, table_val(one, z_tmp));

    // y = y * 2^n
    h->xvfmul_s(vmm_src, vmm_src, vmm_aux2);
    h->xvfmul_s(vmm_src, vmm_src, table_val(two, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    /* Negative values are multiplied by alpha.
     Positive values are not modified. */
    h->xvbsll_v(vmm_aux0, vmm_src, 0);
    h->xvfmin_s(vmm_src, vmm_src, table_val(zero, z_tmp2));
    h->xvfmax_s(vmm_aux0, vmm_aux0, table_val(zero, z_tmp2));
    /* alpha is set to z_tmp in set_coef_to_regs(). */
    h->xvfmadd_s(vmm_src, vmm_src, z_tmp, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvfmax_s(vmm_src, vmm_src, table_val(zero, z_tmp2));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->xvbsll_v(vmm_aux3, vmm_src, 0);

    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    h->xvfsub_s(vmm_src, vmm_src, table_val(one, z_tmp2));
    h->xvfmul_s(vmm_src, vmm_src, vmm_aux4);

    // combine with mask
    h->xvfcmp_clt_s(p_mask, table_val(zero, z_tmp2), vmm_aux3);
    h->xvbitsel_v(vmm_src, vmm_src, vmm_aux3, p_mask);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    // tanh(x) = x(1 + (-1/3)x^2) for |x| < tanh_range
    // tanh(x) = 1 - 2/(1 + exp(2 x)) for otherwise

    const auto &t0 = vmm_src;
    const auto &t1 = vmm_aux1;
    const auto &t3 = vmm_aux3;
    const auto &oneS = vmm_aux4;
    const auto &mask = vmm_aux5; // avoid pred regs used in *conv_kernel*

    table_val(one, oneS);
    // make mask for small x
    h->xvbsll_v(t3, t0, 0);
    h->xvand_v(t1, t0, table_val(positive_mask, z_tmp));
    h->xvslt_w(mask, t1, table_val(tanh_range, z_tmp));

    // 2x
    h->xvfadd_s(t0, t0, t0);
    // exp(2x)
    exp_compute_vector_fwd(t0);
    // 1+exp(2x)
    h->xvfadd_s(t0, t0, oneS);
    // 1/(1+exp(2x))
    // 1st aprox ; a = 1/x + e
    h->xvfrecip_s(t1, t0);

    /* NO WAY for frecps
    // 2nd aprox ; a' = (2 - ax)a = 1/x - e^2 x
    // 3rd aprox ; a'' = (2 - a'x)a'
    */

    /*calculate directly*/
    // 2/(1+exp(2x))
    h->xvfadd_s(t0, t1, t1);
    // 1-2/(1+exp(2x))
    h->xvfsub_s(t0, oneS, t0);

    // tanh(x) = x(1 - x^2/3) for |x| < tanh_range
    h->xvfmul_s(t1, t3, t3);
    h->xvfmadd_s(t1, t1, table_val(tanh_m1d3, z_tmp), oneS);
    h->xvfmul_s(t1, t1, t3);
    // select the correct value according to mask
    h->xvbitsel_v(t0, t0, t1, mask);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvbsll_v(vmm_aux0, vmm_src, 0);

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->xvfmul_s(vmm_src, vmm_src, vmm_src);
    table_val(gelu_tanh_fitting_const, vmm_aux1);
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, table_val(one, z_tmp));
    h->xvfmul_s(vmm_src, vmm_src, vmm_aux0);
    h->xvfmul_s(vmm_src, vmm_src, table_val(gelu_tanh_sqrt_two_over_pi, z_tmp));

    // save x on stack as tanh uses vmm_aux0
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvst(vmm_aux0, h->X_TMP_0, 0);

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvld(vmm_aux0, h->X_TMP_0, 0);
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute 0.5 * x * (1 + tanh(G(x)))
    table_val(half, z_tmp);
    h->xvfmadd_s(vmm_src, vmm_src, z_tmp, z_tmp);
    h->xvfmul_s(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvfmul_s(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvand_v(vmm_src, vmm_src, table_val(positive_mask, z_tmp2));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvfsqrt_s(vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->xvfmadd_s(vmm_src, vmm_src, z_tmp, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvfmax_s(vmm_src, vmm_src, table_val(zero, z_tmp2));
    h->xvfmin_s(vmm_src, vmm_src, z_tmp);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvfmax_s(vmm_src, vmm_src, z_tmp);
    h->xvfmin_s(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // ln(1 + exp(x)) =
    // = ln(1 + exp(n * ln(2) + r)) // divide x by ln(2) and get quot and rem
    // = ln(1 + 2^n * exp(r)) // simplify the exp(n*ln(2)) expression
    // = ln(2 ^ 0 + 2^n * exp(r)) // note 1 = 2^0
    // = ln(2 ^ (n - n) + 2^n * exp(r)) // 2^0 = 2^(n-n)
    // = ln(2 ^ n * (2^-n + exp(r))) // factorize with 2^n
    // = n * ln(2) + ln(2^-n + exp(r)) // take the 2^n factor out of the ln

    // keep src for further computations
    h->xvbsll_v(vmm_aux2, vmm_src, 0);

    h->xvfmin_s(vmm_src, table_val(exp_ln_flt_max_f, z_tmp), vmm_src);
    h->xvfmax_s(vmm_src, table_val(exp_ln_flt_min_f, z_tmp), vmm_src);

    h->xvbsll_v(vmm_aux1, vmm_src, 0);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->xvfmadd_s(vmm_src, vmm_src, table_val(exp_log2ef, z_tmp), table_val(half, z_tmp2));

    // tmp = floorf(fx)
    h->xvfrintrm_s(vmm_aux0, vmm_src);

    // keep vmm_src = fx for further computations
    h->xvbsll_v(vmm_src, vmm_aux0, 0);

    // x = x - fx * ln2
    h->xvfmul_s(vmm_aux0, vmm_aux0, table_val(ln2f, z_tmp));
    h->xvfsub_s(vmm_aux1, vmm_aux1, vmm_aux0);
    // compute exponent polynomial

    table_val(exp_pol, vmm_aux3, 4);
    h->xvfmadd_s(vmm_aux3, vmm_aux3, vmm_aux1, table_val(exp_pol, z_tmp, 3));
    h->xvfmadd_s(vmm_aux3, vmm_aux3, vmm_aux1, table_val(exp_pol, z_tmp, 2));
    h->xvfmadd_s(vmm_aux3, vmm_aux3, vmm_aux1, table_val(exp_pol, z_tmp, 1));
    h->xvfmadd_s(vmm_aux3, vmm_aux3, vmm_aux1, table_val(exp_pol, z_tmp, 0));
    h->xvfmadd_s(vmm_aux3, vmm_aux3, vmm_aux1, table_val(one, z_tmp));

    // We do not count 2^-n here, because n can reach 128 and 2^(-128) is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^-n + exp(r) will be counted (2^-(n-1) + 2*exp(r))/2, because 2^(-127)
    // and 2 are numbers representable in fp32.

    // compute 2^-(n-1)
    // vmm_src now represents n-1
    h->xvfsub_s(vmm_src, vmm_src, table_val(one, z_tmp));
    h->xvfmul_s(vmm_aux1, vmm_src, table_val(minus_one, z_tmp)); //src*(-1)

    h->xvfrintrne_s(vmm_aux1, vmm_aux1);
    h->xvftintrz_w_s(vmm_aux1, vmm_aux1);
    // restore vmm_src to n
    h->xvfadd_s(vmm_src, vmm_src, table_val(one, z_tmp));

    h->xvadd_w(vmm_aux1, vmm_aux1, table_val(exponent_bias, z_tmp));
    h->xvslli_w(vmm_aux1, vmm_aux1, n_mantissa_bits);
    // calculate ln(1 + y)
    h->xvfmadd_s(vmm_aux3, vmm_aux3, table_val(two, z_tmp), vmm_aux1); // 2*exp(r)
    h->xvfmul_s(vmm_aux3, vmm_aux3, table_val(half, z_tmp)); // (2^-(n-1) + 2*exp(r))/2

    // frexp()
    h->xvsrli_w(vmm_src, vmm_aux3, n_mantissa_bits);
    h->xvffint_s_w(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->xvfsub_s(vmm_src, vmm_src, table_val(soft_relu_one_twenty_six, z_tmp));

    // and with mask (to get 0.5 * mantissa)
    h->xvand_v(vmm_aux3, vmm_aux3, table_val(soft_relu_mantissa_sign_mask, z_tmp));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->xvor_v(vmm_aux3, vmm_aux3, table_val(half, z_tmp));
    // y  = y - 1
    h->xvfsub_s(vmm_aux3, vmm_aux3, table_val(one, z_tmp));

    // compute log1p polynomial

    table_val(soft_relu_pol, vmm_aux1, 8);
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 7));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 6));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 5));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 4));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 3));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 2));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 1));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux3, table_val(soft_relu_pol, z_tmp, 0));


    //calculate ln(2) * n
    h->xvfmadd_s(vmm_src, vmm_src, table_val(ln2f, z_tmp), vmm_aux1);
    h->xvfadd_s(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(exp_ln_flt_max_f, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_fwd(
        const Vmm &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->xvbsll_v(vmm_aux3, vmm_src, 0);
    // we store the original sign and make x negative
    h->xvand_v(vmm_aux3, vmm_aux3, table_val(sign_mask, z_tmp));
    h->xvor_v(vmm_src, vmm_src, table_val(sign_mask, z_tmp));

    exp_compute_vector_fwd(vmm_src);

    // dup exp(x)
    h->xvor_v(vmm_aux1, vmm_src, vmm_src);
    // (exp(x) + 1)
    h->xvfadd_s(vmm_aux1, vmm_aux1, table_val(one, z_tmp));
    // y = exp(x) / (exp(x) + 1)
    h->xvfdiv_s(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    table_val(one, vmm_aux2);
    h->xvfsub_s(vmm_aux2, vmm_aux2, vmm_src);

    h->xvand_v(z_tmp, vmm_aux3, vmm_aux3);
    h->xvseqi_w(p_mask, z_tmp, 0);
    h->xvnor_v(p_mask, p_mask, p_mask);

    blend_with_mask(vmm_aux2, vmm_src);

    h->xvor_v(vmm_src, vmm_aux2, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Save src data on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvst(vmm_src, h->X_TMP_0, 0);
    // x*alpha
    h->xvfmul_s(vmm_src, vmm_src, table_val(alpha, z_tmp));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvld(vmm_aux0, h->X_TMP_0, 0);
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->xvfmul_s(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_fwd(
        const Vmm &vmm_src) {

    const auto &t0 = Vmm(IDX(vmm_src));
    const auto &t1 = Vmm(IDX(vmm_aux1));
    const auto &t2 = Vmm(IDX(vmm_aux2));
    const auto &t3 = Vmm(IDX(vmm_aux3));
    const auto &t4 = Vmm(IDX(vmm_aux4));
    //const auto &mask = p_tmp0.s;
    const auto &mask = z_tmp2;
    //const auto &wt0 = h->W_TMP_0;
    const auto &xt0 = h->X_TMP_3;
    auto set_imm = [&](const Vmm &dst, uint32_t imm) {
        h->mov_imm(xt0, imm);
        h->xvreplgr2vr_w(dst, xt0);
        return dst;
    };

    auto gather_table_values = [&](const Vmm &vmm_dst,
                                   const XReg &xr_base,
		                   const Vmm &vmm_idxs) {
        for(int i=0; i<8; i++) {
            h->xvpickve2gr_w(h->X_TMP_1, vmm_idxs, i);
	    h->slli_d(h->X_TMP_1, h->X_TMP_1, 2);
            h->ldx_wu(h->X_TMP_0, xr_base, h->X_TMP_1);
            h->xvinsgr2vr_w(vmm_dst, h->X_TMP_0, i);
        }
    };

    Label tbl1L, tbl2L, exitL;
    const size_t tblL = 5;
    const size_t tblN = 1 << tblL;
    union fi {
        float f;
        uint32_t i;
    };
    h->xvor_v(t4, t0, t0);
    h->xvfmul_s(t0, t0, set_imm(z_tmp, float2int(std::sqrt(2))));
    set_imm(t3, 127 << 23);
    h->xvsub_w(t1, t0, t3);
    h->xvsrai_w(t1, t1, 23); // n
    h->xvffint_s_w(t1, t1); // int -> float
    h->xvand_v(t0, t0, set_imm(z_tmp, 0x7fffff));
    h->xvsrai_w(t2, t0, 23 - tblL); // d
    h->xvslli_w(t2, t2, 2); // d *= 4
    h->xvor_v(t0, t0, t3); // y
    h->xvfmul_s(t0, t0, set_imm(z_tmp, float2int(1 / std::sqrt(2))));
    h->pcaddi(xt0, tbl1L);
    gather_table_values(t3, xt0, t2);
    table_val(one, z_tmp);
    h->xvfmsub_s(t0, t0, t3, z_tmp); // y = y * f - 1
    h->pcaddi(xt0, tbl2L);
    gather_table_values(t2, xt0, t2);
    h->xvfsub_s(t3, t4, z_tmp); // x-1
    set_imm(z_tmp, float2int(1.0 / 32));
    h->xvand_v(z_tmp2, t3, table_val(positive_mask, z_tmp2));
    h->xvfcmp_cle_s(mask, z_tmp2, z_tmp); // 1/32 >= abs(x-1)
    h->xvbitsel_v(t0, t0, t3, mask);
    h->xvxor_v(t2, mask, t2);
    h->xvfmsub_s(t1, t1, set_imm(z_tmp, float2int(std::log(2))), t2); // x = n * log2 - h
    table_val(minus_half, z_tmp);
    h->xvfmadd_s(t2, t2, t0, z_tmp); // f
    table_val(one, z_tmp);
    h->xvfmadd_s(t2, t2, t0, z_tmp); // f * y + 1
    h->xvfmadd_s(t0, t0, t2, t1); // y * f + x
    // check nan/inf
    h->xvxor_v(z_tmp, z_tmp, z_tmp);  //table_val(zero, z_tmp)
    h->xvfcmp_clt_s(mask, t4, z_tmp); // neg
    h->mov_imm(xt0, 0x7fc00000); // qnan
    h->xvreplgr2vr_w(z_tmp, xt0);
    h->xvbitsel_v(t0, t0, z_tmp, mask);
    h->xvxor_v(z_tmp, z_tmp, z_tmp);  //table_val(zero, z_tmp)
    h->xvfcmp_ceq_s(mask, t4, z_tmp); // = 0
    h->mov_imm(xt0, 0xff800000); // -Inf
    h->xvreplgr2vr_w(z_tmp, xt0);
    h->xvbitsel_v(t0, t0, z_tmp, mask);

    h->b(exitL);
    h->L(tbl1L);
    const float *tbl1Addr = (const float *)h->getCurr();
    for (size_t i = 0; i < tblN; i++) {
        fi fi;
        fi.i = (127 << 23) | (i << (23 - tblL));
        fi.f = std::sqrt(2) / fi.f;
        h->dd(fi.i);
    }
    h->L(tbl2L);
    for (size_t i = 0; i < tblN; i++) {
        fi fi;
        fi.f = std::log(tbl1Addr[i]);
        h->dd(fi.i);
    }
    h->L(exitL);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Here we approximate erf(x) using the expression by
    // Abramowitz and Stegun from ``Handbook of Mathematical
    // Functions''
    // NOTE: The performance of this kernel can be further improved
    // with a minimax polynomialial expansion, thereby avoiding division
    // and exp. However, so far, this has costed larger accuracy
    // differences with respect to glibc erf based GELU, in particular
    // ~1.0e-5 -- 1.0e-3 absolute error at s = -5.

    // x = s / sqrt(2)
    h->xvfmul_s(vmm_src, vmm_src, Vmm(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));

    // IMPORTANT: we use vmm_aux3 to save `x` as exp_compute does not use it.
    h->xvor_v(Vmm(IDX(vmm_aux3)), Vmm(IDX(vmm_src)), vmm_src);

    // -exp(-x*x)
    h->xvfmul_s(vmm_src, vmm_src, vmm_src);
    h->xvxor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_src)), Vmm(IDX(table_val(sign_mask, z_tmp))));

    exp_compute_vector_fwd(vmm_src);
    h->xvxor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_src)), Vmm(IDX(table_val(sign_mask, z_tmp))));

    // get sign
    h->xvor_v(Vmm(IDX(vmm_aux0)), Vmm(IDX(vmm_aux3)), vmm_aux3);
    h->xvand_v(Vmm(IDX(vmm_aux0)), Vmm(IDX(vmm_aux0)), Vmm(IDX(table_val(sign_mask, z_tmp))));

    // abs(x)
    h->xvor_v(Vmm(IDX(vmm_aux1)), Vmm(IDX(vmm_aux3)), vmm_aux3);
    abs_compute_vector_fwd(vmm_aux1);

    // t = 1 / (p*x + 1)
    table_val(gelu_erf_approx_const, vmm_aux2);
    h->xvfmadd_s(vmm_aux2, vmm_aux2, vmm_aux1, Vmm(IDX(table_val(one, z_tmp))));

    h->xvfrecip_s(vmm_aux4, vmm_aux2);

    // -exp(-x*x)*t
    h->xvfmul_s(vmm_src, vmm_src, vmm_aux4);

    // compute polynomialial r
    table_val(gelu_erf_pol, vmm_aux1, 4);
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4, Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 3))));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4, Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 2))));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4, Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 1))));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4, Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, Vmm(IDX(table_val(one, z_tmp))));
    h->xvxor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_src)), Vmm(IDX(vmm_aux0)));

    // S = 0.5 * s = x / sqrt^2(2)
    h->xvfmul_s(vmm_aux3, vmm_aux3, Vmm(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux3, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->xvxor_v(z_tmp2, z_tmp2, z_tmp2);
    h->xvfcmp_clt_s(p_mask, z_tmp2, vmm_src);
    h->xvor_v(Vmm(vmm_src.getIdx()), Vmm(z_tmp.getIdx()), z_tmp);
    h->xvbitsel_v(vmm_src, vmm_src, table_val(one, z_tmp2), p_mask);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(vmm_src, table_val(one, z_tmp), _cmp_gt_os);
        // R * alpha, then blend with 1.f
        h->xvfmul_s(vmm_src, vmm_src, Vmm(IDX(table_val(alpha, z_tmp))));
    } else {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
        h->xvfadd_s(vmm_src, vmm_src, Vmm(IDX(table_val(alpha, z_tmp))));
    }
    blend_with_mask(vmm_src, table_val(one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    table_val(one, vmm_aux0);

    h->xvfnmsub_s(vmm_aux0, vmm_src, vmm_src, vmm_aux0);

    h->xvor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_aux0)), vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->xvor_v(Vmm(IDX(vmm_aux0)), Vmm(IDX(vmm_src)), vmm_src);

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    h->xvfmul_s(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    table_val(gelu_tanh_fitting_const_times_three, vmm_aux2);
    h->xvfmadd_s(vmm_aux2, vmm_aux2, vmm_src, Vmm(IDX(table_val(one, z_tmp))));

    table_val(gelu_tanh_fitting_const, vmm_aux1);
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, Vmm(IDX(table_val(one, z_tmp))));
    h->xvfmul_s(vmm_aux0, vmm_aux0, Vmm(IDX(table_val(gelu_tanh_sqrt_two_over_pi, z_tmp))));
    h->xvfmul_s(vmm_src, vmm_src, vmm_aux0);
    h->xvfmul_s(vmm_aux2, vmm_aux2, vmm_aux0);

    // save G2 on stack as tanh uses all available registers
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvst(Vmm(IDX(vmm_aux2)), h->X_TMP_0, 0);

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvld(Vmm(IDX(vmm_aux2)), h->X_TMP_0, 0);
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    // 1) R = G2 * (1 - T) = G2 - G2 * T
    h->xvfnmsub_s(vmm_aux2, vmm_aux2, vmm_src, vmm_aux2);
    // 2) Q = 1 + T
    h->xvfadd_s(vmm_src, vmm_src, Vmm(IDX(table_val(one, z_tmp))));
    // 3) res = Q * (1 + R) = Q + Q * R
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux2, vmm_src);

    h->xvfmul_s(vmm_src, vmm_src, Vmm(IDX(table_val(half, z_tmp))));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 2 * s
    h->xvfadd_s(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_bwd(
        const Vmm &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one, z_tmp));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    table_val(half, vmm_aux0);
    h->xvfdiv_s(vmm_aux0, vmm_aux0, vmm_src);
    h->xvor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_aux0)), vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_bwd(
        const Vmm &vmm_src) {
    table_val(alpha, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // get mask of values > alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(zero, z_tmp));
    // make all negative values zeros
    table_val(zero, z_tmp);
    h->xvfmax_s(vmm_src, vmm_src, z_tmp);

    // everything bigger than 0.f should be 1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    table_val(one, vmm_aux0);

    h->xvfsub_s(vmm_aux0, vmm_aux0, vmm_src);

    h->xvfmul_s(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = alpha * s
    h->xvfmul_s(vmm_src, vmm_src, Vmm(IDX(table_val(alpha, z_tmp))));

    // Save R on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvst(Vmm(IDX(vmm_src)), h->X_TMP_0, 0);

    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvld(Vmm(IDX(vmm_aux0)), h->X_TMP_0, 0);

    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute Q * (1 + R * (1 - Q))
    // T = R * (1 - Q) = R - R * Q
    h->xvfnmsub_s(vmm_aux0, vmm_aux0, vmm_src, vmm_aux0);

    // Q * (1 + T) = Q + Q * T
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux0, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 / s
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->xvfrecip_s(vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_bwd(
        const Vmm &vmm_src) {
    // set result with 1.
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    table_val(one, vmm_aux1);

    // get mask of values > beta and blend with 0.f
    h->xvfcmp_clt_s(p_mask, z_tmp, vmm_src);
    h->xvxor_v(z_tmp2, z_tmp2, z_tmp2);
    h->xvbitsel_v(vmm_aux1, vmm_aux1, z_tmp2, p_mask);
    // get mask of values <= alpha and blend with 0.f
    h->xvfcmp_cle_s(p_mask, vmm_src, vmm_aux0);
    h->xvbitsel_v(vmm_aux1, vmm_aux1, z_tmp2, p_mask);

    h->xvor_v(Vmm(vmm_src.getIdx()), Vmm(vmm_aux1.getIdx()), vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = s / sqrt(2)
    h->xvfmul_s(vmm_src, vmm_src, Vmm(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));

    // Save R on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvst(Vmm(IDX(vmm_src)), h->X_TMP_0, 0);

    // Q = exp(-R*R)
    h->xvfmul_s(vmm_src, vmm_src, vmm_src);
    h->xvxor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_src)), Vmm(IDX(table_val(sign_mask, z_tmp))));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvld(Vmm(IDX(vmm_aux2)), h->X_TMP_0, 0);
    h->xvfmul_s(vmm_aux2, vmm_aux2, Vmm(IDX(table_val(gelu_erf_one_over_sqrt_pi, z_tmp))));
    h->xvfmul_s(vmm_aux2, vmm_aux2, vmm_src);

    // -Q
    h->xvxor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_src)), Vmm(IDX(table_val(sign_mask, z_tmp))));

    // get sign
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvld(Vmm(IDX(vmm_aux0)), h->X_TMP_0, 0);
    h->xvand_v(Vmm(IDX(vmm_aux0)), Vmm(IDX(vmm_aux0)),
            Vmm(IDX(table_val(sign_mask, z_tmp))));

    // abs(x)
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->xvld(Vmm(IDX(vmm_aux1)), h->X_TMP_0, 0);
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    abs_compute_vector_fwd(vmm_aux1);

    // W = 1 / (p * s + 1)
    table_val(gelu_erf_approx_const, vmm_aux3);
    table_val(one, vmm_aux4);
    h->xvfmadd_s(vmm_aux3, vmm_aux3, vmm_aux1, vmm_aux4);
    h->xvfdiv_s(vmm_aux4, vmm_aux4, vmm_aux3);

    // Q * W
    h->xvfmul_s(vmm_src, vmm_src, vmm_aux4);

    // compute polynomial r
    table_val(gelu_erf_pol, vmm_aux1, 4);
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4,
            Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 3))));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4,
            Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 2))));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4,
            Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 1))));
    h->xvfmadd_s(vmm_aux1, vmm_aux1, vmm_aux4,
            Vmm(IDX(table_val(gelu_erf_pol, z_tmp, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->xvfmadd_s(vmm_src, vmm_src, vmm_aux1, Vmm(IDX(table_val(one, z_tmp))));
    h->xvxor_v(Vmm(IDX(vmm_src)), Vmm(IDX(vmm_src)), Vmm(IDX(vmm_aux0)));

    // P = T + 0.5
    h->xvfadd_s(vmm_aux2, vmm_aux2, Vmm(IDX(table_val(half, z_tmp))));
    // res = P + 0.5 * erf
    h->xvfmadd_s(vmm_src, vmm_src, Vmm(IDX(table_val(half, z_tmp))), vmm_aux2);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_gprs_count() {
    using namespace alg_kind;
    switch (alg_) {
        case eltwise_tanh_use_dst_for_bwd:
        case eltwise_tanh:
        case eltwise_gelu_tanh: return 0;
        default: return 0;
    }
    return 0;
};

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::round_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->xvfrintrne_s(vmm_src, vmm_src);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_vecs_count() {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return (alpha_ == 0.f) ? 1 : 3;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 6; /* = exp + 2 */
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return 9;
            case eltwise_square: return 0;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 0;
            case eltwise_linear: return 2;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 5;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return 5; /* = exp + 1 */
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return 4;
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 6;
            case eltwise_clip: return 2;
            case eltwise_gelu_erf: return 6;
            case eltwise_round: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return 1;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 4; /* = exp */
            case eltwise_tanh_use_dst_for_bwd: return 2;
            case eltwise_tanh: return 9;
            case eltwise_square: return 1;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 2;
            case eltwise_linear: return 1;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 5; /* = logistic */
            case eltwise_logistic_use_dst_for_bwd: return 2;
            case eltwise_logistic: return 5; /* = logistic */
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_exp: return 4; /* = exp */
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 1;
            case eltwise_clip: return 3;
            case eltwise_gelu_erf: return 6;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        const injector_utils::vmm_index_set_iterator_t &start_idx_it,
        const injector_utils::vmm_index_set_iterator_t &end_idx_it) {
    using namespace alg_kind;
    std::for_each(start_idx_it, end_idx_it, [&](size_t idx) {
        if (is_fwd_) {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu:
                    if (alpha_ == 0.f)
                        relu_zero_ns_compute_vector_fwd(Vmm(idx));
                    else
                        relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_square:
                    square_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_abs: abs_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_swish: swish_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_log: log_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_clip: clip_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_round: round_compute_vector_fwd(Vmm(idx)); break;
                default: assert(!"unsupported eltwise algorithm");
            }
        } else {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu: relu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_square:
                    square_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_abs: abs_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_swish: swish_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_log: log_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_clip: clip_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(Vmm(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
            h->xvfmul_s(Vmm(IDX(Vmm(idx))), Vmm(IDX(Vmm(idx))),
                    Vmm(IDX(table_val(scale, z_tmp))));
        }
    });
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    const auto &start_idx_it = vmm_idxs.begin();
    const auto &end_idx_it = vmm_idxs.end();
    assert(*start_idx_it < *vmm_idxs.rbegin() + 1
            && *vmm_idxs.rbegin() <= vecs_count);

    injector_preamble(vmm_idxs);
    compute_body(start_idx_tail, end_idx_it);
    injector_preamble_tail(start_idx_it);
    compute_body(start_idx_it, start_idx_tail);
    injector_postamble();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants
    // for a given algorithm, compute right offset for them to be used
    // in table_val() and save the hexadecimal value of them, which
    // will be finally used in prepare_table(). We rely on fact that
    // the map iterator order is deterministic for a fixed map.

    // common values used in several algorithms
    static const table_t common_values {{zero, {0x00000000, true}},
            {half, {0x3f000000, true}}, {one, {0x3f800000, true}},
            {two, {0x40000000, true}}, {minus_one, {0xbf800000, true}},
            {minus_two, {0xc0000000, true}}, {ln2f, {0x3f317218, true}},
            {positive_mask, {0x7fffffff, true}},
            {sign_mask, {0x80000000, true}},
            {minus_half, {0xbf000000, true}},
            {exponent_bias, {0x0000007f, true}}};

    // exp(x) constants
    static const table_t exp_consts {{exp_log2ef, {0x3fb8aa3b, true}},
            {exp_ln_flt_max_f, {0x42b17218, true}},
            {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}} // p5 = 0.00828929059f
    };
    // exp(x) constants2
    static const table_t exp_consts2 {
            {exp_coeff1, {0x3f31721c, true}},
            {exp_coeff2, {0x3e772df2, true}},
            {exp_not_mask17, {~((1u << 17) - 1), true}},
    };

    // tanh(x) constants for four interval approximation
    static const table_t tanh_consts {
            {tanh_range, {0x3d4ccccd, true}},
            {tanh_m1d3, {0xbeaaaaab, true}},
    };

    // soft_relu(x) constants
    static const table_t soft_relu_consts {
            {soft_relu_one_twenty_six, {0x42fc0000, true}},
            {soft_relu_mantissa_sign_mask, {0x807fffff, true}},
    };

    // soft_relu ln(1 + x) polynomial approximation
    static const table_t soft_relu_polynomial {
            {soft_relu_pol, {0xb2b4637d, true}}, // p0 = 0.0000000244f
            {soft_relu_pol, {0x3f7fff8e, true}}, // p1 = 0.9999976971f
            {soft_relu_pol, {0xbf001759, true}}, // p2 = -0.5002478215f
            {soft_relu_pol, {0x3ea70608, true}}, // p3 = 0.3272714505f
            {soft_relu_pol, {0xbea3d7bf, true}}, // p4 = -0.3153830071f
            {soft_relu_pol, {0xbe361d04, true}}, // p5 = -0.1701777461f
            {soft_relu_pol, {0xbfa8f1e6, true}}, // p6 = -1.3254635147f
            {soft_relu_pol, {0xbfe1e812, true}}, // p7 = -1.7971917960f
            {soft_relu_pol, {0xbfc4d30e, true}}, // p8 = -1.5652673123f
    };

    // gelu_tanh(x) constants (formula defined)
    static const table_t gelu_tanh_consts {
            {gelu_tanh_fitting_const, {0x3d372713, true}},
            {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
            {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
    };

    // gelu_erf(x) constants (formula defined)
    static const table_t gelu_erf_consts {
            {gelu_erf_approx_const, {0x3ea7ba05, true}},
            {gelu_erf_one_over_sqrt_two, {0x3f3504f3, true}},
            {gelu_erf_one_over_sqrt_pi, {0x3f106eba, true}},
    };

    // gelu_erf(x) polynomial approximation
    static const table_t gelu_erf_polynomial {
            {gelu_erf_pol, {0x3e827906, true}}, // p1 = 0.254829592f
            {gelu_erf_pol, {0xbe91a98e, true}}, // p2 = -0.284496736f
            {gelu_erf_pol, {0x3fb5f0e3, true}}, // p3 = 1.421413741f
            {gelu_erf_pol, {0xbfba00e3, true}}, // p4 = -1.453152027f
            {gelu_erf_pol, {0x3f87dc22, true}}, // p5 = 1.061405429f
    };

    // This object takes care about which constants and polynomials to include.
    struct need_t {
        need_t(alg_kind_t alg) {
            using namespace alg_kind;
            switch (alg) {
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu:
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp:
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                case eltwise_swish: exp_ = true; break;
                case eltwise_gelu_erf: gelu_erf_ = true; break;
                case eltwise_gelu_tanh:
                    exp_ = true;
                    gelu_tanh_ = true;
                    break;
                case eltwise_log: log_ = true; break;
                case eltwise_soft_relu: soft_relu_ = true; break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh:
                    exp_ = true;
                    tanh_ = true;
                    break;
                default: break;
            }
        }

        bool exp_ = false;
        bool tanh_ = false;
        bool soft_relu_ = false;
        bool gelu_tanh_ = false;
        bool gelu_erf_ = false;
        bool log_ = false;

        bool exp() const { return exp_ || soft_relu_ || gelu_erf_; }
        bool tanh() const { return tanh_ || gelu_tanh_; }
        bool soft_relu() const { return soft_relu_; }
        bool gelu_tanh() const { return gelu_tanh_; }
        bool gelu_erf() const { return gelu_erf_; }
        bool log() const { return log_; }
    };

    need_t need(alg_);

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val,
                                     const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    };

    push_arg_entry_of(scale, float2int(scale_), true);
    push_arg_entry_of(alpha, float2int(alpha_), true);
    push_arg_entry_of(beta, float2int(beta_), true);
    push_entries_of(common_values);
    if (need.exp()) push_entries_of(exp_consts);
    if (need.exp()) push_entries_of(exp_polynomial);
    if (need.exp()) push_entries_of(exp_consts2);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_polynomial);

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen : sizeof(table_entry_val_t);
    }
}

template struct jit_uni_eltwise_injector_f32<lasx>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
