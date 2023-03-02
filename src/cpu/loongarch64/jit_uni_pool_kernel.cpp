/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/loongarch64/jit_uni_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace Xbyak_loongarch64;
using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

static bcast_set_t get_supported_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc};
}

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::~jit_uni_pool_kernel() = default;

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::jit_uni_pool_kernel(
        const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
    , jpp(ajpp) {
    //, bf16_emu_(nullptr) {
    //if (jpp.is_bf16 && !isa_has_bf16(jpp.isa))
    //    bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
    //            bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
    //            bf16_emu_reserv_4, bf16_emu_reserv_5);

    if (jpp.with_postops) {
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        static constexpr int lsx_single_block_size
                = cpu_isa_traits<lsx>::vlen / sizeof(float);
        size_t postop_tail = static_cast<size_t>(jpp.c_tail);
        const bool high_half_block_empty = isa == lsx
                && static_cast<size_t>(jpp.c_tail) > lsx_single_block_size;
        if (high_half_block_empty) postop_tail -= lsx_single_block_size;

        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<std::size_t>(4), this->t7,
                this->a2, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(*dst_md), postop_tail, this->a5,
                use_exact_tail_scalar_bcast};

        const binary_injector::static_params_t bsp {
                reg_param, get_supported_bcast_strategies(), rhs_sp};

        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, jpp.post_ops, bsp);
    }
}

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel<isa>::init_conf(jit_pool_conf_t &jpp,
        memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
        int nthreads) {

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    const int ndims = src_d.ndims();

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.ow = dst_d.dims()[ndims - 1];
    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];

    const bool is_avx512 = false;
    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c_block = is_avx512 ? 16 : 8;

    jpp.alg = pd.alg_kind;

    using namespace format_tag;
    const auto blocked_fmt_tag = utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);

    // src_d.data_type() is equal to dst_d.data_type(). This is checked in init
    auto ncsp_fmt_tag = format_tag::undef;

    const unsigned int L3_cache_size_per_core
            = platform::get_per_core_cache_size(3);
    const size_t block_size
            = ((size_t)jpp.id * jpp.ih * jpp.iw + jpp.od * jpp.oh * jpp.ow)
            * jpp.c_block * types::data_type_size(src_d.data_type());

    const bool forward_ncsp_allowed = !jpp.is_backward
            && jpp.c_without_padding > 3
            && ((jpp.ih > 1 && jpp.iw > 1
                        && block_size <= L3_cache_size_per_core)
                    || src_d.data_type() == data_type::bf16);

    const bool backward_ncsp_allowed = jpp.is_backward
            && ((jpp.ih > 1 && jpp.iw > 1 && jpp.c_without_padding > 1
                        && block_size <= L3_cache_size_per_core)
                    || (src_d.data_type() == data_type::bf16
                            && !(jpp.alg == pooling_max
                                    && block_size > L3_cache_size_per_core)));

    ncsp_fmt_tag = ((forward_ncsp_allowed || backward_ncsp_allowed)
                           && is_avx512 && ndims <= 5)
            ? utils::pick(ndims - 3, ncw, nchw, ncdhw)
            : format_tag::undef;

    const auto nspc_fmt_tag = (ndims <= 5)
            ? utils::pick(ndims - 3, nwc, nhwc, ndhwc)
            : format_tag::undef;

    const auto fmt_tag = src_d.matches_one_of_tag(
            blocked_fmt_tag, ncsp_fmt_tag, nspc_fmt_tag);

    if (!dst_d.matches_tag(fmt_tag)) return status::unimplemented;

    if (fmt_tag == ncsp_fmt_tag) {
        // transform input to blocked f32, call f32 jit, transform result to
        // plain output
        jpp.is_bf16 = false;
        jpp.dt_size = types::data_type_size(data_type::f32);
        jpp.tag_kind = jit_memory_tag_kind_t::ncsp;
    } else {
        jpp.is_bf16 = (src_d.data_type() == data_type::bf16
                && dst_d.data_type() == data_type::bf16);
        jpp.dt_size = types::data_type_size(src_d.data_type());
        jpp.tag_kind = (fmt_tag == nspc_fmt_tag)
                ? jit_memory_tag_kind_t::nspc
                : jit_memory_tag_kind_t::blocked;
    }

    jpp.isa = isa;

    const bool args_ok = true && mayiuse(isa) && (fmt_tag != format_tag::undef)
            && utils::one_of(pd.alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding);
    if (!args_ok) return status::unimplemented;

    jpp.c = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            ? utils::rnd_up(jpp.c_without_padding, jpp.c_block)
            : jpp.c_without_padding;
    if (jpp.tag_kind == jit_memory_tag_kind_t::blocked)
        assert(src_d.padded_dims()[1] == jpp.c);
    jpp.nb_c = utils::div_up(jpp.c, jpp.c_block);
    jpp.c_tail = jpp.c_without_padding % jpp.c_block;
    jpp.is_c_padded = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            && src_d.padded_dims()[1] != jpp.c_without_padding;

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    const int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.ur = 0;
    if (jpp.alg == pooling_max) {
        jpp.ur = is_avx512 ? 16 : 4;

        if (isa == lasx && jpp.c_tail > 0)
            // Additional register needed for tail mask
            jpp.ur -= 1;

        if (jpp.is_training)
            jpp.ur = is_avx512 ? 9 : 3;
        else if (jpp.is_backward)
            jpp.ur = is_avx512 ? 6 : 3;
    } else {
        if (jpp.is_backward)
            jpp.ur = is_avx512 ? 12 : 6;
        else
            jpp.ur = is_avx512 ? 24 : 12;
    }
    if (jpp.is_bf16) {
        jpp.ur = (!isa_has_bf16(jpp.isa))
                ? jpp.ur - 4 // Free registers for AVX512 emulation
                : jpp.ur - 1; // Free register for cvt from bf16 to f32
    }

    // select jpp.ur_bc
    if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
        auto min_ur_w = nstl::max(1, utils::div_up(jpp.l_pad, jpp.stride_w));
        int min_ur_w1 = utils::div_up(right_pad, jpp.stride_w);
        if (min_ur_w < min_ur_w1) { min_ur_w = min_ur_w1; }
        jpp.ur_bc = nstl::min(jpp.nb_c, nstl::max(1, jpp.ur / min_ur_w));
        //take into account threading - to have enough work for parallelization
        float best_eff = 0;
        for (int ur_bc = jpp.ur_bc; ur_bc > 0; ur_bc--) {

            const auto nb2_c = utils::div_up(jpp.nb_c, ur_bc);
            auto work = jpp.is_backward
                    ? (ndims == 5 && jpp.simple_alg ? jpp.od : 1)
                    : (ndims == 5 ? jpp.od : jpp.oh);
            work *= jpp.mb * nb2_c;
            auto eff = (float)work / utils::rnd_up(work, nthreads);
            if (eff > best_eff) {

                best_eff = eff;
                jpp.ur_bc = ur_bc;
            }
            if (eff > 0.9) break; // Heuristic threshold
        }

        //take into account cache re-usage after zeroing on backward
        if (jpp.is_backward && ndims < 5) {
            const int L2 = platform::get_per_core_cache_size(2)
                    / sizeof(jpp.dt_size);
            int ur_bc = nstl::max(1, L2 / (jpp.kh * jpp.iw * jpp.c_block));
            jpp.ur_bc = nstl::min(jpp.ur_bc, ur_bc);
        }

        jpp.ur_bc_tail = jpp.nb_c % jpp.ur_bc;
    } else {
        jpp.ur_bc = 1;
        jpp.ur_bc_tail = 0;
    }
    auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
    if (utils::div_up(jpp.l_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;
    if (utils::div_up(right_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;

    // scratchpad for c_block slice of input and/or output
    using namespace memory_tracking::names;
    const int nscr = nstl::min(dnnl_get_max_threads(), jpp.mb * jpp.nb_c);
    if (jpp.tag_kind == jit_memory_tag_kind_t::ncsp) {
        scratchpad.book(key_pool_src_plain2blocked_cvt,
                jpp.c_block * jpp.id * jpp.ih * jpp.iw * nscr, jpp.dt_size);
        scratchpad.book(key_pool_dst_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr, jpp.dt_size);
        scratchpad.book<uint32_t>(key_pool_ind_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr);
    }

    const auto attr = *ppd->attr();
    if (!post_ops_ok(jpp, attr, dst_d)) return status::unimplemented;

    jpp.post_ops = attr.post_ops_;

    return status::success;
}

static int reg_ind(int shift, int bc, int j, int ur_bc, int ur_w) noexcept {
    return shift * ur_bc * ur_w + bc * ur_w + j;
};

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::prepare_tail_mask() {
    static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};
    mov_imm(tmp_gpr, reinterpret_cast<size_t>(&mask[8 - jpp.c_tail]));
    uni_xvld(vmm_c_tail_mask, tmp_gpr, 0);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::put_one_in_vmm() {
    mov_imm(tmp_gpr, 1);
    uni_broadcast_reg_val(tmp_gpr.getIdx(), vmm_one.getIdx());
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::uni_broadcast_reg_val(
        const int reg_idx, const int vmm_idx) {
    uni_replgr2vr_w(Vmm(vmm_idx), XReg(reg_idx));
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::push_vmm_val(const int idx) {
    Vmm val_to_store(idx);
    addi_d(sp, sp, -1 * val_to_store.getBit());
    uni_xvst(val_to_store, sp, 0);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::pop_vmm_val(const int idx) {
    Vmm val_to_load(idx);
    uni_xvld(val_to_load, sp, 0);
    addi_d(sp, sp, val_to_load.getBit());
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::load(const int idx,
        const XReg &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {

    if (is_c_tail_proccessing && !jpp.is_c_padded) {
        if (isa == lasx) {
            uni_xvld(Vmm(idx), reg_ptr, offset);
            uni_xvand_v(Vmm(idx), Vmm(idx), vmm_c_tail_mask);
        } else if (isa == lsx) {
            for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++) {
                uni_ld_w(tmp_gpr, reg_ptr, offset + i * jpp.dt_size);
                vinsgr2vr_w(VReg(idx), tmp_gpr, i);
            }
        }
    } else {
        uni_xvld(Vmm(idx), reg_ptr, offset);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::store(const int idx,
        const XReg &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {

    if (is_c_tail_proccessing) {
        if (!jpp.is_c_padded) {
            store_bytes(Vmm(idx), reg_ptr, offset, jpp.c_tail * sizeof(float));
        } else {
            if (jpp.with_postops) {
                uni_vpxor(ymm_tmp_1, ymm_tmp_1, ymm_tmp_1);
                xvbitsel_v(Vmm(idx), ymm_tmp_1, Vmm(idx), vmm_c_tail_mask);
            }
            uni_xvst(Vmm(idx), reg_ptr, offset);
        }
    } else
        uni_xvst(Vmm(idx), reg_ptr, offset);
}

template <cpu_isa_t isa>
bool jit_uni_pool_kernel<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    const auto &entries = post_ops.entry_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    if (!jpp.is_backward) {
        for (const auto &entry : entries) {
            if (entry.is_eltwise()) {
                const auto alg = entry.eltwise.alg;
                jpp.with_eltwise = eltwise_injector::is_supported(isa, alg);
            } else if (entry.is_binary()) {
                if (false && entry.binary.src1_desc.data_type == data_type::bf16)
                    return false;

                jpp.with_binary = true;
            } else
                return false;
        }

        jpp.with_postops = jpp.with_eltwise || jpp.with_binary;
    }

    return binary_injector::binary_args_broadcast_supported(
            post_ops, dst_d, get_supported_bcast_strategies());
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::apply_postops(int ur_bc, int ur_w, int c_block,
        const std::function<bool(int, bool)> &is_tail_predicate) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    const int end_idx = vmm_idx_upper_bound() + 1;
    const int start_idx = end_idx - (ur_bc * ur_w);
    const bool lsx_postops_disabled
            = isa == lsx && disable_postops_when_sse_high_half_processed_;

    if (jpp.with_binary && !lsx_postops_disabled) {

        static constexpr int lsx_simd_w
                = cpu_isa_traits<lsx>::vlen / sizeof(float);
        const int sse_elem_off = sse_high_half ? lsx_simd_w : 0;

        for (int jj = 0; jj < ur_w; jj++) {
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto vmm_idx
                        = vreg(reg_ind(0, bci, jj, ur_bc, ur_w)).getIdx();
                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                        vmm_idx, ptr_a(reg_param, GET_OFF(c_elem_off)));
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        vmm_idx, bci * c_block + sse_elem_off);
                if (is_tail_predicate
                        && is_tail_predicate(
                                bci, true /*process_with_postops*/)) {
                    rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
                }
            }
        }
    }
    postops_injector_->compute_vector_range(start_idx, end_idx, rhs_arg_params);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::maybe_recalculate_divisor(
        int jj, int ur_w, int pad_l, int pad_r, bool with_c_tail_proccessing) {
    if (jpp.alg == pooling_avg_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= nstl::max(0, pad_l - jj * stride_w);
        non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj) * stride_w);

        if (non_zero_kw != prev_kw) {
            mov_imm(tmp_gpr, float2int((float)non_zero_kw));
            uni_replgr2vr_w(vmm_tmp, tmp_gpr);
            if (with_c_tail_proccessing && isa == lasx) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                uni_broadcast_reg_val(
                        reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
            }
            uni_fmul_s(vmm_tmp, vmm_tmp, vmm_ker_area_h);
            if (with_c_tail_proccessing && isa == lasx) {
                pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::avg_step(int ur_w, int ur_bc, int pad_l,
        int pad_r, bool with_c_tail_proccessing) {

    auto iw = jpp.iw;
    auto kw = jpp.kw;
    auto stride_w = jpp.stride_w;
    auto c_block = jpp.c_block;
    auto dt_size = jpp.dt_size;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc,
                                            bool process_with_postops = false) {
        if (isa == lsx && (!jpp.is_c_padded || process_with_postops)) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward)
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
        for (int bci = 0; bci < ur_bc; bci++) {
            const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
            auto accvr = vreg(accr_i);
            if (jpp.is_backward) {
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
                load(accvr.getIdx(), reg_output, output_offset,
                        is_tail_processing(bci));
                uni_fdiv_s(accvr, accvr, vmm_tmp);
            } else {
                uni_vpxor(accvr, accvr, accvr);
            }
        }
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        push_xreg(reg_input);
        push_xreg(reg_output);
        add_d(aux_reg_input_d, reg_input, zero);
        ld_d(ki, reg_param, GET_OFF(kd_padding));
        L(kd_label);
        add_d(aux_reg_input, aux_reg_input_d, zero);
    } else {
        add_d(aux_reg_input, reg_input, zero);
    }

    xor_(kj, kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);

            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                auto inpvr = vreg(inpr_i);
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = dt_size * aux_input_offset;
                if (jpp.is_backward) {
                    load(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                    uni_fadd_s(inpvr, inpvr, accvr);

                    store(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                } else {
                    load(vmm_tmp_1.getIdx(), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                    uni_fadd_s(accvr, accvr, vmm_tmp_1);
                }
            }
        }
        add_imm(aux_reg_input, aux_reg_input, jpp.dt_size * iw * c_off, tmp_gpr);
        addi_d(kj, kj, 1);
        blt(kj, reg_kh, kh_label);
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off, tmp_gpr);
        addi_d(ki, ki, -1);
        blt(zero, ki, kd_label);
        pop_xreg(reg_output);
        pop_xreg(reg_input);
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto accvr = vreg(accr_i);
                uni_fdiv_s(accvr, accvr, vmm_tmp);
            }
        }

        if (jpp.with_postops)
            apply_postops(ur_bc, ur_w, c_block, is_tail_processing);

        for (int jj = 0; jj < ur_w; jj++) {
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto output_offset
                        = dt_size * (jj * c_off + bci * c_block);

                store(reg_idx(accr_i), reg_output, output_offset,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_fwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto is_tail_processing = [&](int bc, bool process_with_postops = false) {
        if (isa == lsx && (!jpp.is_c_padded || process_with_postops)) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    mov_imm(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));
    uni_replgr2vr_w(vmm_tmp, tmp_gpr);

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
        uni_bsll_v(accvr, vmm_tmp, 0);
        if (jpp.is_training) {
            const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w));
            uni_vpxor(indvr, indvr, indvr);
        }
    }
    if (jpp.is_training) {
        uni_replgr2vr_w(vmm_k_offset, reg_k_shift);
    }
    if (jpp.ndims == 5) {
        push_xreg(reg_input);
        push_xreg(reg_output);
        add_d(aux_reg_input_d, reg_input, zero);
        ld_d(ki, reg_param, GET_OFF(kd_padding));
        L(kd_label);
        add_d(aux_reg_input, aux_reg_input_d, zero);
    } else {
        add_d(aux_reg_input, reg_input, zero);
    }
    xor_(kj, kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i);
                const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w));
                const auto cvtvr = vreg(reg_ind(3, bci, jj, ur_bc, ur_w));
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = jpp.dt_size * aux_input_offset;
                load(reg_idx(inpr_i), aux_reg_input, input_offset,
                        is_tail_processing(bci));

                if (isa == lasx) {
                    uni_fcmp_clt_s(cvtvr, accvr, inpvr);
                    xvbitsel_v(accvr, accvr, inpvr, cvtvr);
                    if (jpp.is_training)
                        xvbitsel_v(indvr, indvr, vmm_k_offset, cvtvr);
                }
            }
            if (jpp.is_training) {
                if (with_c_tail_proccessing && isa == lasx) {
                    push_vmm_val(vmm_c_tail_mask.getIdx());
                    put_one_in_vmm();
                }

                xvadd_w(vmm_k_offset, vmm_k_offset, vmm_one);

                if (with_c_tail_proccessing && isa == lasx)
                    pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
        }
        add_imm(aux_reg_input, aux_reg_input, jpp.dt_size * iw * c_off, tmp_gpr);
        addi_d(kj, kj, 1);
        blt(kj, reg_kh, kh_label);
    }

    if (jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off, tmp_gpr);
        if (jpp.is_training) {
            ld_d(tmp_gpr, reg_param, GET_OFF(kd_padding_shift));
            uni_replgr2vr_w(vmm_tmp, tmp_gpr);

            xvadd_w(vmm_k_offset, vmm_k_offset, vmm_tmp);
        }

        addi_d(ki, ki, -1);
        blt(zero, ki, kd_label);
        pop_xreg(reg_output);
        pop_xreg(reg_input);
    }

    if (with_c_tail_proccessing && jpp.is_c_padded && isa == lsx)
        addi_d(tmp_gpr, zero, 0); // needed zero to fill padded tail

    if (jpp.with_postops)
        apply_postops(ur_bc, ur_w, c_block, is_tail_processing);

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        const auto output_offset = jpp.dt_size * (jj * c_off + bci * c_block);

        store(reg_idx(accr_i), reg_output, output_offset,
                is_tail_processing(bci));

        if (jpp.is_training) {
            const size_t step_index = (jj * c_off + bci * c_block)
                    * types::data_type_size(jpp.ind_dt);

            const auto indr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
            auto vr = vreg(indr_i);
            if (jpp.ind_dt == data_type::u8) {
                auto xr = xreg(indr_i);
                if (isa == lasx) {
                    auto yr = yreg(indr_i);
                    if (is_tail_processing(bci) && !jpp.is_c_padded) {
                        const int max_nr_of_vals
                                = jpp.c_tail > (jpp.c_block / 2)
                                ? (jpp.c_block / 2)
                                : jpp.c_tail;
                        for (int i = 0; i < max_nr_of_vals; ++i) {
                            // bytes which should be stored are located in
                            // least significant bits(8 to be precise) of 32 bits parts
                            // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                            uni_xvstelm_b(xr, reg_index, step_index + i, 4 * i);
                        }

                        if (jpp.c_tail > (jpp.c_block / 2)) {
                            XVReg higher_128bits(vmm_mask.getIdx());
                            xvpermi_q(higher_128bits, yr, 0x31);
                            for (int i = 0; i < jpp.c_tail - (jpp.c_block / 2);
                                    ++i) {
                                // bytes which should be stored are located in
                                // least significant bits(8 to be precise) of 32 bits parts
                                // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                                uni_xvstelm_b(vmm_mask, reg_index,
                                    step_index + (jpp.c_block / 2) + i, 4 * i);
                            }
                        }
                    } else {
                        if (is_tail_processing(bci)) {
                            assert(jpp.is_c_padded);
                            xvand_v(yr, yr, vmm_c_tail_mask);
                        }
                        if (jj == 0) {
                            uni_replgr2vr_w(vmm_tmp, reg_shuf_mask);
                        }
                        if (mayiuse(lasx)) {
                            xvshuf_b(yr, yr, yr, vmm_tmp);
                            uni_xvstelm_w(xr, reg_index, step_index, 0);
                            uni_xvstelm_w(yr, reg_index, step_index + (jpp.c_block / 2), 4);
                        }
                    }
                }
            } else {
                store(vr.getIdx(), reg_index, step_index,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_bwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc) {
        if (isa == lsx) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half)
                            || (jpp.c_tail == (jpp.c_block / 2) && sse_high_half
                                    && jpp.is_c_padded));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto outr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        auto out_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        load(reg_idx(outr_i), reg_output, out_offset, is_tail_processing(bci));
        const size_t step_index = (jj * c_off + bci * c_block)
                * types::data_type_size(jpp.ind_dt);

        const auto indr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
        auto indvr = vreg(indr_i);
        if (jpp.ind_dt == data_type::u8) {
            auto indxr = xreg(indr_i);
            if (isa == lasx) {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
                    for (int i = 0; i < jpp.c_tail; i++) {
                        uni_ld_b(tmp_gpr, reg_index, step_index + i);
                        vinsgr2vr_b(indxr, tmp_gpr, i);
                    }
                } else {
                    uni_xvldrepl_d(indxr, reg_index, step_index);
                }
                if (!mayiuse(lasx)) {
                    avx_pmovzxbd(indvr, indxr, xmm_tmp);
                } else {
                    vext2xv_wu_bu(indvr, indvr);
                }
            }
        } else {
            load(indvr.getIdx(), reg_index, step_index,
                    is_tail_processing(bci));
        }
    }
    uni_replgr2vr_w(vmm_k_offset, reg_k_shift);

    if (jpp.simple_alg && jpp.ndims == 5) {
        push_xreg(reg_input);
        push_xreg(reg_output);
        if (isa == lsx) {
            // Save a0 since it is used in maskmovdqu
            assert(dst_ptr == a0);
            push_xreg(dst_ptr);
        }
        add_d(aux_reg_input_d, reg_input, zero);
        ld_d(ki, reg_param, GET_OFF(kd_padding));
        ld_d(reg_kd_pad_shift, reg_param, GET_OFF(kd_padding_shift));
        L(kd_label);
        add_d(aux_reg_input, aux_reg_input_d, zero);
    } else {
        add_d(aux_reg_input, reg_input, zero);
    }

    xor_(kj, kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto outvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto indvr = vreg(reg_ind(1, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i);
                const auto cvtvr = vreg(reg_ind(3, bci, jj, ur_bc, ur_w));
                int aux_inp_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_inp_offset >= iw * c_off) continue;
                int inp_offset = jpp.dt_size * aux_inp_offset;
                load(reg_idx(inpr_i), aux_reg_input, inp_offset,
                        is_tail_processing(bci));

                if (isa == lasx) {
                    if (mayiuse(lasx)) {
                        xvseq_w(cvtvr, indvr, vmm_k_offset);
                    } else {
                        avx_pcmpeqd(cvtvr, indvr, vmm_k_offset, xmm_tmp);
                    }
                    uni_fadd_s(inpvr, inpvr, outvr);
                    if (is_tail_processing(bci)) {
                        uni_xvand_v(cvtvr, cvtvr, vmm_c_tail_mask);
                    }
                    Label end_cond_move[8];
                    for (int mi = 0; mi < 8; ++mi) {
                        xvpickve2gr_w(tmp_gpr, cvtvr, mi);
                        beqz(tmp_gpr, end_cond_move[mi]);
                        uni_xvstelm_w(inpvr, aux_reg_input, inp_offset + mi * jpp.dt_size, mi);
                        L(end_cond_move[mi]);
                    }
                }
            }

            if (with_c_tail_proccessing && isa == lasx) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                put_one_in_vmm();
            }

            xvadd_w(vmm_k_offset, vmm_k_offset, vmm_one);

            if (with_c_tail_proccessing && isa == lasx)
                pop_vmm_val(vmm_c_tail_mask.getIdx());
        }
        add_imm(aux_reg_input, aux_reg_input, jpp.dt_size * iw * c_off, tmp_gpr);
        addi_d(kj, kj, 1);
        blt(kj, reg_kh, kh_label);
    }
    if (jpp.simple_alg && jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off, tmp_gpr);

        uni_replgr2vr_w(vmm_tmp, reg_kd_pad_shift);
        xvadd_w(vmm_k_offset, vmm_k_offset, vmm_tmp);

        addi_d(ki, ki, -1);
        blt(zero, ki, kd_label);
        if (isa == lsx) {
            // Save a0 since it is used in maskmovdqu
            assert(dst_ptr == a0);
            pop_xreg(dst_ptr);
        }
        pop_xreg(reg_output);
        pop_xreg(reg_input);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::zero_diff_src(
        int ur_bc, bool with_c_tail_proccessing) {
    const int c_off = (jpp.tag_kind == jit_memory_tag_kind_t::nspc)
            ? jpp.c
            : jpp.c_block;

    Label l_skip, l_ih_loop, l_id_loop;

    auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    ld_d(reg_zero_id, reg_param, GET_OFF(zero_id));
    beqz(reg_zero_id, l_skip);

    ld_d(reg_zero_ih, reg_param, GET_OFF(zero_ih));
    beqz(reg_zero_ih, l_skip);

    ld_d(reg_zero_ptr, reg_param, GET_OFF(zero_ptr));

    Vmm vzero = vmm_tmp;
    uni_vpxor(vzero, vzero, vzero);

    const int width_size = jpp.iw * c_off * jpp.dt_size;

    auto aux_reg_zero_ptr = tmp_gpr;

    L(l_id_loop);
    {
        add_d(aux_reg_zero_ptr, reg_zero_ptr, zero);
        add_d(aux_reg_zero_ih, reg_zero_ih, zero);
        L(l_ih_loop);
        {
            const int step = c_off * jpp.dt_size;

            // TODO: maybe a big code generated here
            for_(int i = 0; i < width_size; i += step)
            for (int bci = 0; bci < ur_bc; bci++) {
                const int offs = i + bci * jpp.c_block * jpp.dt_size;

                store(vzero.getIdx(), reg_zero_ptr, offs,
                            is_tail_processing(bci));
            }
            add_imm(reg_zero_ptr, reg_zero_ptr, width_size, tmp_gpr);
            addi_d(aux_reg_zero_ih, aux_reg_zero_ih, -1);
            bnez(aux_reg_zero_ih, l_ih_loop);
        }
        add_d(reg_zero_ptr, aux_reg_zero_ptr, zero);
        add_imm(reg_zero_ptr, reg_zero_ptr, width_size * jpp.ih, tmp_gpr);
        addi_d(reg_zero_id, reg_zero_id, -1);
        bnez(reg_zero_id, l_id_loop);
    }

    L(l_skip);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::generate() {

    this->preamble();

    Label idx_table;

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;

    int vlen = cpu_isa_traits<isa>::vlen;

    ld_d(reg_input, reg_param, GET_OFF(src));
    ld_d(reg_output, reg_param, GET_OFF(dst));
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
        ld_d(reg_index, reg_param, GET_OFF(indices));
    ld_d(reg_kh, reg_param, GET_OFF(kh_padding));
    ld_d(reg_k_shift, reg_param, GET_OFF(kh_padding_shift));
    ld_d(reg_ker_area_h, reg_param, GET_OFF(ker_area_h));
    ld_d(reg_nbc, reg_param, GET_OFF(ur_bc));

    int r_pad
            = nstl::max(0, calculate_end_padding(l_pad, ow, iw, stride_w, kw));

    auto process_oi = [&](int ur_w, int ur_bc, int lpad, int rpad,
                              bool with_c_tail_proccessing,
                              bool inc_reg = true) {
        step(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);

        if (isa == lsx) {
            if (with_c_tail_proccessing && jpp.c_tail <= (jpp.c_block / 2)) {

                // In nspc format in case of c tail processing if c tail is
                // equal or lower than 4 we don't have to process
                // last high half block, because it doesn't exist
                if (!jpp.is_c_padded) ur_bc -= 1;
                /*
                 * In case of c_tail_processing if c_tail is equal or lower than 4
                 * applying postops never make sense. In case of blocked format it
                 * can cause overwriting zero padding or segfault because the element
                 * corresponding to the piece with padded zeros doesn't exist in binary
                 * postops arg1 tensor (nchw format) in per_oc bcast strategy.
                 */
                disable_postops_when_sse_high_half_processed_
                        = jpp.tag_kind == jit_memory_tag_kind_t::blocked;
            }
            sse_high_half = true;
            step_high_half(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);
            sse_high_half = false;
            disable_postops_when_sse_high_half_processed_ = false;
        }

        if (!inc_reg) return;

        auto dt_size = jpp.dt_size;
        auto shift = (isa == lsx) ? vlen : 0;
        add_imm(reg_input, reg_input, dt_size * (ur_w * stride_w - lpad) * c_off - shift, tmp_gpr);
        add_imm(reg_output, reg_output, dt_size * ur_w * c_off - shift, tmp_gpr);
        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            auto ishift = (isa == lsx) ? jpp.c_block / 2 : 0;
            auto ind_dt_size = types::data_type_size(jpp.ind_dt);
            add_imm(reg_index, reg_index, (ur_w * c_off - ishift) * ind_dt_size, tmp_gpr);
        }
    };

    auto perform_ker = [&](int ur_bc, bool with_c_tail_processing) {
        prev_kw = 0; // re-initialize this value for avg steps

        if (jpp.is_backward && jpp.simple_alg)
            zero_diff_src(ur_bc, with_c_tail_processing);

        if (jpp.alg == pooling_avg_exclude_padding
                && (!with_c_tail_processing || (isa != lasx))) {
            // vmm_ker_area_h and vmm_c_tail_mask are stored in one register
            // so when vmm_c_tail_mask is used we need to load vmm_ker_area_h
            // exactly where this information is needed with the
            // vmm_c_tail_mask information being saved first
            uni_broadcast_reg_val(
                    reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
        }

        if (jpp.alg == pooling_avg_include_padding) {
            mov_imm(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));
            uni_replgr2vr_w(vmm_tmp, tmp_gpr);
        }

        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            if (!with_c_tail_processing || (isa != lasx)) {
                // The same situation as above(vmm_ker_area_h).
                put_one_in_vmm();
            }

            if (isa == lasx) { mov_imm(reg_shuf_mask, 0x0c080400); }
        }

        auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
        auto ur_w_tail = jpp.ow % ur_w;

        int n_oi = ow / ur_w;

        int r_pad1
                = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w, kw);
        if (r_pad1 > 0) n_oi--;

        if (l_pad > 0) {
            n_oi--;
            if (n_oi < 0 && r_pad1 > 0)
                process_oi(ur_w, ur_bc, l_pad, r_pad1, with_c_tail_processing);
            else
                process_oi(ur_w, ur_bc, l_pad, 0, with_c_tail_processing);
        }

        xor_(oi_iter, oi_iter, oi_iter);
        if (n_oi > 0) {
            Label ow_loop;
            L(ow_loop);
            {
                process_oi(ur_w, ur_bc, 0, 0, with_c_tail_processing);

                addi_d(oi_iter, oi_iter, 1);
                mov_imm(tmp_gpr, n_oi);
                blt(oi_iter, tmp_gpr, ow_loop);
            }
        }

        if (r_pad1 > 0 && n_oi >= 0)
            process_oi(ur_w, ur_bc, 0, r_pad1, with_c_tail_processing);

        if (ur_w_tail != 0)
            process_oi(
                    ur_w_tail, ur_bc, 0, r_pad, with_c_tail_processing, false);
    };
    Label ur_bc_tail_label, c_tail_processing_label, finish_label;

    if (jpp.ur_bc_tail > 0) {
        mov_imm(tmp_gpr, jpp.ur_bc);
        bne(reg_nbc, tmp_gpr, ur_bc_tail_label);
    } else if (jpp.c_tail != 0) {
        // ur_bc contains number of channel blocks to processing
        // b_c contains number of channel blocks already processed
        // If reg_nbc + tmp_gpr == jpp.nb_c then this is
        // information that probably channel tail processing will be needed.
        ld_d(tmp_gpr, reg_param, GET_OFF(b_c));
        add_d(tmp_gpr, tmp_gpr, reg_nbc);
        mov_imm(X_TMP_1, jpp.nb_c);
        beq(tmp_gpr, X_TMP_1, c_tail_processing_label);
    }

    perform_ker(jpp.ur_bc, false);

    if (jpp.ur_bc_tail > 0) {
        b(finish_label);

        // If ur_bc_tail exists then we know that this is
        // last set of blocks to process and we need
        // care of c tail processing if number of channels
        // is not divided by number of channels in block
        L(ur_bc_tail_label);
        if (jpp.c_tail != 0) prepare_tail_mask();
        perform_ker(jpp.ur_bc_tail, jpp.c_tail != 0);

        L(finish_label);
    } else if (jpp.c_tail != 0) {
        b(finish_label);

        L(c_tail_processing_label);
        prepare_tail_mask();
        perform_ker(jpp.ur_bc, true);

        L(finish_label);
    }

    this->postamble();

    if (jpp.with_eltwise && postops_injector_)
        postops_injector_->prepare_table();
}

template struct jit_uni_pool_kernel<lasx>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
