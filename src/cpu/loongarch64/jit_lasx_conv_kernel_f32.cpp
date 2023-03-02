/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/loongarch64/injectors/injector_utils.hpp"
#include "cpu/loongarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/loongarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/loongarch64/jit_lasx_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak_loongarch64;

jit_lasx_conv_fwd_kernel_f32::jit_lasx_conv_fwd_kernel_f32(
        const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, lasx)
    , jcp(ajcp)
    , attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 15;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const size_t tail_size = jcp.oc_without_padding % isa_simd_width_;

        rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx, t7, t5,
                preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md), tail_size,
                use_exact_tail_scalar_bcast};
        static_params_t static_params {this->param1, rhs_arg_static_params};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<lasx>>(
                this, jcp.post_ops, static_params);
    }
}

void jit_lasx_conv_fwd_kernel_f32::oh_step_unroll_kw(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_block = jcp.ic_block;
    int ic_tail = jcp.ic_tail;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w
                - nstl::max(0,
                        div_up(ki * dilate_w + pad_r - (kw - 1) * dilate_w, stride_w));

        auto compute = [=](int cur_ic_blk) {
            for (int ifm2 = 0; ifm2 < cur_ic_blk; ifm2++) {
                for (int jj = jj_start; jj < jj_end; jj++) {
                    size_t inp_off = get_input_offset(ifm2, filter_w_to_input(ki, jj, pad_l));

                    uni_xvldrepl_w(XVReg(oc_blocks * ur_w + jj),
                                    aux_reg_input, inp_off);
                }

                for (int ii = 0; ii < oc_blocks; ii++) {
                    uni_xvld(xr15, aux_reg_kernel, get_kernel_offset(ii, ki, ifm2));

                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (mayiuse(lasx)){
                            xvfmadd_s(XVReg(ur_w * ii + jj),
                                        XVReg(oc_blocks * ur_w + jj),
                                        xr15,
                                        XVReg(ur_w * ii + jj));
                        }
                        /*
                        else {
                            xvfmul_s(ytmp, XVReg(15), XVReg(oc_blocks * ur_w + jj));

                            xvfadd_s(XVReg(ur_w * ii + jj),
                                        XVReg(ur_w * ii + jj), ytmp);
                        }*/
                }
            }
        };

        if (ic_tail) {
            if (jcp.ic == ic_tail)
                compute(ic_tail);
            else {
                Label ic_blk_tail, ic_blk_done;
                mov_imm(X_TMP_0, ic_block);
                blt(reg_channel, X_TMP_0, ic_blk_tail);

                compute(ic_block);
                b(ic_blk_done);

                L(ic_blk_tail);
                compute(ic_tail);

                L(ic_blk_done);
            }
        } else {
            compute(ic_block);
        }
    }
}

void jit_lasx_conv_fwd_kernel_f32::oh_step_nopad(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    Label kw_loop;

    int kw = jcp.kw;
    int ic_blk = jcp.ic_block;

    xor_(ki_iter, ki_iter, ki_iter);
    L(kw_loop);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                size_t inp_off = get_input_offset(ifm2, filter_w_to_input(0, jj, pad_l));
                uni_xvldrepl_w(XVReg(oc_blocks * ur_w + jj), aux_reg_input, inp_off);
            }
            for (int ii = 0; ii < oc_blocks; ii++) {
                uni_xvld(xr15, aux_reg_kernel, get_kernel_offset(ii, 0, ifm2));

                for (int jj = jj_start; jj < jj_end; jj++)
                    if (mayiuse(lasx)){
                        xvfmadd_s(XVReg(ur_w * ii + jj),
                                XVReg(oc_blocks * ur_w + jj), xr15, XVReg(ur_w * ii + jj));
                    }
                    /*
                    else {
                        xvfmul_s(ytmp, XVReg(15), XVReg(oc_blocks * ur_w + jj));

                        xvfadd_s(XVReg(ur_w * ii + jj), XVReg(ur_w * ii + jj), ytmp);
                    }*/
            }
        }
        add_imm(aux_reg_kernel,
                aux_reg_kernel,
                get_kernel_offset(0, 1, 0),
                reg_long_offt);

        add_imm(aux_reg_input,
                aux_reg_input,
                get_input_offset(0, filter_w_to_input(1)),
                reg_long_offt);

        addi_d(ki_iter, ki_iter, 1);

        mov_imm(X_TMP_0, kw);
        blt(ki_iter, X_TMP_0, kw_loop);
    }
}

static int get_ymm_idx(
        const int ur_w, const int oc_block_idx, const int ur_w_idx) {
    return (ur_w * oc_block_idx + ur_w_idx);
}

static XVReg get_ymm(const int ur_w, const int oc_block_idx, const int ur_w_idx) {
    return XVReg(get_ymm_idx(ur_w, oc_block_idx, ur_w_idx));
}

template <typename F>
void iterate(const int load_loop_blk, const int ur, const int load_dim_tail,
        const F &f) {
    for (int i = 0; i < load_loop_blk; ++i) {
        const bool mask_flag = (load_dim_tail > 0) && (i == load_loop_blk - 1);
        for (int j = 0; j < ur; ++j)
            f(mask_flag, i, j);
    }
}
template <typename F>
void iterate(const int load_loop_blk, const int ur, const F &f) {
    iterate(load_loop_blk, ur, 0, f);
}

void jit_lasx_conv_fwd_kernel_f32::apply_postops(
        const int oc_blocks, const int ur_w, const int oc_tail) {
    if (jcp.with_eltwise || jcp.with_binary) {
        Label regular_store;
        andi(X_TMP_0, reg_ci_flag, FLAG_IC_LAST);
        beqz(X_TMP_0, regular_store);

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            const auto temp_offset_reg = t4;
            iterate(oc_blocks, ur_w, oc_tail,
                    [&](const bool mask_flag, const int i, const int j) {
                        const int aux_output_offset
                                = get_output_offset(i, j) / sizeof(float);
                        const auto vmm_idx = get_ymm_idx(ur_w, i, j);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_addr.emplace(
                                vmm_idx, ptr_a(param1, GET_OFF(oc_l_off)));

                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_val.emplace(
                                vmm_idx, i * jcp.oc_block);
                        rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_offset);
                        rhs_arg_params_tail.vmm_idx_to_out_off_oprnd.emplace(
                                vmm_idx, temp_offset_reg);
                        if (mask_flag)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

            const injector_utils::register_preserve_guard_t register_guard(
                    this, {temp_offset_reg});

            add_d(temp_offset_reg, reg_output, zero);

            ld_d(X_TMP_0, param1, GET_OFF(dst_orig));
            sub_d(temp_offset_reg, temp_offset_reg, X_TMP_0);

            srli_d(temp_offset_reg, temp_offset_reg, std::log2(sizeof(float)));

            Label postops_done;
            if (oc_tail) {
                Label postops_no_tail;
                andi(X_TMP_0, reg_oc_flag, FLAG_OC_LAST);
                beqz(X_TMP_0, postops_no_tail);

                postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params_tail);

                b(postops_done);

                L(postops_no_tail);
            }
            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
            L(postops_done);

        } else {
            iterate(oc_blocks, ur_w, [&](const bool, const int i, const int j) {
                vmm_idxs.emplace(get_ymm_idx(ur_w, i, j));
            });
            postops_injector_->compute_vector_range(vmm_idxs);
        }
        L(regular_store);
    }
}

void jit_lasx_conv_fwd_kernel_f32::width_blk_step(
        int ur_w, int pad_l, int pad_r, int oc_blocks) {
    int kw = jcp.kw;
    int oc_blk = jcp.oc_block;
    int oc_tail = jcp.oc_tail;

    if (oc_tail) {
        push_xreg(reg_oc_blocks);

        ld_d(reg_oc_flag, param1, GET_OFF(oc_flag));
    }

    auto load_output_bias_and_add_bias = [=](bool is_tail) {
        Label init_done, init_first;

        if (!jcp.with_sum) {
            andi(X_TMP_0, reg_ci_flag, FLAG_IC_FIRST);
            bnez(X_TMP_0, init_first);
        }

        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                const auto ymm = get_ymm(ur_w, ii, jj);
                if (is_tail && ii == oc_blocks - 1)
                    load_bytes(ymm, reg_output, get_output_offset(ii, jj),
                            oc_tail * sizeof(float));
                else{
                    uni_xvld(ymm, reg_output, get_output_offset(ii, jj));
                }
            }

        if (jcp.with_sum && jcp.with_bias) {
            andi(X_TMP_0, reg_ci_flag, FLAG_IC_FIRST);
            beqz(X_TMP_0, init_done);

            for (int ii = 0; ii < oc_blocks; ii++)
                for (int jj = 0; jj < ur_w; jj++) {
                    const XVReg ymm = get_ymm(ur_w, ii, jj);
                    if (is_tail && ii == oc_blocks - 1) {
                        load_bytes(ytmp, reg_bias, sizeof(float) * ii * oc_blk,
                                oc_tail * sizeof(float));

                        xvfadd_s(ymm, ymm, ytmp);
                    } else {
                        uni_xvld(ytmp, reg_bias, sizeof(float) * ii * oc_blk);
                        xvfadd_s(ymm, ymm, ytmp);
                    }
                }
        }
        b(init_done);

        L(init_first);

        if (jcp.with_bias) {
            for (int ii = 0; ii < oc_blocks; ii++)
                for (int jj = 0; jj < ur_w; jj++) {
                    const XVReg ymm = get_ymm(ur_w, ii, jj);
                    if (is_tail && ii == oc_blocks - 1)
                        load_bytes(ymm, reg_bias, sizeof(float) * ii * oc_blk,
                                oc_tail * sizeof(float));
                    else{
                        uni_xvld(ymm, reg_bias, sizeof(float) * ii * oc_blk);
                    }
                }
        } else {
            for (int ii = 0; ii < oc_blocks; ii++)
                for (int jj = 0; jj < ur_w; jj++) {
                    const XVReg ymm = get_ymm(ur_w, ii, jj);
                    xvxor_v(ymm, ymm, ymm);
                }
        }
        L(init_done);
    };

    if (oc_tail) {
        if (jcp.nb_oc > jcp.nb_oc_blocking) {
            Label load_tail, load_done;
            andi(X_TMP_0, reg_oc_flag, FLAG_OC_LAST);
            bnez(X_TMP_0, load_tail);

            load_output_bias_and_add_bias(false);
            b(load_done);

            L(load_tail);
            load_output_bias_and_add_bias(true);

            L(load_done);
        } else {
            load_output_bias_and_add_bias(true);
        }
    } else {
        load_output_bias_and_add_bias(false);
    }

    if (one_of(jcp.ndims, 3, 4)) {
        add_d(aux_reg_input, reg_input, zero);

        add_d(aux_reg_kernel, reg_kernel, zero);
    }

    Label skip_kh_loop, skip_kd_loop, kd_loop;
    if (jcp.ndims == 5) {
        push_xreg(reg_output);

        push_xreg(oi_iter);

        ld_d(reg_ki, param1, GET_OFF(kd_padding));

        ld_d(aux_reg_ker_d, param1, GET_OFF(filt));

        add_d(aux_reg_inp_d, reg_input, zero);

        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1) < jcp.f_pad) {
            beq(reg_ki, zero, skip_kd_loop);
        }
        L(kd_loop);
        ld_d(kj, param1, GET_OFF(kh_padding));

    } else {
        add_d(kj, reg_kh, zero);
    }

    if (jcp.ndims == 5) {
        add_d(aux_reg_input, aux_reg_inp_d, zero);

        add_d(aux_reg_kernel, aux_reg_ker_d, zero);
    }

    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        beq(kj, zero, skip_kh_loop);
    }
    Label kh_loop;
    L(kh_loop);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, oc_blocks);
            add_imm(aux_reg_input, aux_reg_input,
                    get_input_offset(0, filter_h_to_input(1))
                            - get_input_offset(0, filter_w_to_input(kw)),
                    X_TMP_0);
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks);
            add_imm(aux_reg_kernel,
                    aux_reg_kernel,
                    get_kernel_offset(0, kw, 0),
                    reg_long_offt);

            add_imm(aux_reg_input,
                    aux_reg_input,
                    get_input_offset(0, filter_h_to_input(1)),
                    reg_long_offt);
        }

        addi_d(kj, kj, -1);
        blt(zero, kj, kh_loop);
    }

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add_imm(aux_reg_inp_d,
                aux_reg_inp_d,
                get_input_offset(0, filter_d_to_input(1)),
                reg_long_offt);

        add_imm(aux_reg_ker_d,
                aux_reg_ker_d,
                get_kernel_offset(0, jcp.kw * jcp.kh, 0),
                reg_long_offt);

        addi_d(reg_ki, reg_ki, -1);
        blt(zero, reg_ki, kd_loop);

        L(skip_kd_loop);

        pop_xreg(oi_iter);
        pop_xreg(reg_output);
    }

    apply_postops(oc_blocks, ur_w, oc_tail);

    auto store_output = [=](bool is_tail, int tail) {
        const auto is_padding = jcp.oc_without_padding != jcp.oc;
        if (is_padding){
            uni_vpxor(ytmp, ytmp, ytmp);
        }
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                XVReg reg_out = get_ymm(ur_w, ii, jj);
                if (is_tail && ii == oc_blocks - 1) {
                    if (is_padding && jcp.with_binary) {
                        uni_xvst(ytmp, reg_output, get_output_offset(ii, jj));
                    }
                    store_bytes(reg_out, reg_output, get_output_offset(ii, jj),
                            tail * sizeof(float));
                } else{
                    uni_xvst(reg_out, reg_output, get_output_offset(ii, jj));
                }
            }
    };

    if (oc_tail) {
        if (jcp.nb_oc > jcp.nb_oc_blocking) {
            Label store_tail, store_done;
            andi(X_TMP_0, reg_oc_flag, FLAG_OC_LAST);
            bnez(X_TMP_0, store_tail);

            store_output(false, oc_tail);
            b(store_done);

            L(store_tail);
            store_output(true, oc_tail);

            L(store_done);
        } else {
            store_output(true, oc_tail);
        }
    } else {
        Label regular_store;
        Label store_done;
        const int tail = jcp.oc_without_padding % jcp.oc_block;
        if (jcp.with_binary && tail) {
            andi(X_TMP_0, reg_ci_flag, FLAG_IC_LAST);
            beqz(X_TMP_0, regular_store);

            if (!oc_tail){
                ld_d(reg_oc_flag, param1, GET_OFF(oc_flag));
            }

            andi(X_TMP_0, reg_oc_flag, FLAG_OC_LAST);
            beqz(X_TMP_0, regular_store);

            store_output(true, tail);

            b(store_done);
        }

        L(regular_store);
        store_output(false, oc_tail);

        L(store_done);
    }

    if (oc_tail) {
        pop_xreg(reg_oc_blocks);
    }
}

inline void jit_lasx_conv_fwd_kernel_f32::solve_common(int oc_blocks) {
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int str_w = jcp.stride_w;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, jcp.r_pad);
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, str_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1, oc_blocks); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0, oc_blocks); // "lpad"

        add_imm(reg_input, reg_input,
                    get_input_offset(0, filter_w_to_input(0, ur_w, l_pad)),
                    X_TMP_0);

        add_imm(reg_output, reg_output,
                 get_output_offset(0, ur_w),
                 X_TMP_0);
    }

    Label ow_loop;
    xor_(oi_iter, oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop);

        width_blk_step(ur_w, 0, 0, oc_blocks); // "middle"

        add_imm(reg_input, reg_input,
                    get_input_offset(0, filter_w_to_input(0, ur_w)),
                    X_TMP_0);

        add_imm(reg_output, reg_output,
                    get_output_offset(0, ur_w), X_TMP_0);

        addi_d(oi_iter, oi_iter, 1);

        mov_imm(X_TMP_0, n_oi);
        blt(oi_iter, X_TMP_0, ow_loop);
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        width_blk_step(ur_w, 0, r_pad1, oc_blocks); // "rpad"

        add_imm(reg_input, reg_input,
                    get_input_offset(0, filter_w_to_input(0, ur_w)),
                    X_TMP_0);

        add_imm(reg_output, reg_output,
                    get_output_offset(0, ur_w), X_TMP_0);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad, oc_blocks); // "tail"
}

void jit_lasx_conv_fwd_kernel_f32::generate() {
    this->preamble();

    ld_d(reg_input, this->param1, GET_OFF(src));

    ld_d(reg_output, this->param1, GET_OFF(dst));

    ld_d(reg_kernel, this->param1, GET_OFF(filt));

    if (jcp.with_bias){
        ld_d(reg_bias, this->param1, GET_OFF(bias));
    }

    ld_d(reg_kh, this->param1, GET_OFF(kh_padding));

    ld_d(reg_ci_flag, this->param1, GET_OFF(flags));

    ld_d(reg_oc_blocks, this->param1, GET_OFF(oc_blocks));

    if (is_src_layout_nxc()){
        ld_d(reg_channel, param1, GET_OFF(reduce_work));
    }

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;

    Label tail, exit;

    if (jcp.nb_oc > jcp.nb_oc_blocking) {
        mov_imm(X_TMP_1, jcp.nb_oc_blocking);
        bne(reg_oc_blocks, X_TMP_1, nb_oc_tail ? tail : exit);

        solve_common(jcp.nb_oc_blocking);
        b(exit);

        if (nb_oc_tail) {
            L(tail);
            mov_imm(X_TMP_1, nb_oc_tail);
            bne(reg_oc_blocks, X_TMP_1, exit);

            solve_common(nb_oc_tail);
        }

        L(exit);
    } else if (jcp.nb_oc == jcp.nb_oc_blocking) {
        solve_common(jcp.nb_oc_blocking);
    } else {
        solve_common(nb_oc_tail);
    }

    this->postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

status_t jit_lasx_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr) {
    if (!mayiuse(lasx)) return status::unimplemented;

    jcp.nthr = dnnl_get_max_threads();

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    auto wei_tag_OIxio = with_groups
            ? pick(ndims - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
            : pick(ndims - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);
    auto wei_tag_Oxio = with_groups ? pick(ndims - 3, gOwi8o, gOhwi8o, gOdhwi8o)
                                    : pick(ndims - 3, Owi8o, Ohwi8o, Odhwi8o);

    jcp.src_tag
            = src_d.matches_one_of_tag(dat_tag_ncx, dat_tag_nxc, dat_tag_nCx8c);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag_OIxio, wei_tag_Oxio);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());

    bool is_data_layout_nxc
            = everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);

    // Disable this kernel on high width 1d object as gemm performs better until
    // optimizations can be made to fix it.
    if (is_data_layout_nxc && ndims == 3 && jcp.ow > 11 * 1024)
        return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    const auto &post_ops = attr.post_ops_;

    jcp.with_sum = post_ops.find(primitive_kind::sum) != -1;
    const int eltwise_ind = post_ops.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    const int binary_ind = post_ops.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;

    jcp.post_ops = post_ops;

    const int simd_w = 8;
    const bool flat = jcp.ic < simd_w;
    const bool mimo = !flat;

    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off
            = one_of(jcp.src_tag, ncw, nchw, ncdhw) && jcp.ngroups > 1 ? jcp.ic
                                                                       : 1;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo) jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    if (jcp.with_eltwise || jcp.with_binary)
        if (!mayiuse(lasx)) return status::unimplemented;

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    const bool post_ops_ok_ = post_ops_ok({lasx, {eltwise, binary, sum},
            jcp.post_ops, &dst_d, sum_at_pos_0_only, sum_requires_scale_one,
            {broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc}});
    if (!post_ops_ok_) return status::unimplemented;

    bool args_ok = true
            && IMPLICATION(flat,
                    jcp.wei_tag == wei_tag_Oxio
                            && ((jcp.src_tag == dat_tag_ncx
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && IMPLICATION(mimo,
                    jcp.wei_tag == wei_tag_OIxio
                            && ((jcp.src_tag == dat_tag_nCx8c
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1];
    if (!args_ok) return status::unimplemented;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.nb_oc_blocking = 4; /* the optimal value for the kernel */

    // Thus, we can only assign 14 or 15 YMMs for data storage
    const int num_avail_regs = mayiuse(lasx) ? 15 : 14;
    if (!mayiuse(lasx)) {
        if ((jcp.nb_oc_blocking + 1) * jcp.ur_w > num_avail_regs) {
            // current register assignment requires more YMMs than available
            // adjust one of nb_oc_block, ur_w preserving to ur_w >= l_pad
            if (jcp.ur_w > jcp.l_pad && jcp.ur_w > 1)
                jcp.ur_w -= 1;
            else {
                for (int b = 3; b > 1; b--) {
                    if (jcp.nb_oc % b == 0) {
                        jcp.nb_oc_blocking = b;
                        break;
                    }
                }
                if ((jcp.nb_oc_blocking + 1) * jcp.ur_w > num_avail_regs) {
                    // No optimal size for 'nb_oc_blocking' with regards to
                    // 'nb_oc', default to only unroll by 'ur_w'.
                    jcp.nb_oc_blocking = 1;
                }
            }
        }
    }

    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true && IMPLICATION(!is_data_layout_nxc, jcp.oc % simd_w == 0)
            && jcp.l_pad <= jcp.ur_w
            && IMPLICATION(jcp.kw > 7,
                    (jcp.t_pad == 0 && jcp.l_pad == 0)
                            || (jcp.stride_w == 1 && jcp.stride_h == 1))
            && IMPLICATION(mimo && !is_data_layout_nxc, jcp.ic % simd_w == 0);
    if (!args_ok) return status::unimplemented;

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc
            ? jcp.oc % simd_w
            : (jcp.with_binary ? jcp.oc_without_padding % simd_w : 0);

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));

    if (r_pad_no_tail > jcp.ur_w * jcp.stride_w && jcp.ow / jcp.ur_w > 1) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = nstl::min(r_pad_no_tail / jcp.stride_w + jcp.ur_w_tail,
                nstl::min(jcp.ow, num_avail_regs / 2));
        jcp.nb_oc_blocking = (num_avail_regs - jcp.ur_w) / jcp.ur_w;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0,
                calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                        jcp.stride_w, ext_kw));
        if (jcp.ur_w < nstl::max(jcp.l_pad, r_pad_no_tail))
            return status::unimplemented;
    }
    assert(jcp.nb_oc_blocking > 0);
    assert(jcp.ur_w * (jcp.nb_oc_blocking + 1) <= num_avail_regs);

    jcp.ic_block = flat ? jcp.ic : simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.nb_ic_blocking = 12;
    jcp.nb_ic_blocking_max = 16;

    /* adjust the thread decomposition
     * to improve the perf for small problem size
     * the threshold L1_cache_size is empirical
     * simply set the thread as 4 for now
     * TODO: Add get_thr_eff func to get the optimal thread number*/
    size_t wei_size = (size_t)sizeof(float) * jcp.ic * jcp.oc * jcp.kh * jcp.kw
            * jcp.kd;
    size_t inp_size = (size_t)jcp.typesize_in * jcp.mb * jcp.ic * jcp.ih
            * jcp.iw * jcp.id;
    size_t out_size = (size_t)jcp.typesize_out * jcp.mb * jcp.oc * jcp.oh
            * jcp.ow * jcp.od;
    size_t total_size = jcp.ngroups * (wei_size + inp_size + out_size);

    const unsigned int L1_cache_size = platform::get_per_core_cache_size(1);

    if (jcp.ngroups < jcp.nthr && total_size < L1_cache_size) {
        jcp.nthr = nstl::min(jcp.nthr, 4);
    }

    return status::success;
}

void jit_lasx_conv_fwd_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book<float>(key_conv_padded_bias, jcp.oc);
}

void jit_lasx_conv_bwd_data_kernel_f32::compute_loop(
        int ur_w, int l_overflow, int r_overflow) {
    int kw = jcp.kw;
    int ow = jcp.ow;

    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;
    int oc_tail = jcp.oc_tail;
    int ic_tail = jcp.ic_tail;

    Label kd_loop, skip_kd_loop;
    Label oc_loop, skip_oc_loop;

    for (int ii = 0; ii < nb_ic_block; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            xvxor_v(XVReg(ur_w * ii + jj),
                    XVReg(ur_w * ii + jj),
                    XVReg(ur_w * ii + jj));
        }

    if (oc_tail) {
        push_xreg(reg_long_offt);
        ld_d(reg_reduce_work, param1, GET_OFF(reduce_work));
    }

    if (one_of(jcp.ndims, 3, 4)) {
        bge(zero, reg_channel_work, skip_oc_loop);

        xor_(reg_channel, reg_channel, reg_channel);

        add_d(aux_reg_ddst_oc_loop, reg_ddst, zero);

        add_d(aux_reg_kernel_oc_loop, reg_kernel, zero);

        L(oc_loop);
        add_d(aux_reg_ddst, aux_reg_ddst_oc_loop, zero);

        add_d(aux_reg_kernel, aux_reg_kernel_oc_loop, zero);
    }

    if (jcp.ndims == 5) {
        assert(jcp.nb_oc_blocking == 1);
        push_xreg(oi_iter);

        ld_d(reg_ki, this->param1, GET_OFF(kd_padding));

        bge(zero, reg_ki, skip_kd_loop);

        add_d(aux_reg_dst_d, reg_ddst, zero);

        ld_d(aux_reg_ker_d, this->param1, GET_OFF(filt));

        L(kd_loop);

        ld_d(kj, this->param1, GET_OFF(kh_padding));
    } else {
        add_d(kj, reg_kh, zero);
    }

    if (jcp.ndims == 5) {
        add_d(aux_reg_ddst, aux_reg_dst_d, zero);

        add_d(aux_reg_kernel, aux_reg_ker_d, zero);
    }

    Label kh_loop, skip_kh_loop;
    bge(zero, kj, skip_kh_loop);

    L(kh_loop);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow); // 0;
            int jj_end = get_iw_end(ur_w, ki, r_overflow); // ur_w;

            auto compute = [=](int cur_oc_blk) {
                for (int ofm2 = 0; ofm2 < cur_oc_blk; ofm2++) {
                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        int aux_output_offset = get_ddst_offset(
                                0, filter_w_to_ddst(ki, jj, jcp.l_pad), ofm2);
                        uni_xvldrepl_w(XVReg(nb_ic_block * ur_w + jj / stride_w),
                                    aux_reg_ddst, aux_output_offset);
                    }

                    for (int ii = 0; ii < nb_ic_block; ii++) {
                        uni_xvld(xr15, aux_reg_kernel, get_kernel_offset(0, ii, ki, ofm2));

                        for (int jj = jj_start; jj < jj_end; jj += stride_w){
                            xvfmadd_s(XVReg(ur_w * ii + jj),
                                    XVReg(nb_ic_block * ur_w + jj / stride_w),
                                    xr15,
                                    XVReg(ur_w * ii + jj));
                        }
                    }
                }
            };

            if (oc_tail) {
                if (jcp.oc == oc_tail)
                    compute(oc_tail);
                else {
                    Label oc_blk_tail, oc_blk_done;
                    mov_imm(X_TMP_0, oc_block);
                    blt(reg_reduce_work, X_TMP_0, oc_blk_tail);

                    compute(oc_block);
                    b(oc_blk_done);

                    L(oc_blk_tail);
                    compute(oc_tail);

                    L(oc_blk_done);
                }
            } else {
                compute(oc_block);
            }
        }

        add_imm(aux_reg_kernel, aux_reg_kernel,
                get_kernel_offset(0, 0, stride_h * kw, 0),
                X_TMP_0);

        sub_imm(aux_reg_ddst, aux_reg_ddst,
                get_ddst_offset(0, (jcp.dilate_h + 1) * ow, 0), X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, kj, kh_loop);
    }
    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        sub_imm(aux_reg_dst_d, aux_reg_dst_d,
                get_ddst_offset(0, (jcp.dilate_d + 1) * jcp.oh * ow, 0), X_TMP_0);

        add_imm(aux_reg_ker_d, aux_reg_ker_d,
                get_kernel_offset(0, 0, jcp.kw * jcp.kh, 0),
                X_TMP_0);

        addi_d(reg_ki, reg_ki, -1);
        blt(zero, reg_ki, kd_loop);

        L(skip_kd_loop);

        pop_xreg(oi_iter);
    }

    if (one_of(jcp.ndims, 3, 4)) {
        int ddst_oc_shift = get_ddst_offset(1, 0, 0);
        int kernel_oc_shift = get_kernel_offset(1, 0, 0, 0);

        add_imm(aux_reg_ddst_oc_loop, aux_reg_ddst_oc_loop, ddst_oc_shift,
                X_TMP_0);

        add_imm(aux_reg_kernel_oc_loop, aux_reg_kernel_oc_loop, kernel_oc_shift,
                X_TMP_0);

        if (oc_tail){
            sub_imm(reg_reduce_work, reg_reduce_work, jcp.oc_block, X_TMP_0);
        }
        addi_d(reg_channel, reg_channel, 1);

        blt(reg_channel, reg_channel_work, oc_loop);

        L(skip_oc_loop);
        ld_d(reg_channel, param1, GET_OFF(channel));
    }

    if (oc_tail){
        pop_xreg(reg_long_offt);
    }

    auto load_store_dsrc = [=](bool is_tail) {
        ld_d(reg_channel, param1, GET_OFF(channel));

        Label no_update_label;
        beq(reg_channel, zero, no_update_label);

        for (int ii = 0; ii < nb_ic_block; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                if (is_tail && ii == nb_ic_block - 1){
                    load_bytes(xr15, reg_dsrc, get_dsrc_offset(ii, jj),
                            ic_tail * sizeof(float));
                }
                else{
                    uni_xvld(xr15, reg_dsrc, get_dsrc_offset(ii, jj));
                }
                xvfadd_s(XVReg(ur_w * ii + jj), XVReg(ur_w * ii + jj), xr15);
            }

        L(no_update_label);

        for (int ii = 0; ii < nb_ic_block; ii++)
            for (int jj = 0; jj < ur_w; jj++) {
                if (is_tail && ii == nb_ic_block - 1){
                    store_bytes(XVReg(ur_w * ii + jj), reg_dsrc,
                            get_dsrc_offset(ii, jj), ic_tail * sizeof(float));
                }
                else{
                    uni_xvst(XVReg(ur_w * ii + jj),
                                reg_dsrc, get_dsrc_offset(ii, jj));
                }
            }
    };

    if (ic_tail) {
        Label load_store_tail, load_store_done;
        ld_d(reg_ci_flag, param1, GET_OFF(flags));

        andi(X_TMP_0, reg_ci_flag, FLAG_IC_LAST);
        bnez(X_TMP_0, load_store_tail);

        load_store_dsrc(false);
        b(load_store_done);

        L(load_store_tail);
        load_store_dsrc(true);

        L(load_store_done);
    } else {
        load_store_dsrc(false);
    }
}

void jit_lasx_conv_bwd_data_kernel_f32::generate() {
    preamble();

    ld_d(reg_dsrc, this->param1, GET_OFF(src));

    ld_d(reg_ddst, this->param1, GET_OFF(dst));

    ld_d(reg_kernel, this->param1, GET_OFF(filt));

    ld_d(reg_kh, this->param1, GET_OFF(kh_padding));

    ld_d(reg_channel, param1, GET_OFF(channel));

    ld_d(reg_channel_work, param1, GET_OFF(ch_blocks));

    int ddst_shift = get_ddst_offset(0, filter_w_to_ddst(0, jcp.ur_w), 0);
    int dsrc_shift = get_dsrc_offset(0, jcp.ur_w);

    const int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);

    int l_overflow = nstl::max(0, (ext_kw - 1 - jcp.l_pad) / jcp.stride_w);
    int r_overflow = nstl::max(
            0, (ext_kw - 1 - nstl::max(0, jcp.r_pad)) / jcp.stride_w);
    int r_overflow1 = nstl::max(
            0, (ext_kw - 1 - jcp.r_pad - jcp.ur_w_tail) / jcp.stride_w);

    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (jcp.ur_w == jcp.iw) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow1);
        add_imm(reg_dsrc, reg_dsrc, dsrc_shift, X_TMP_0);
        add_imm(reg_ddst, reg_ddst, ddst_shift, X_TMP_0);

        if (jcp.ur_w_tail != 0)
            compute_loop(jcp.ur_w_tail, 0, r_overflow);
    } else {
        xor_(oi_iter, oi_iter, oi_iter);
        if (l_overflow > 0) {
            compute_loop(jcp.ur_w, l_overflow, 0);
            add_imm(reg_dsrc, reg_dsrc, dsrc_shift, X_TMP_0);
            add_imm(reg_ddst, reg_ddst, ddst_shift, X_TMP_0);
            addi_d(oi_iter, oi_iter, 1);
        }

        if ((l_overflow <= 0 && n_oi > 0) || (l_overflow > 0 && n_oi > 1)) {
            Label ow_loop;
            L(ow_loop);
            {
                compute_loop(jcp.ur_w, 0, 0);
                add_imm(reg_dsrc, reg_dsrc, dsrc_shift, X_TMP_0);
                add_imm(reg_ddst, reg_ddst, ddst_shift, X_TMP_0);

                addi_d(oi_iter, oi_iter, 1);

                mov_imm(X_TMP_0, n_oi);
                blt(oi_iter, X_TMP_0, ow_loop);
            }
        }

        if (r_overflow1 > 0) {
            compute_loop(jcp.ur_w, 0, r_overflow1);
            add_imm(reg_dsrc, reg_dsrc, dsrc_shift, X_TMP_0);
            add_imm(reg_ddst, reg_ddst, ddst_shift, X_TMP_0);
        }

        if (jcp.ur_w_tail != 0) compute_loop(jcp.ur_w_tail, 0, r_overflow);
    }

    this->postamble();
}

status_t jit_lasx_conv_bwd_data_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(lasx)) return status::unimplemented;

    jcp.nthr = dnnl_get_max_threads();

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    int ndims = diff_src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    const int simd_w = 8;

    /* derivatives */
    jcp.idp = jcp.id + 2 * jcp.f_pad;
    jcp.ihp = jcp.ih + 2 * jcp.t_pad;
    jcp.iwp = jcp.iw + 2 * jcp.l_pad;
    jcp.ohp = jcp.oh; /* do we really need */
    jcp.owp = jcp.ow; /* padded output ??? */

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    auto wei_tag = with_groups
            ? pick(ndims - 3, gOIw8o8i, gOIhw8o8i, gOIdhw8o8i)
            : pick(ndims - 3, OIw8o8i, OIhw8o8i, OIdhw8o8i);

    jcp.src_tag = diff_src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);

    jcp.typesize_in = types::data_type_size(diff_src_d.data_type());
    jcp.typesize_out = types::data_type_size(diff_dst_d.data_type());

    bool is_data_layout_nxc
            = everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;

    /* gemm-based convolution performs better in these cases */
    if (jcp.ic < simd_w && jcp.kw > 3 && jcp.stride_w > 1)
        return status::unimplemented;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ic_block = (!is_data_layout_nxc && jcp.ic % simd_w) ? 1 : simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % simd_w : 0;

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.nb_ic_blocking = 1;
    jcp.nb_oc_blocking = 1;
    jcp.ur_w = 1;

    if (one_of(ndims, 3, 4) && jcp.ow < 40)
        jcp.nb_oc_blocking = jcp.ow < 15 ? 4 : 2;

    auto required_dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;

    bool args_ok = true && jcp.stride_w == jcp.stride_h && jcp.stride_d == 1
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.ic % simd_w == 0 && jcp.oc % simd_w == 0)
            && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.dst_tag == required_dat_tag
            && jcp.src_tag == required_dat_tag && jcp.wei_tag == wei_tag;
    if (!args_ok) return status::unimplemented;

    const int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    const int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);

    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    int l_overflow = nstl::max(0, (ext_kw - 1 - jcp.l_pad) / jcp.stride_w);

    const int max_regs = 15; /* Maximum number of registers available for
                                result accumulation and delta dst data.
                                One additional register is reserved for weights
                                data. */

    /* Find the best blocking with maximum number of fma instructions
       per ur_w * nb_ic_blocking compute loops. Number of required registers
       is num_regs = ur_w * nb_ic_blocking + ur_w / stride_w <= max_regs.
       ur_w must be divisible by stride_w */
    if (jcp.stride_w + 1 > max_regs) /* Minimal possible registers
                                         distribution exceeds max_regs */
        return status::unimplemented;

    int best_nfmas = 0;
    for (int b = 1; b <= 4; b++) {
        if (jcp.nb_ic % b != 0) continue;

        for (int u = jcp.stride_w; u * b + u / jcp.stride_w <= max_regs
                && u < jcp.iw + jcp.stride_w;
                u += jcp.stride_w) {
            int ur_w = nstl::min(u, jcp.iw);
            /* maximum 1 step with l_overflow so far */
            if (l_overflow * jcp.stride_w > ur_w && ur_w != jcp.iw) continue;
            int nfmas = div_up(ur_w, jcp.stride_w) * b;
            if (nfmas > best_nfmas
                    || (nfmas == best_nfmas && jcp.ur_w < ur_w)) {
                jcp.ur_w = ur_w;
                jcp.nb_ic_blocking = b;
                best_nfmas = nfmas;
            }
        }
    }
    if (best_nfmas == 0) /* can't find appropriate blocking */
        return status::unimplemented;

    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    int r_overflow_no_tail = nstl::max(
            0, (ext_kw - 1 - jcp.r_pad - jcp.ur_w_tail) / jcp.stride_w);

    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok) return status::unimplemented;

    /* adjust the thread decomposition
     * to improve the perf for small problem size
     * the threshold L1_cache_size is empirical
     * simply set the thread to 4 for now
     * TODO: Add get_thr_eff func to get optimal thread number */
    size_t wei_size = (size_t)sizeof(float) * jcp.ic * jcp.oc * jcp.kh * jcp.kw
            * jcp.kd;
    size_t inp_size = (size_t)jcp.typesize_in * jcp.mb * jcp.ic * jcp.ih
            * jcp.iw * jcp.id;
    size_t out_size = (size_t)jcp.typesize_out * jcp.mb * jcp.oc * jcp.oh
            * jcp.ow * jcp.od;
    size_t total_size = jcp.ngroups * (wei_size + inp_size + out_size);
    const unsigned int L1_cache_size = platform::get_per_core_cache_size(1);

    if (jcp.ngroups < jcp.nthr && total_size < L1_cache_size) {
        jcp.nthr = nstl::min(jcp.nthr, 4);
    }

    return status::success;
}

void jit_lasx_conv_bwd_data_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

void jit_lasx_conv_bwd_weights_kernel_f32::generate() {
    this->preamble();

    ld_d(reg_input, this->param1, GET_OFF(src));

    ld_d(reg_output, this->param1, GET_OFF(dst));

    ld_d(reg_kernel, this->param1, GET_OFF(filt));

    compute_oh_loop_common();
    this->postamble();
}

status_t jit_lasx_conv_bwd_weights_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(lasx)) return status::unimplemented;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    auto wei_tag_OIxio = with_groups
            ? pick(ndims - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
            : pick(ndims - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);
    auto wei_tag_Oxio = with_groups ? pick(ndims - 3, gOwi8o, gOhwi8o, gOdhwi8o)
                                    : pick(ndims - 3, Owi8o, Ohwi8o, Odhwi8o);

    jcp.src_tag
            = src_d.matches_one_of_tag(dat_tag_ncx, dat_tag_nxc, dat_tag_nCx8c);
    jcp.wei_tag
            = diff_weights_d.matches_one_of_tag(wei_tag_OIxio, wei_tag_Oxio);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);

    bool is_data_layout_nxc
            = everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);

    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;

    const int simd_w = 8;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));
    jcp.back_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd));

    const int max_h_pad = ext_kh;
    const int max_w_pad = ext_kw;
    const bool boundaries_ok = true && jcp.t_pad < max_h_pad
            && jcp.b_pad < max_h_pad && jcp.l_pad < max_w_pad
            && jcp.r_pad < max_w_pad && jcp.f_pad == 0 && jcp.back_pad == 0;
    if (!boundaries_ok) return status::unimplemented;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo) jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % simd_w : 0;

    bool args_ok = true
            && IMPLICATION(flat,
                    jcp.wei_tag == wei_tag_Oxio
                            && ((jcp.src_tag == dat_tag_ncx
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && IMPLICATION(mimo,
                    jcp.wei_tag == wei_tag_OIxio
                            && ((jcp.src_tag == dat_tag_nCx8c
                                        && jcp.dst_tag == dat_tag_nCx8c)
                                    || (jcp.src_tag == dat_tag_nxc
                                            && jcp.dst_tag == dat_tag_nxc)))
            && IMPLICATION(mimo && !is_data_layout_nxc, jcp.ic % simd_w == 0)
            && IMPLICATION(!is_data_layout_nxc, jcp.oc % simd_w == 0)
            && jcp.kw < 14 && jcp.kh <= jcp.t_pad + jcp.ih /* [bwd_w:r1] */
            && jcp.kh <= jcp.ih /* [bwd_w:r2] */
            && jcp.kd <= jcp.f_pad + jcp.id && jcp.kd <= jcp.id
            && jcp.t_pad < jcp.kh /* XXX: must fix the kernel! */
            && jcp.dilate_d == 0 && jcp.dilate_h == 0 && jcp.dilate_w == 0
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1];
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = flat ? jcp.ic : simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    return status::success;
}

void jit_lasx_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && (jcp.oc_without_padding % jcp.oc_block != 0)) {
        const size_t nelems_padded_bias
                = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book<float>(key_conv_padded_bias, nelems_padded_bias);
    }
}

inline void jit_lasx_conv_bwd_weights_kernel_f32::od_step_comeback_pointers() {
    Label kd_comeback_loop;
    mov_imm(kj, jcp.kd);

    L(kd_comeback_loop);
    {
        sub_imm(aux_reg_input, aux_reg_input,
                get_input_offset(0, jcp.iw * jcp.ih), X_TMP_0);

        sub_imm(aux_reg_kernel, aux_reg_kernel,
                get_kernel_offset(jcp.kw * jcp.kh, 0), X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, kj, kd_comeback_loop);
    }
}

inline void jit_lasx_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers() {
    add_d(kj, reg_kh, zero);

    Label kh_comeback_loop;
    L(kh_comeback_loop);
    {
        sub_imm(reg_input, reg_input,
                get_input_offset(0, jcp.iw), X_TMP_0);

        sub_imm(reg_kernel, reg_kernel,
                get_kernel_offset(jcp.kw, 0), X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, kj, kh_comeback_loop);
    }
}

inline void jit_lasx_conv_bwd_weights_kernel_f32::compute_ic_block_step(
        int ur_w, int pad_l, int pad_r, int ic_block_step, int input_offset,
        int kernel_offset, int output_offset) {

    if (ic_block_step <= 0) return;

    const int kw = jcp.kw;
    const int oc_tail = jcp.oc_tail;

    if (oc_tail) {
        push_xreg(reg_kh);
        ld_d(reg_ci_flag, param1, GET_OFF(flags));
    }

    auto load_compute_store = [=](bool is_tail) {
        for (int i_kw = 0; i_kw < kw; i_kw++)
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                size_t off = get_kernel_offset(i_kw, i_ic) + kernel_offset;
                if (is_tail){
                    load_bytes(XVReg(i_kw * ic_block_step + i_ic), reg_kernel,
                            off, oc_tail * sizeof(float));
                }
                else{
                    uni_xvld(XVReg(i_kw * ic_block_step + i_ic),
                            reg_kernel, off);
                }
            }

        for (int i_ur = 0; i_ur < ur_w; i_ur++) {
            if (is_tail){
                load_bytes(XVReg(kw * ic_block_step + 0), reg_output,
                        get_output_offset(0, i_ur) + output_offset,
                        oc_tail * sizeof(float));
            }
            else{
                uni_xvld(XVReg(kw * ic_block_step + 0),
                        reg_output,
                        get_output_offset(0, i_ur) + output_offset);
            }

            for (int i_kw = 0; i_kw < kw; i_kw++) {
                int i_iw = i_ur * jcp.stride_w + i_kw;
                if (i_iw - pad_l < 0
                        || i_iw > (ur_w - 1) * jcp.stride_w + kw - 1 - pad_r)
                    continue;
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    size_t i_off = get_input_offset(i_ic, i_iw - pad_l);
                    uni_xvldrepl_w(XVReg(kw * ic_block_step + 1), reg_input, i_off);

                    xvfmadd_s(XVReg(i_kw * ic_block_step + i_ic),
                                XVReg(kw * ic_block_step + 0),
                                XVReg(kw * ic_block_step + 1),
                                XVReg(i_kw * ic_block_step + i_ic));
                }
            }
        }

        for (int i_kw = 0; i_kw < kw; i_kw++)
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                size_t off = get_kernel_offset(i_kw, i_ic) + kernel_offset;
                if (is_tail){
                    store_bytes(XVReg(i_kw * ic_block_step + i_ic), reg_kernel,
                            off, oc_tail * sizeof(float));
                }

                else{
                    uni_xvst(XVReg(i_kw * ic_block_step + i_ic), reg_kernel, off);
                }
            }
    };

    if (oc_tail) {
        Label load_tail, load_done;
        andi(X_TMP_0, reg_ci_flag, FLAG_OC_LAST);
        bnez(X_TMP_0, load_tail);

        load_compute_store(false);
        b(load_done);

        L(load_tail);
        load_compute_store(true);

        L(load_done);
    } else {
        load_compute_store(false);
    }

    if (oc_tail){
        pop_xreg(reg_kh);
    }
}

inline void jit_lasx_conv_bwd_weights_kernel_f32::compute_oh_step_disp() {
    int ic_block_step;
    if (one_of(jcp.src_tag, ncw, nchw, ncdhw)) {
        ic_block_step = jcp.kw >= 5 ? 1 : jcp.ic_block;
    } else if (one_of(jcp.src_tag, nwc, nhwc, ndhwc)) {
        ic_block_step = jcp.kw > 7 ? 1 : jcp.kw > 3 ? 2 : jcp.kw > 1 ? 4 : 8;
        if (jcp.ic_block % ic_block_step != 0) {
            ic_block_step = jcp.ic_block < ic_block_step ? jcp.ic_block : 1;
        }
        if (jcp.ic < ic_block_step) ic_block_step = jcp.ic;
    } else {
        ic_block_step = jcp.kw > 7 ? 1 : jcp.kw > 3 ? 2 : jcp.kw > 1 ? 4 : 8;
    }

    const int max_ur_w = jcp.ow > 56 ? 14 : 28;

    if (jcp.ow <= max_ur_w || one_of(jcp.src_tag, nwc, nhwc, ndhwc))
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        od_step_comeback_pointers();
        add_d(reg_input, aux_reg_input, zero);

        add_d(reg_kernel, aux_reg_kernel, zero);
    } else {
        oh_step_comeback_pointers();
    }
}

inline void jit_lasx_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow(
        int ic_block_step, int max_ur_w) {
    UNUSED(max_ur_w);

    const int r_pad = jcp.r_pad;
    const int ic_tail = jcp.ic_tail;
    const int ic_block = jcp.ic_block;
    const int ic_block_step_tail = jcp.ic % ic_block_step;
    const size_t inp_icblk_stride = get_input_offset(ic_block_step, 0);

    if (ic_tail) {
        push_xreg(reg_ih_count);
        ld_d(reg_channel, param1, GET_OFF(channel));
    }

    Label kd_loop;
    if (jcp.ndims == 5) {
        add_d(aux_reg_input, reg_input, zero);

        add_d(aux_reg_kernel, reg_kernel, zero);

        mov_imm(ki, jcp.kd);

        L(kd_loop);
        add_d(reg_input, aux_reg_input, zero);

        add_d(reg_kernel, aux_reg_kernel, zero);
    }

    add_d(kj, reg_kh, zero);

    Label kh_loop, kh_loop_ic_tail, kh_loop_done;
    if (ic_tail) {
        mov_imm(X_TMP_0, ic_block);
        blt(reg_channel, X_TMP_0, kh_loop_ic_tail);
    }

    L(kh_loop);
    {
        xor_(b_ic, b_ic, b_ic);
        Label ic_block_loop;
        L(ic_block_loop);
        {
            compute_ic_block_step(
                    jcp.ow, jcp.l_pad, r_pad, ic_block_step, 0, 0, 0);
            add_imm(reg_input,
                    reg_input,
                    inp_icblk_stride,
                    reg_long_offt);

            add_imm(reg_kernel, reg_kernel, get_kernel_offset(0, ic_block_step),
                    X_TMP_0);
            add_imm(b_ic, b_ic, ic_block_step, X_TMP_0);

            mov_imm(X_TMP_0, ic_block);
            blt(b_ic, X_TMP_0, ic_block_loop);
        }
        add_imm(reg_input, reg_input,
                get_input_offset(0, jcp.iw) - get_input_offset(ic_block, 0),
                X_TMP_0);
        add_imm(reg_kernel, reg_kernel, get_kernel_offset((jcp.kw - 1), 0),
                X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, kj, kh_loop);
    }
    b(kh_loop_done);

    L(kh_loop_ic_tail);
    {
        Label ic_block_loop, ic_block_loop_done;

        mov_imm(X_TMP_0, ic_block_step);
        blt(reg_channel, X_TMP_0, ic_block_loop_done);

        mov_imm(b_ic, ic_tail);

        L(ic_block_loop);
        {
            compute_ic_block_step(
                    jcp.ow, jcp.l_pad, r_pad, ic_block_step, 0, 0, 0);
            add_imm(reg_input,
                    reg_input,
                    inp_icblk_stride,
                    reg_long_offt);

            add_imm(reg_kernel, reg_kernel, get_kernel_offset(0, ic_block_step),
                    X_TMP_0);

            sub_imm(b_ic, b_ic, ic_block_step, X_TMP_0);

            mov_imm(X_TMP_1, ic_block_step);
            bge(b_ic, X_TMP_1, ic_block_loop);
        }

        L(ic_block_loop_done);

        if (ic_block_step_tail) {
            compute_ic_block_step(
                    jcp.ow, jcp.l_pad, r_pad, ic_block_step_tail, 0, 0, 0);
            add_imm(reg_input, reg_input, get_input_offset(ic_block_step_tail, 0),
                    X_TMP_0);
            add_imm(reg_kernel, reg_kernel,
                    get_kernel_offset(0, ic_block_step_tail), X_TMP_0);
        }

        add_imm(reg_input, reg_input,
                get_input_offset(0, jcp.iw) - get_input_offset(ic_tail, 0),
                X_TMP_0);
        add_imm(reg_kernel, reg_kernel,
                get_kernel_offset(0, ic_block - ic_tail) + get_kernel_offset((jcp.kw - 1), 0),
                X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, kj, kh_loop_ic_tail);
    }

    L(kh_loop_done);

    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input, get_input_offset(0, jcp.ih * jcp.iw),
                X_TMP_0);
        add_imm(aux_reg_kernel, aux_reg_kernel, get_kernel_offset(jcp.kh * jcp.kw, 0),
                X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, ki, kd_loop);
    }
    if (ic_tail){
        pop_xreg(reg_ih_count);
    }
}

inline void jit_lasx_conv_bwd_weights_kernel_f32::compute_oh_step_common(
        int ic_block_step, int max_ur_w) {
    // TODO: suppport channel tails for nxc format

    const int ic_block = jcp.ic_block;
    const int stride_w = jcp.stride_w;
    Label kd_loop;

    const int r_pad = jcp.r_pad;

    int ur_w = nstl::min(jcp.ow, max_ur_w);
    int ur_w_trips = jcp.ow / ur_w;
    int ur_w_tail = jcp.ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0) || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    int input_comeback
            = get_input_offset(0, ur_w_trips * ur_w * stride_w - jcp.l_pad);
    int output_comeback = get_output_offset(0, ur_w_trips * ur_w);

    if (jcp.ndims == 5) {
        add_d(aux_reg_input, reg_input, zero);

        add_d(aux_reg_kernel, reg_kernel, zero);

        mov_imm(ki, jcp.kd);

        L(kd_loop);

        add_d(reg_input, aux_reg_input, zero);

        add_d(reg_kernel, aux_reg_kernel, zero);
    }

    add_d(kj, reg_kh, zero);

    Label kh_loop;
    L(kh_loop);
    {
        xor_(b_ic, b_ic, b_ic);
        Label ic_block_loop;
        L(ic_block_loop);
        {
            if (jcp.l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(
                        ur_w, jcp.l_pad, 0, ic_block_step, 0, 0, 0);
                add_imm(reg_input, reg_input,
                        get_input_offset(0, ur_w * stride_w - jcp.l_pad),
                        X_TMP_0);
                add_imm(reg_output, reg_output,
                        get_output_offset(0, ur_w), X_TMP_0);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips, reg_ur_w_trips);
                Label ow_block_loop;
                L(ow_block_loop);
                {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add_imm(reg_output, reg_output,
                            get_output_offset(0, ur_w), X_TMP_0);
                    add_imm(reg_input, reg_input,
                            get_input_offset(0, ur_w * stride_w), X_TMP_0);

                    addi_d(reg_ur_w_trips, reg_ur_w_trips, 1);

                    mov_imm(X_TMP_0, ur_w_trips);
                    blt(reg_ur_w_trips, X_TMP_0, ow_block_loop);
                }
            }

            if (ur_w_tail > 0)
                compute_ic_block_step(
                        ur_w_tail, 0, r_pad, ic_block_step, 0, 0, 0);

            sub_imm(reg_input, reg_input, input_comeback, X_TMP_0);

            sub_imm(reg_output, reg_output, output_comeback, X_TMP_0);

            size_t inp_icblk_stride = get_input_offset(ic_block_step, 0);
            add_imm(reg_input,
                    reg_input,
                    inp_icblk_stride,
                    reg_long_offt);

            add_imm(reg_kernel, reg_kernel,
                    get_kernel_offset(0, ic_block_step), X_TMP_0);
            add_imm(b_ic, b_ic, ic_block_step, X_TMP_0);
            mov_imm(X_TMP_0, jcp.ic_block);
            blt(b_ic, X_TMP_0, ic_block_loop);
        }
        add_imm(reg_input, reg_input,
                get_input_offset(0, jcp.iw) - get_input_offset(ic_block, 0),
                X_TMP_0);
        add_imm(reg_kernel, reg_kernel,
                get_kernel_offset((jcp.kw - 1), 0),
                X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, kj, kh_loop);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input,
                get_input_offset(0, jcp.ih * jcp.iw), X_TMP_0);
        add_imm(aux_reg_kernel, aux_reg_kernel,
                get_kernel_offset(jcp.kh * jcp.kw, 0), X_TMP_0);

        addi_d(kj, kj, -1);
        blt(zero, ki, kd_loop);
    }
}

inline void jit_lasx_conv_bwd_weights_kernel_f32::compute_oh_loop_common() {
    const int t_pad = jcp.t_pad;
    const int stride_h = jcp.stride_h;
    int b_pad = jcp.b_pad;

    Label oh_tpad_loop, oh_loop, oh_loop_end;

    mov_imm(reg_kh, jcp.kh);

    xor_(reg_ih_count, reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj, reg_oj);
    if (t_pad > 0) {
        assert(jcp.kh <= t_pad + jcp.ih); /* [bwd_w:r1] */
        mov_imm(reg_kh, jcp.kh <= t_pad + jcp.ih ? jcp.kh - t_pad : jcp.ih);

        add_imm(reg_kernel, reg_kernel,
                get_kernel_offset(t_pad * jcp.kw, 0), X_TMP_0);

        L(oh_tpad_loop);
        {
            compute_oh_step_disp();
            add_imm(reg_output, reg_output,
                    get_output_offset(0, jcp.ow), X_TMP_0);
            sub_imm(reg_kernel, reg_kernel,
                    get_kernel_offset(stride_h * jcp.kw, 0), X_TMP_0);

            addi_d(reg_oj, reg_oj, 1);

            add_imm(reg_ih_count, reg_ih_count, stride_h, X_TMP_0);
            add_imm(reg_kh, reg_kh, stride_h, X_TMP_0);

            /* the overlap between input and kernel may not reach kernel size.
             * so far we do not support that (until we put constant here) */
            const int final_inp_ker_overlap = jcp.kh; /* [bwd_w:r2] */
            mov_imm(X_TMP_0, final_inp_ker_overlap);
            blt(reg_kh, X_TMP_0, oh_tpad_loop);
        }

        if (t_pad % stride_h != 0) {
            int inp_corr = stride_h - t_pad % stride_h;
            add_imm(reg_kernel, reg_kernel,
                    get_kernel_offset(inp_corr * jcp.kw, 0), X_TMP_0);
            add_imm(reg_input, reg_input,
                    get_input_offset(0, inp_corr * jcp.iw),
                    X_TMP_0);
        }
    }
    mov_imm(X_TMP_1, jcp.ih + t_pad - jcp.kh + 1);
    bge(reg_ih_count, X_TMP_1, oh_loop_end);

    mov_imm(X_TMP_1, jcp.oh);
    bge(reg_oj, X_TMP_1, oh_loop);

    mov_imm(reg_kh, jcp.kh);

    L(oh_loop);
    {
        compute_oh_step_disp();
        add_imm(reg_input, reg_input, get_input_offset(0, stride_h * jcp.iw), X_TMP_0);
        add_imm(reg_output, reg_output, get_output_offset(0, jcp.ow), X_TMP_0);

        addi_d(reg_oj, reg_oj, 1);
        add_imm(reg_ih_count, reg_ih_count, stride_h, X_TMP_0);

        mov_imm(X_TMP_1, jcp.ih + t_pad - jcp.kh + 1);
        bge(reg_ih_count, X_TMP_1, oh_loop_end);

        mov_imm(X_TMP_0, jcp.oh);
        blt(reg_oj, X_TMP_0, oh_loop);
    }
    L(oh_loop_end);
    if (b_pad > 0) {
        Label oh_bpad_loop, oh_bpad_loop_end;
        mov_imm(X_TMP_1, jcp.oh);
        bge(reg_oj, X_TMP_1, oh_bpad_loop_end);

        mov_imm(reg_kh, jcp.ih + t_pad);

        sub_d(reg_kh, reg_kh, reg_ih_count);

        L(oh_bpad_loop);
        {
            compute_oh_step_disp();
            addi_d(reg_input, reg_input, get_input_offset(0, stride_h * jcp.iw));
            addi_d(reg_output, reg_output, get_output_offset(0, jcp.ow));

            sub_imm(reg_kh, reg_kh, stride_h, X_TMP_0);

            bge(zero, reg_kh, oh_bpad_loop_end);

            addi_d(reg_oj, reg_oj, 1);

            mov_imm(X_TMP_0, jcp.oh);
            blt(reg_oj, X_TMP_0, oh_bpad_loop);
        }
        L(oh_bpad_loop_end);
    }
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
