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

#include <assert.h>
#include <limits>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/injectors/injector_utils.hpp"
#include "cpu/loongarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/loongarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/loongarch64/jit_lasx_1x1_conv_kernel_f32.hpp"
#include "cpu/loongarch64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;

using namespace Xbyak_loongarch64;

jit_lasx_1x1_conv_kernel_f32::jit_lasx_1x1_conv_kernel_f32(
        const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
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

        rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx, a6, a2,
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

void jit_lasx_1x1_conv_kernel_f32::generate_bcast_loop(int load_loop_blk) {
    add_d(aux1_reg_bcast_data, reg_bcast_data, zero);
    add_d(aux_reg_output_data, reg_output_data, zero);
    add_d(bcast_loop_iter, reg_bcast_loop_work, zero);

    Label bcast_loop, bcast_loop_tail, large_tail;

    mov_imm(X_TMP_0, jcp.bcast_block);
    blt(bcast_loop_iter, X_TMP_0, bcast_loop_tail);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        const int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            if (i == num_substeps - 1) L(large_tail);
            generate_reduce_loop(load_loop_blk, jcp.ur);
            if (i < num_substeps - 1) {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep, X_TMP_0);
                add_imm(aux_reg_output_data, aux_reg_output_data, jcp.bcast_loop_output_substep, X_TMP_0);
            } else {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step - (num_substeps - 1) * jcp.bcast_loop_bcast_substep, X_TMP_0);
                add_imm(aux_reg_output_data, aux_reg_output_data,
                            jcp.bcast_loop_output_step - (num_substeps - 1) * jcp.bcast_loop_output_substep, X_TMP_0);
            }
            add_imm(bcast_loop_iter, bcast_loop_iter, -1 * jcp.ur, X_TMP_0);
        }
        mov_imm(X_TMP_0, jcp.bcast_block);
        bge(bcast_loop_iter, X_TMP_0, bcast_loop);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        if (jcp.ur_tail >= jcp.ur) {
            mov_imm(X_TMP_0, jcp.ur);
            bge(bcast_loop_iter, X_TMP_0, large_tail);
        }
        if (jcp.ur_tail % jcp.ur > 0) {
            bge(zero, bcast_loop_iter, bcast_loop_tail_out);
            generate_reduce_loop(load_loop_blk, jcp.ur_tail % jcp.ur);
            L(bcast_loop_tail_out);
        }
    }
}

static int vreg_accum_idx(const int load_loop_blk, int i, int j) {
    return (j * load_loop_blk + i);
}

static XVReg vreg_accum(const int load_loop_blk, int i, int j) {
    return XVReg(vreg_accum_idx(load_loop_blk, i, j));
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

void jit_lasx_1x1_conv_kernel_f32::apply_postops(
        const int load_loop_blk, const int ur, const int load_dim_tail) {
    if (jcp.with_eltwise || jcp.with_binary) {
        assert(ur * load_loop_blk < 14);

        Label store_nopost_ops;
        andi(X_TMP_0, reg_reduce_pos_flag, FLAG_REDUCE_LAST);
        beqz(X_TMP_0, store_nopost_ops);

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            const auto oc_off_oprnd = a5;

            iterate(load_loop_blk, ur, load_dim_tail,
                    [&](const bool mask_flag, const int i, const int j) {
                        const int aux_output_offset
                                = (i * get_output_i_offset(jcp)
                                        + j * get_output_j_offset(jcp));
                        const auto vmm_idx
                                = vreg_accum_idx(load_loop_blk, i, j);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_addr.emplace(
                                vmm_idx, ptr_a(param1, GET_OFF(oc_l_off)));
                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_val.emplace(
                                vmm_idx, i * jcp.oc_block);
                        rhs_arg_params_tail.vmm_idx_to_oc_off_oprnd.emplace(
                                vmm_idx, oc_off_oprnd);
                        rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_offset);
                        rhs_arg_params_tail.vmm_idx_to_out_off_oprnd.emplace(
                                vmm_idx, oc_off_oprnd);
                        if (mask_flag)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

            const injector_utils::register_preserve_guard_t register_guard(
                    this, {abi_param1, oc_off_oprnd});
            const size_t reg_guard_stack_occupied
                    = register_guard.stack_space_occupied();
            uni_ld_d(abi_param1, sp, reg_abi_param1_backup + reg_guard_stack_occupied);
            uni_ld_d(oc_off_oprnd, sp, reg_binary_post_op_acc_off + reg_guard_stack_occupied);

            Label postops_done;
            if (load_dim_tail) {
                Label postops_no_tail;
                mov_imm(X_TMP_0, load_loop_blk * jcp.load_loop_iter_step);
                bge(reg_load_loop_work, X_TMP_0, postops_no_tail);
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail);
                b(postops_done);
                L(postops_no_tail);
            }
            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
            L(postops_done);
        } else {
            iterate(load_loop_blk, ur, load_dim_tail,
                    [&](const bool, const int i, const int j) {
                        vmm_idxs.emplace(vreg_accum_idx(load_loop_blk, i, j));
                    });
            postops_injector_->compute_vector_range(vmm_idxs);
        }
        L(store_nopost_ops);
    }
};

void jit_lasx_1x1_conv_kernel_f32::generate_reduce_loop(
        int load_loop_blk, int ur) {
    const int load_dim_tail
            = ((jcp.with_binary
                       && one_of(jcp.prop_kind, forward_training,
                               forward_inference))
                              ? jcp.oc_without_padding
                              : jcp.load_dim)
            % jcp.load_block;
    const int reduce_dim_tail = jcp.reduce_dim % jcp.reduce_block;

    auto vreg_load = [=](int i) { return XVReg(ur * load_loop_blk + i); };

    auto get_load_offset_bwd_w = [=](int u, int i) {
        size_t u0 = u % jcp.reduce_loop_unroll;
        size_t u1 = u / jcp.reduce_loop_unroll;
        return u1 * jcp.reduce_loop_load_step
                + sizeof(float) * get_load_bwd_w_offset(jcp, i, u0);
    };

    auto load_ptr = [=](int u, int i) {
        size_t offt;
        size_t u0 = u % jcp.reduce_loop_unroll;
        size_t u1 = u / jcp.reduce_loop_unroll;
        switch (jcp.prop_kind) {
            case backward_data:
                offt = (i * jcp.oc_block + u0) * jcp.ic_block;
                break;
            case backward_weights:
                offt = get_load_bwd_w_offset(jcp, i, u0);
                break;
            default:
                offt = (i * rnd_up(jcp.ic, jcp.ic_block) + u0) * jcp.oc_block;
        }
        return (u1 * jcp.reduce_loop_load_step + sizeof(float) * offt);
    };

    auto get_output_offset = [=](int i, int j) {
        switch (jcp.prop_kind) {
            case backward_weights: return sizeof(float) * jcp.oc_block * j;
            case backward_data:
            default:
                return (i * get_output_i_offset(jcp)
                               + j * get_output_j_offset(jcp))
                        * sizeof(float);
        }
    };

    auto output_ptr = [=](int i, int j) {
        switch (jcp.prop_kind) {
            case backward_weights:
                return (sizeof(float) * jcp.oc_block * j);
            case backward_data:
            default:
                return ((i * get_output_i_offset(jcp) + j * get_output_j_offset(jcp))
                                * sizeof(float));
        }
    };

    auto init = [=]() {
        Label init_done, init_zero;

        if (jcp.with_bias
                && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            andi(X_TMP_0, reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            beqz(X_TMP_0, init_zero);

            for (int i = 0; i < load_loop_blk; i++) {
                for (int j = 0; j < ur; ++j) {
                    if (load_dim_tail > 0 && i == load_loop_blk - 1) {
                        Label load_bias_tail, load_bias_done;
                        mov_imm(X_TMP_0, load_loop_blk * jcp.load_loop_iter_step);
                        blt(reg_load_loop_work, X_TMP_0, load_bias_tail);
                        uni_xvld(vreg_accum(load_loop_blk, i, j),
                            reg_bias_data, sizeof(float) * jcp.oc_block * i);
                        b(load_bias_done);

                        L(load_bias_tail);
                        load_bytes(vreg_accum(load_loop_blk, i, j),
                                reg_bias_data, i * jcp.oc_block * sizeof(float),
                                load_dim_tail * sizeof(float));
                        L(load_bias_done);
                    } else {
                        uni_xvld(vreg_accum(load_loop_blk, i, j),
                            reg_bias_data, sizeof(float) * jcp.oc_block * i);
                    }
                }
            }
            b(init_done);
        }

        L(init_zero);
        for (int i = 0; i < load_loop_blk; ++i)
            for (int j = 0; j < ur; ++j) {
                auto r = vreg_accum(load_loop_blk, i, j);
                xvxor_v(r, r, r);
            }

        L(init_done);
        for (int i = 0; i < load_loop_blk; ++i) {
            if (jcp.prop_kind == backward_weights && load_dim_tail > 0
                    && i == load_loop_blk - 1) {
                Label load_init_tail, load_init_done;
                mov_imm(X_TMP_0, load_loop_blk * jcp.load_loop_iter_step);
                blt(reg_load_loop_work, X_TMP_0, load_init_tail);
                uni_xvld(vreg_load(i), aux_reg_load_data, load_ptr(0, i));
                b(load_init_done);

                L(load_init_tail);
                xvxor_v(vreg_load(i), vreg_load(i), vreg_load(i));
                load_bytes(vreg_load(i), aux_reg_load_data,
                        get_load_offset_bwd_w(0, i),
                        load_dim_tail * sizeof(float));
                L(load_init_done);
            } else {
                uni_xvld(vreg_load(i), aux_reg_load_data, load_ptr(0, i));
            }
        }
        uni_xvldrepl_w(vreg_bcast, aux_reg_bcast_data, get_bcast_offset(jcp, 0, 0));
    };

    auto store = [=]() {
        Label store_noadd;

        if (!jcp.with_sum) {
            andi(X_TMP_0, reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            bnez(X_TMP_0, store_noadd);
        }

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                auto r = vreg_accum(load_loop_blk, i, j);
                if (jcp.with_sum && load_dim_tail > 0
                        && i == load_loop_blk - 1) {
                    Label sum_tail, sum_done;
                    mov_imm(X_TMP_0, load_loop_blk * jcp.load_loop_iter_step);
                    blt(reg_load_loop_work, X_TMP_0, sum_tail);
                    if (jcp.prop_kind == backward_weights && i != 0) {
                        mov_imm(X_TMP_1, i);
                        mul_d(X_TMP_1, reg_output_stride, X_TMP_1);
                        add_d(X_TMP_1, aux_reg_output_data, X_TMP_1);
                        uni_xvld(XVReg(30), X_TMP_1, output_ptr(i, j));
                    } else {
                        uni_xvld(XVReg(30), aux_reg_output_data, output_ptr(i, j));
                    }
                    xvfadd_s(r, r, XVReg(30));
                    b(sum_done);

                    L(sum_tail);
                    load_bytes(vtmp, aux_reg_output_data,
                            get_output_offset(i, j),
                            load_dim_tail * sizeof(float));
                    xvfadd_s(r, r, vtmp);
                    L(sum_done);
                } else {
                    if (jcp.prop_kind == backward_weights && i != 0) {
                        mov_imm(X_TMP_1, i);
                        mul_d(X_TMP_1, reg_output_stride, X_TMP_1);
                        add_d(X_TMP_1, aux_reg_output_data, X_TMP_1);
                        uni_xvld(XVReg(30), X_TMP_1, output_ptr(i, j));
                    } else {
                        uni_xvld(XVReg(30), aux_reg_output_data, output_ptr(i, j));
                    }
                    xvfadd_s(r, r, XVReg(30));
                }
            }

        L(store_noadd);

        apply_postops(load_loop_blk, ur, load_dim_tail);

        if (jcp.prop_kind == backward_weights && load_dim_tail > 0) {
            push_xreg(reg_bcast_data);
            push_xreg(aux_reg_bcast_data);
        }

        const auto is_padding = jcp.oc_without_padding != jcp.oc;
        if (is_padding) uni_vpxor(vtmp, vtmp, vtmp);
        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
                if (load_dim_tail > 0 && i == load_loop_blk - 1) {
                    Label store_tail, store_done;
                    mov_imm(X_TMP_0, load_loop_blk * jcp.load_loop_iter_step);
                    blt(reg_load_loop_work, X_TMP_0, store_tail);
                    if (jcp.prop_kind == backward_weights && i != 0) {
                        mov_imm(X_TMP_1, i);
                        mul_d(X_TMP_1, reg_output_stride, X_TMP_1);
                        add_d(X_TMP_1, aux_reg_output_data, X_TMP_1);
                        uni_xvst(vreg_accum(load_loop_blk, i, j), X_TMP_1, output_ptr(i, j));
                    } else {
                        uni_xvst(vreg_accum(load_loop_blk, i, j), aux_reg_output_data, output_ptr(i, j));
                    }
                    b(store_done);

                    L(store_tail);
                    if (jcp.prop_kind == backward_weights) {
                        if (i) {
                            add_d(reg_tmp_output_stride, reg_output_stride, zero);
                            mov_imm(reg_output_stride_scale, i);
                            mul_d(reg_tmp_output_stride, reg_output_stride, reg_output_stride_scale);
                        } else {
                            xor_(reg_tmp_output_stride, reg_tmp_output_stride, reg_tmp_output_stride);
                        }
                        add_d(X_TMP_1, aux_reg_output_data, reg_tmp_output_stride);
                        uni_xvst(vreg_accum(load_loop_blk, i, j), X_TMP_1, output_ptr(i, j));
                    } else {
                        if (is_padding && jcp.with_binary) {
                            uni_xvst(vtmp, aux_reg_output_data, get_output_offset(i, j));
                        }
                        store_bytes(vreg_accum(load_loop_blk, i, j),
                                aux_reg_output_data, get_output_offset(i, j),
                                load_dim_tail * sizeof(float));
                    }
                    L(store_done);
                } else {
                    if (jcp.prop_kind == backward_weights && i != 0) {
                        mov_imm(X_TMP_1, i);
                        mul_d(X_TMP_1, reg_output_stride, X_TMP_1);
                        add_d(X_TMP_1, aux_reg_output_data, X_TMP_1);
                        uni_xvst(vreg_accum(load_loop_blk, i, j), X_TMP_1, output_ptr(i, j));
                    } else {
                        uni_xvst(vreg_accum(load_loop_blk, i, j), aux_reg_output_data, output_ptr(i, j));
                    }
                }
            }

        if (jcp.prop_kind == backward_weights && load_dim_tail > 0) {
            pop_xreg(aux_reg_bcast_data);
            pop_xreg(reg_bcast_data);
        }
    };

    auto fma_block = [=](bool last_block) {
        const bool is_tail = reduce_dim_tail && last_block;
        const int u_end = is_tail ? reduce_dim_tail : jcp.reduce_loop_unroll;
        for (int u = 0; u < u_end; ++u) {
            for (int j = 0; j < ur; ++j) {
                for (int i = 0; i < load_loop_blk; ++i) {
                    if (jcp.isa == lasx)
                        xvfmadd_s(vreg_accum(load_loop_blk, i, j),
                                vreg_load(i), vreg_bcast, vreg_accum(load_loop_blk, i, j));
                    else {
                        xvfmul_s(vtmp, vreg_bcast, vreg_load(i));
                        xvfadd_s(vreg_accum(load_loop_blk, i, j),
                                vreg_accum(load_loop_blk, i, j), vtmp);
                    }
                    if (j == ur - 1 && !(last_block && u == u_end - 1)) {
                        if (jcp.prop_kind == backward_weights
                                && load_dim_tail > 0
                                && i == load_loop_blk - 1) {
                            Label fma_load_tail, fma_load_done;
                            mov_imm(X_TMP_0, load_loop_blk * jcp.load_loop_iter_step);
                            blt(reg_load_loop_work, X_TMP_0, fma_load_tail);
                            uni_xvld(vreg_load(i), aux_reg_load_data, load_ptr(u + 1, i));
                            b(fma_load_done);

                            L(fma_load_tail);
                            xvxor_v(vreg_load(i), vreg_load(i), vreg_load(i));
                            load_bytes(vreg_load(i), aux_reg_load_data,
                                    get_load_offset_bwd_w(u + 1, i),
                                    load_dim_tail * sizeof(float));
                            L(fma_load_done);
                        } else {
                            uni_xvld(vreg_load(i), aux_reg_load_data, load_ptr(u + 1, i));
                        }
                    }
                }
                if (j < ur - 1) {
                    assert(j + 1 < jcp.ur);
                    assert(u <= jcp.reduce_loop_unroll);
                    uni_xvldrepl_w(vreg_bcast, aux_reg_bcast_data, get_bcast_offset(jcp, u, j + 1));
                }
            }
            if (!last_block || u < u_end - 1) {
                assert(u + 1 <= jcp.reduce_loop_unroll);
                uni_xvldrepl_w(vreg_bcast, aux_reg_bcast_data, get_bcast_offset(jcp, u + 1, 0));
            }
        }
    };

    Label reduce_loop, reduce_loop_tail;

    add_d(aux_reg_load_data, reg_load_data, zero);
    add_d(aux_reg_bcast_data, aux1_reg_bcast_data, zero);

    init();

    add_d(reduce_loop_iter, reg_reduce_loop_work, zero);
    add_imm(reduce_loop_iter, reduce_loop_iter, -1 * jcp.reduce_loop_unroll, X_TMP_0);
    bge(zero, reduce_loop_iter, reduce_loop_tail);

    L(reduce_loop);
    {
        fma_block(false);
        add_imm(aux_reg_bcast_data, aux_reg_bcast_data, jcp.reduce_loop_bcast_step, X_TMP_0);
        add_imm(aux_reg_load_data, aux_reg_load_data, jcp.reduce_loop_load_step, X_TMP_0);
        add_imm(reduce_loop_iter, reduce_loop_iter, -1 * jcp.reduce_loop_unroll, X_TMP_0);
        blt(zero, reduce_loop_iter, reduce_loop);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_lasx_1x1_conv_kernel_f32::generate_diff_bias_loop(int load_loop_blk) {
    if (!jcp.with_bias || jcp.prop_kind != backward_weights) return;

    Label diff_bias_loop, diff_bias_loop_out, diff_bias_init_out;
    Label diff_bias_load;

    auto diff_bias_reg = [=](int i) { return XVReg(i); };

    uni_ld_d(reg_diff_bias_data, sp, reg_diff_bias_data_stack_offt);
    beqz(reg_diff_bias_data, diff_bias_loop_out);

    andi(X_TMP_0, reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
    beqz(X_TMP_0, diff_bias_load);

    for (int i = 0; i < load_loop_blk; ++i) {
        auto r = diff_bias_reg(i);
        xvxor_v(r, r, r);
    }
    b(diff_bias_init_out);

    L(diff_bias_load);
    for (int i = 0; i < load_loop_blk; ++i)
        uni_xvld(diff_bias_reg(i), reg_diff_bias_data, i * jcp.oc_block * sizeof(float));

    L(diff_bias_init_out);
    add_d(aux_reg_load_data, reg_load_data, zero);
    add_d(reduce_loop_iter, reg_reduce_loop_work, zero);
    L(diff_bias_loop);
    {
        for (int u = 0; u < jcp.reduce_loop_unroll; ++u) {
            for (int i = 0; i < load_loop_blk; ++i) {
                uni_xvld(XVReg(30), aux_reg_load_data, (i * jcp.os + u) * jcp.oc_block * sizeof(float));
                xvfadd_s(diff_bias_reg(i), diff_bias_reg(i), XVReg(30));
            }
        }
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
        add_imm(aux_reg_load_data, aux_reg_load_data, jcp.reduce_loop_load_step, X_TMP_0);
        add_imm(reduce_loop_iter, reduce_loop_iter, -1 * jcp.reduce_loop_unroll, X_TMP_0);
        bnez(reduce_loop_iter, diff_bias_loop);
    }

    for (int i = 0; i < load_loop_blk; i++)
        uni_xvst(diff_bias_reg(i), reg_diff_bias_data, i * jcp.oc_block * sizeof(float));
    add_imm(reg_diff_bias_data, reg_diff_bias_data, load_loop_blk * jcp.oc_block * sizeof(float), X_TMP_0);
    uni_st_d(reg_diff_bias_data, sp, reg_diff_bias_data_stack_offt);

    L(diff_bias_loop_out);
}

void jit_lasx_1x1_conv_kernel_f32::generate() {
    preamble();

    if (jcp.with_binary || (jcp.with_bias && jcp.prop_kind == backward_weights))
        addi_d(sp, sp, -1 * stack_space_needed);

    if (jcp.with_binary) {
        const auto zeroed_reg = a3;
        xor_(zeroed_reg, zeroed_reg, zeroed_reg);
        uni_st_d(zeroed_reg, sp, reg_binary_post_op_acc_off);
        uni_st_d(abi_param1, sp, reg_abi_param1_backup);
    }

    ld_d(reg_bcast_data, param1, GET_OFF(bcast_data));
    ld_d(reg_load_data, param1, GET_OFF(load_data));
    ld_d(reg_output_data, param1, GET_OFF(output_data));
    if (jcp.with_bias) {
        if (jcp.prop_kind == backward_weights) {
            ld_d(reg_diff_bias_data, param1, GET_OFF(bias_data));
            uni_st_d(reg_diff_bias_data, sp, reg_diff_bias_data_stack_offt);
        } else
            ld_d(reg_bias_data, param1, GET_OFF(bias_data));
    }

    ld_d(reg_load_loop_work, param1, GET_OFF(load_dim));
    ld_d(reg_bcast_loop_work, param1, GET_OFF(bcast_dim));
    ld_d(reg_reduce_loop_work, param1, GET_OFF(reduce_dim));
    ld_d(reg_reduce_pos_flag, param1, GET_OFF(first_last_flag));
    if (jcp.prop_kind == backward_weights)
        ld_d(reg_output_stride, param1, GET_OFF(output_stride));

    auto generate_load_loop_body = [=](int load_loop_blk) {
        generate_bcast_loop(load_loop_blk);
        add_imm(reg_load_data, reg_load_data, load_loop_blk * jcp.load_loop_load_step, X_TMP_0);
        switch (jcp.prop_kind) {
            case forward_training:
            case forward_inference:
                add_imm(reg_bias_data, reg_bias_data,
                        load_loop_blk * jcp.oc_block * sizeof(float), X_TMP_0);
                add_imm(reg_output_data, reg_output_data,
                        get_load_loop_output_fwd_offset(jcp, load_loop_blk), X_TMP_0);
                if (jcp.with_binary) {
                    uni_ld_d(aux_reg_load_data, sp, reg_binary_post_op_acc_off);
                    add_imm(aux_reg_load_data, aux_reg_load_data, jcp.load_block * load_loop_blk, X_TMP_0);
                    uni_st_d(aux_reg_load_data, sp, reg_binary_post_op_acc_off);
                }
                break;
            case backward_data:
                add_imm(reg_output_data, reg_output_data,
                        get_load_loop_output_bwd_d_offset(jcp, load_loop_blk), X_TMP_0);
                break;
            case backward_weights:
                for (int i = 0; i < load_loop_blk; i++)
                    add_d(reg_output_data, reg_output_data, reg_output_stride);
                break;
            default: assert(!"invalid prop_kind");
        }
        add_imm(reg_load_loop_work, reg_load_loop_work, -1 * load_loop_blk * jcp.load_loop_iter_step, X_TMP_0);
    };

    Label load_loop_blk_8;
    Label load_loop_blk_16;
    Label load_loop_blk_24;
    Label load_loop_blk_end;

    mov_imm(X_TMP_0, 8);
    bge(X_TMP_0, reg_load_loop_work, load_loop_blk_8);

    mov_imm(X_TMP_0, 32);
    beq(reg_load_loop_work, X_TMP_0, load_loop_blk_16);

    mov_imm(X_TMP_0, 16);
    bge(X_TMP_0, reg_load_loop_work, load_loop_blk_16);

    L(load_loop_blk_24);
    {
        generate_diff_bias_loop(3);
        generate_load_loop_body(3);
        mov_imm(X_TMP_0, 32);
        beq(reg_load_loop_work, X_TMP_0, load_loop_blk_16);
        mov_imm(X_TMP_0, 24);
        bge(reg_load_loop_work, X_TMP_0, load_loop_blk_24);
    }

    mov_imm(X_TMP_0, 8);
    bge(X_TMP_0, reg_load_loop_work, load_loop_blk_8);

    L(load_loop_blk_16);
    {
        generate_diff_bias_loop(2);
        generate_load_loop_body(2);
        mov_imm(X_TMP_0, 16);
        bge(reg_load_loop_work, X_TMP_0, load_loop_blk_16);
    }

    L(load_loop_blk_8);
    {
        bge(zero, reg_load_loop_work, load_loop_blk_end);
        generate_diff_bias_loop(1);
        generate_load_loop_body(1);
    }

    L(load_loop_blk_end);

    if (jcp.with_binary || (jcp.with_bias && jcp.prop_kind == backward_weights))
        addi_d(sp, sp, stack_space_needed);

    postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

status_t jit_lasx_1x1_conv_kernel_f32::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr) {
    if (!mayiuse(lasx)) return status::unimplemented;
    jcp.isa = lasx;

    // configuration struct could do some stuff below
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();

    jcp.nthr = dnnl_get_max_threads();

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.ic = jcp.ic_without_padding;

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

    jcp.with_bias = pick_by_prop_kind(jcp.prop_kind, cd.bias_desc.format_kind,
                            format_kind::undef, cd.diff_bias_desc.format_kind)
            != format_kind::undef;

    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;

    jcp.typesize_in = sizeof(prec_traits<data_type::f32>::type);
    jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);

    const auto &post_ops = attr.post_ops_;
    const int dw_conv_ind = post_ops.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;

    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    const int sum_ind = post_ops.find(primitive_kind::sum, 0, dw_conv_ind);
    jcp.with_sum = sum_ind != -1;
    const int eltwise_ind
            = post_ops.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;
    const int binary_ind
            = post_ops.find(primitive_kind::binary, 0, dw_conv_ind);
    jcp.with_binary = binary_ind != -1;

    if (dw_conv_ind >= 0) {
        // dw_conv and post_ops after it are handled externally, so skip them
        jcp.post_ops.entry_.assign(post_ops.entry_.cbegin(),
                post_ops.entry_.cbegin() + dw_conv_ind);
    } else {
        jcp.post_ops = post_ops;
    }

    const auto dat_tag_nxc = utils::pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx8c = utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
    const bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
    const auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;

    const int is_bwd_d = jcp.prop_kind == backward_data;
    format_tag_t wei_tag = with_groups
            ? utils::pick(2 * ndims - 6 + is_bwd_d, gOIw8i8o, gOIw8o8i,
                    gOIhw8i8o, gOIdhw8o8i, gOIhw8i8o, gOIdhw8o8i)
            : utils::pick(2 * ndims - 6 + is_bwd_d, OIw8i8o, OIw8o8i, OIhw8i8o,
                    OIhw8o8i, OIdhw8i8o, OIdhw8o8i);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);

    const int simd_w = 8;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    if (jcp.with_eltwise || jcp.with_binary)
        if (jcp.isa < lasx) return status::unimplemented;

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    const bool post_ops_ok_ = post_ops_ok({lasx, {eltwise, binary, sum},
            jcp.post_ops, &dst_d, sum_at_pos_0_only, sum_requires_scale_one,
            {broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc}});
    if (!post_ops_ok_) return status::unimplemented;

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == dat_tag
            && jcp.wei_tag == wei_tag && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    args_ok = true && jcp.id == jcp.od && jcp.ih == jcp.oh && jcp.iw == jcp.ow
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0)
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    // TODO: remove this restriction
    // optimized 1x1 bwd_w does not support
    if (jcp.prop_kind == backward_weights && jcp.isa != lasx)
        return status::unimplemented;

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.ur = jcp.isa == lasx ? 4 : 3; // support
    if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);

    int load_blocking {0};
    int load_blocking_max {0};
    int bcast_blocking {0};
    int bcast_blocking_max {0};
    int reduce_blocking {0};
    int reduce_blocking_max {0};

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.reduce_dim = jcp.ic;
        jcp.reduce_block = jcp.ic_block;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.is;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? 1 : jcp.is) * sizeof(float);
        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block) * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block) * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step
                = rnd_up(jcp.ic, jcp.ic_block) * jcp.oc_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        load_blocking = is_data_layout_nxc
                ? jcp.load_dim
                : 120; // assumes the kernel is jcp.ur x 3
        load_blocking_max = is_data_layout_nxc ? jcp.load_dim : 144;
        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 192;
        reduce_blocking = is_data_layout_nxc ? jcp.reduce_dim
                                             : 128; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_data) {
        jcp.reduce_dim = jcp.oc;
        jcp.reduce_block = jcp.oc_block;

        jcp.load_dim = jcp.ic;
        jcp.load_block = jcp.ic_block;

        jcp.bcast_dim = jcp.os;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? 1 : jcp.os) * sizeof(float);
        jcp.reduce_loop_load_step = jcp.reduce_loop_unroll
                * rnd_up(jcp.ic, jcp.ic_block) * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block) * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block) * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.ic_block;

        load_blocking = is_data_layout_nxc
                ? jcp.load_dim
                : 96; // assumes the kernel is jcp.ur x 3
        load_blocking_max = is_data_layout_nxc ? jcp.load_dim : 144;

        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 196;
        reduce_blocking = is_data_layout_nxc ? jcp.reduce_dim
                                             : 64; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_weights) {
        jcp.reduce_dim = jcp.os;
        jcp.reduce_block = 1;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block) * sizeof(float);
        jcp.reduce_loop_load_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block) * sizeof(float);

        jcp.bcast_loop_output_step
                = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = jcp.oc_block * jcp.ur * sizeof(float);
        jcp.bcast_loop_bcast_step = jcp.ic_block
                * (is_data_layout_nxc ? 1 : jcp.is) * sizeof(float);
        jcp.bcast_loop_bcast_substep = jcp.ur * sizeof(float);

        jcp.load_loop_load_step = jcp.oc_block
                * (is_data_layout_nxc ? 1 : jcp.os) * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        const bool no_load_tail = jcp.load_dim % jcp.load_block == 0;
        const bool modify_load_blocking
                = IMPLICATION(is_data_layout_nxc, no_load_tail);
        while (modify_load_blocking) {
            if (load_blocking <= 32)
                break;
            else if (load_blocking % 2 == 0)
                load_blocking /= 2;
            else if (load_blocking % 3 == 0)
                load_blocking /= 3;
            else
                break;
        }
        load_blocking *= jcp.load_block;
        load_blocking_max = load_blocking;
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.load_dim % load_blocking == 0));

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        const int bcast_blocking_lim = is_data_layout_nxc ? 17 : 9;
        const bool no_bcast_tail = jcp.bcast_dim % jcp.bcast_block == 0;
        const bool small_size_for_bcast
                = static_cast<dim_t>(jcp.id) * jcp.ih * jcp.iw <= 1024;

        // TODO Verify if the size limitation helps for blocked format as well
        const bool modify_bcast_blocking = IMPLICATION(
                is_data_layout_nxc, no_bcast_tail && small_size_for_bcast);

        while (modify_bcast_blocking) {
            if (bcast_blocking <= bcast_blocking_lim)
                break;
            else if (bcast_blocking % 2 == 0)
                bcast_blocking /= 2;
            else if (bcast_blocking % 3 == 0)
                bcast_blocking /= 3;
            else
                break;
        }
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.bcast_dim % bcast_blocking == 0));

        reduce_blocking = is_data_layout_nxc
                ? rnd_up(nstl::min(jcp.ow, 128), jcp.reduce_block)
                : 128; // affects L1$ utilization
        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);

    assert(jcp.bcast_block % jcp.ur == 0);
    jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.bcast_block;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = div_up(load_blocking, jcp.load_block);
    jcp.nb_load_blocking_max = div_up(load_blocking_max, jcp.load_block);
    jcp.nb_reduce_blocking = div_up(reduce_blocking, jcp.reduce_block);
    jcp.nb_reduce_blocking_max = div_up(reduce_blocking_max, jcp.reduce_block);

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    if (jcp.prop_kind == backward_weights) {
        const auto mb_with_nb_reduce
                = static_cast<dim_t>(jcp.mb) * jcp.nb_reduce;
        // prevent too large argument to cpu reducer
        if (mb_with_nb_reduce > std::numeric_limits<int>::max())
            return status::unimplemented;
    }

    return status::success;
}

void jit_lasx_1x1_conv_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {
    using namespace dnnl::impl::memory_tracking::names;

    if (jcp.with_bias && jcp.prop_kind != backward_data
            && (jcp.oc != jcp.oc_without_padding // blocked format
                    || (jcp.prop_kind == backward_weights // nxc format
                            && jcp.oc % jcp.oc_block != 0))) {
        const size_t nelems_padded_bias
                = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book<float>(key_conv_padded_bias, nelems_padded_bias);
    }
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
