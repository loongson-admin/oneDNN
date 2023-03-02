/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "cpu/loongarch64/cpu_isa_traits.hpp"
#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/gemm/f32/jit_lasx_gemv_t_f32_kern.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace Xbyak_loongarch64;

// Convert between vector register lengths.
static inline VReg make_xmm(const XVReg &v) {
    return VReg(v.getIdx());
}

// Inner loop.
void jit_lasx_gemv_t_f32_kern::innerloop(int unroll_m, int unroll_n) {
    if ((unroll_m > M_UNROLL_) || (unroll_n > N_UNROLL_) || (unroll_m < 0)
            || (unroll_n < 0))
        return;

    int um_vecs = (unroll_m + 7) >> 3;
    int load_size = (unroll_m > 8 ? 8 : unroll_m) * size_;

    // Load x.
    for (int i = 0; i < um_vecs; i++) {
        load_bytes(x_regs_[i], XO_, size_ * (8 * i - offset_x_), load_size);
    }
    add_imm(XO_, XO_, size_ * unroll_m, X_TMP_0);

    slli_d(t0, LDA_, 1);
    add_d(t1, t0, LDA_);

    // Load A
    for (int j = 0; j < unroll_n; j++) {
        for (int i = 0; i < um_vecs; i++) {
            XVReg a = a_regs_[i][j];

            if (j > 0) {
                add_d(X_TMP_1, AO_, (j == 1 ? LDA_ : (j == 2 ? t0 : t1)));
                load_bytes(a, X_TMP_1, size_ * (8 * i - offset_a_), load_size);
            } else
                load_bytes(a, AO_, size_ * (8 * i - offset_a_), load_size);
        }
    }

    add_imm(AO_, AO_, size_ * unroll_m, X_TMP_0);

    for (int j = 0; j < unroll_n; j++) {
        XVReg acc = acc_[j];

        for (int i = 0; i < um_vecs; i++) {
            xvfmadd_s(acc, x_regs_[i], a_regs_[i][j], acc);
        }
    }
}

// Outer loop.
void jit_lasx_gemv_t_f32_kern::outerloop(
        int unroll_x, int unroll_y, Label *&cur_outerloop_label) {
    if ((unroll_x > M_UNROLL_) || (unroll_y > N_UNROLL_) || (unroll_y < 0)
            || (unroll_x < 0))
        return;

    Label label_m_loop, label_n_loop, label_m_remainder_loops[5];

    L(*cur_outerloop_label);
    cur_outerloop_label++;
    if (unroll_y >= N_UNROLL_) {
        add_d(I_, N_, zero);
        mov_imm(X_TMP_0, unroll_y);
        blt(I_, X_TMP_0, *cur_outerloop_label);
    } else {
        andi(X_TMP_0, I_, unroll_y);
        bge(zero, X_TMP_0, *cur_outerloop_label);
    }

    L_aligned(label_n_loop);
    {

        add_d(YO_, Y_, zero);
        mov_imm(X_TMP_0, unroll_y);
        mul_d(X_TMP_0, INCY_, X_TMP_0);
        add_d(Y_, YO_, X_TMP_0);

        add_d(AO_, A_, zero);
        mov_imm(X_TMP_0, unroll_y);
        mul_d(X_TMP_0, LDA_, X_TMP_0);
        add_d(A_, AO_, X_TMP_0);

        add_d(XO_, X_, zero);

        for (int i = 0; i < unroll_y; i++) {
            auto acc = acc_[i];
            xvxor_v(acc, acc, acc);
        }

        add_d(J_, M_, zero);
        mov_imm(X_TMP_0, unroll_x);
        blt(J_, X_TMP_0, label_m_remainder_loops[0]);

        L_aligned(label_m_loop);
        {
            innerloop(unroll_x, unroll_y);
            add_imm(J_, J_, -1 * unroll_x, X_TMP_0);
            mov_imm(X_TMP_0, unroll_x);
            bge(J_, X_TMP_0, label_m_loop);
        }


        // Update y.
        for (int j = 0; j < unroll_y; j++) {
            XVReg acc = acc_[j];

            xvpickev_w(xr30, acc, acc);
            xvpickod_w(xr31, acc, acc);
            xvfadd_s(acc, xr31, xr30);
            xvpermi_q(scratch_, acc, 0x1);
            xvfadd_s(acc, acc, scratch_);
            xvpickev_w(xr30, acc, acc);
            xvpickod_w(xr31, acc, acc);
            xvfadd_s(acc, xr31, xr30);
        }
        for (int j = 0; j < unroll_y; j++) {
            // TODO Handle negative increments
            XVReg y = y_regs_[j];
            XVReg acc = acc_[j];

            mov_imm(X_TMP_0, j);
            mul_d(YO2_, INCY_, X_TMP_0);
            add_d(YO2_, YO_, YO2_);

            load_bytes(y, YO2_, 0, 4);

            vfmadd_s(make_xmm(y), make_xmm(alpha_), make_xmm(acc), make_xmm(y));

            store_bytes(y, YO2_, 0, 4);
        }

        int label_idx = 0;
        for (int ux = 8; ux > 0; ux >>= 1) {
            L(label_m_remainder_loops[label_idx++]);
            if (unroll_x > ux) {
                andi(X_TMP_0, J_, ux);
                bge(zero, X_TMP_0, label_m_remainder_loops[label_idx]);

                for (int i = 0; i < unroll_y; i++) {
                    auto acc = acc_[i];
                    xvxor_v(acc, acc, acc);
                }

                innerloop(ux, unroll_y);


                // Update y.
                for (int j = 0; j < unroll_y; j++) {
                    XVReg acc = acc_[j];

                    xvpickev_w(xr30, acc, acc);
                    xvpickod_w(xr31, acc, acc);
                    xvfadd_s(acc, xr31, xr30);
                    xvpermi_q(scratch_, acc, 0x1);
                    xvfadd_s(acc, acc, scratch_);
                    xvpickev_w(xr30, acc, acc);
                    xvpickod_w(xr31, acc, acc);
                    xvfadd_s(acc, xr31, xr30);
                }
                for (int j = 0; j < unroll_y; j++) {
                    // TODO Handle negative increments
                    XVReg y = y_regs_[j];
                    XVReg acc = acc_[j];

                    mov_imm(X_TMP_0, j);
                    mul_d(YO2_, INCY_, X_TMP_0);
                    add_d(YO2_, YO_, YO2_);

                    load_bytes(y, YO2_, 0, 4);

                    vfmadd_s(make_xmm(y), make_xmm(alpha_), make_xmm(acc), make_xmm(y));

                    store_bytes(y, YO2_, 0, 4);
                }
            }
        }
        L(label_m_remainder_loops[label_idx]);

        if (unroll_y >= N_UNROLL_) {
            add_imm(I_, I_, -1 * unroll_y, X_TMP_0);
            mov_imm(X_TMP_0, unroll_y);
            bge(I_, X_TMP_0, label_n_loop);
        }
    }

}

void jit_lasx_gemv_t_f32_kern::generate() {
    // Prologue
    preamble();

    vldrepl_w(make_xmm(alpha_), ALPHA_, 0);

    ld_d(INCY_, sp, get_size_of_abi_save_regs()); // INCY_ is 9 param in sp

    add_imm(A_, A_, offset_a_ * size_, X_TMP_0);
    add_imm(X_, X_, offset_x_ * size_, X_TMP_0);

    ld_d(M_, M_, 0);
    ld_d(N_, N_, 0);
    ld_d(LDA_, LDA_, 0);
    ld_d(INCY_, INCY_, 0);

    slli_d(LDA_, LDA_, 2);
    slli_d(INCY_, INCY_, 2);

    Label outerloop_labels[4];
    Label *cur_outerloop_label = &outerloop_labels[0];

    // Main n loop.
    outerloop(M_UNROLL_, N_UNROLL_, cur_outerloop_label);

    // n remainder loops.
    for (int un = 2; un > 0; un >>= 1)
        if (N_UNROLL_ > un) outerloop(M_UNROLL_, un, cur_outerloop_label);

    L(*cur_outerloop_label);

    // Epilogue.
    postamble();
}

// Function signature: gemv(*m, *n, *alpha, *a, *lda, *x, *incx, *y, *incy)
jit_lasx_gemv_t_f32_kern::jit_lasx_gemv_t_f32_kern()
    : jit_generator(nullptr, 100000) {
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
