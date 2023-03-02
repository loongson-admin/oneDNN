/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#include "cpu/loongarch64/gemm/s8x8s32/jit_lasx_gemm_s8u8s32_kern.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

using namespace Xbyak_loongarch64;

// Convert between vector register lengths.
static inline VReg make_xmm(const XVReg &v) {
    return VReg(v.getIdx());
}

void jit_lasx_gemm_s8u8s32_kern::c_load(
        const XVReg &dst, const XReg &src, int64_t offset, int nelems) {

    xvxor_v(dst, dst, dst);
    switch (nelems) {
        case 1: load_bytes(dst, src, offset, 4); break;   // 32 bit
        case 2: load_bytes(dst, src, offset, 8); break;   // 64 bit
        case 4: load_bytes(dst, src, offset, 16); break;  // 128 bit
        default:
            assert(nelems >= 8);
            load_bytes(dst, src, offset, 32);   // 256 bit
            break;
    }
}

void jit_lasx_gemm_s8u8s32_kern::c_store(
        const XVReg &dst, const XReg &src, int64_t offset, int nelems) {
    switch (nelems) {
        case 1: store_bytes(dst, src, offset, 4); break;
        case 2: store_bytes(dst, src, offset, 8); break;
        case 4: store_bytes(dst, src, offset, 16); break;
        default:
            assert(nelems >= 8);
            store_bytes(dst, src, offset, 32);
            break;
    }
}

void jit_lasx_gemm_s8u8s32_kern::dot_product(
        const XVReg &dst, const XVReg &b, const XVReg &a) {
    xvmulwev_h_bu_b(xr30, b, a);
    xvmulwod_h_bu_b(xr31, b, a);
    xvsadd_h(xr31, xr30, xr31);
    xvhaddw_w_h(xr31, xr31, xr31);
    xvadd_w(dst, dst, xr31);
}

void jit_lasx_gemm_s8u8s32_kern::kernel_loop(
        int unroll_m, int unroll_n, bool cfetch) {
    int um_vecs = (unroll_m + 7) >> 3;
    Label label_kernel_loop;

    L_aligned(label_kernel_loop);
    {
        for (int h = 0; h < 4; h++) {
            for (int j = 0; j < max_unroll_n_; j++) {
                const XVReg b = b_regs_;

                if (unroll_n > j) {
                    uni_xvldrepl_w(b, BO_, 4 * j + 4 * h * unroll_n - offset_b_);

                    for (int i = 0; i < um_vecs; i++)
                        dot_product(c_regs_[i][j], b, a_regs_[i]);
                }

                if (h == 0 && j == 0){
                    uni_preld(0, AO_, prefetch_size_a_ - offset_a_);
                }
                else if (h == 0 && j == 1){
                    uni_preld(0, BO_, prefetch_size_b_ - offset_b_);
                }
                else if (h == 0 && j == 2 && um_vecs >= 2){
                    uni_preld(0, AO_, prefetch_size_a_ + 64 - offset_a_);
                }
                else if (h == 1 && j == 1 && um_vecs >= 3){
                    uni_preld(0, AO_, prefetch_size_a_ + 128 - offset_a_);
                }
                else if (h == 2 && j == 0){
                    uni_preld(0, AO_, prefetch_size_a_ + 192 - offset_a_);
                }
                else if (h == 2 && j == 1 && cfetch){
                    uni_preld(8, CO2_, 0);
                }
                else if (h == 2 && j == 2 && um_vecs >= 2){
                    uni_preld(0, AO_, prefetch_size_a_ + 256 - offset_a_);
                }
                else if (h == 2 && j == 3 && um_vecs >= 2 && cfetch){
                    uni_preld(8, CO2_, 16 * size_);
                }
                else if (h == 3 && j == 1 && um_vecs >= 3){
                    uni_preld(0, AO_, prefetch_size_a_ + 320 - offset_a_);
                }
                else if (h == 3 && j == 2){
                    addi_d(AA_, AA_, 8);
                }
                else if (h == 3 && j == 3 && cfetch){
                    add_d(CO2_, CO2_, LDC_);
                }
            }

            for (int i = 0; i < um_vecs; i++){
                uni_xvld(a_regs_[i], AO_, 32 * i + (h + 1) * 4 * unroll_m - offset_a_);
            }

            if (h == 2){
                uni_preld(0, AA_, 0);
            }
        }

        add_imm(AO_, AO_, unroll_m * 16, X_TMP_0);
        add_imm(BO_, BO_, unroll_n * 16, X_TMP_0);
        addi_d(LoopCount_, LoopCount_, -1);
        blt(zero, LoopCount_, label_kernel_loop);
    }
}

// k remainder loop for kernel.
void jit_lasx_gemm_s8u8s32_kern::remainder_kernel(
        int unroll_m, int unroll_n, int unroll_k, int bwidth) {
    XVReg b = b_regs_;

    int um_vecs = (unroll_m + 7) >> 3;
    for (int h = 0; h < unroll_k; h++) {
        for (int j = 0; j < unroll_n; j++) {
            switch (bwidth) {
                case 4: uni_xvldrepl_w(b, BO_, j * bwidth + 4 * h * unroll_n - offset_b_); break;
                case 2: uni_xvldrepl_h(b, BO_, j * bwidth + 4 * h * unroll_n - offset_b_); break;
                case 1: uni_xvldrepl_b(b, BO_, j * bwidth + 4 * h * unroll_n - offset_b_); break;
            }
            for (int i = 0; i < um_vecs; i++)
                dot_product(c_regs_[i][j], b, a_regs_[i]);
        }

        if (unroll_k > 1)
            for (int i = 0; i < um_vecs; i++){
                uni_xvld(a_regs_[i], AO_, 32 * i + (h + 1) * 4 * unroll_m - offset_a_);
            }
    }

    add_imm(AO_, AO_, unroll_m * unroll_k * bwidth, X_TMP_0);
    add_imm(BO_, BO_, unroll_n * unroll_k * bwidth, X_TMP_0);
}

void jit_lasx_gemm_s8u8s32_kern::innerloop(int unroll_m, int unroll_n) {
    int um_vecs = (unroll_m + 7) >> 3;
    int stage1 = unroll_n, stage2 = 16;

    Label label_k_main_loop_2;
    Label label_k_main_loop_3;
    Label label_k_remainder_loop_begin;
    Label label_k_rem_3, label_k_rem_2, label_k_rem_1, label_k_rem_0;

    add_d(AO_, A_, zero);
    for (int i = 0; i < um_vecs; i++){
        uni_xvld(a_regs_[i], AO_, 32 * i - offset_a_);
    }

    srai_d(LoopCount_, K_, 4);
    bge(zero, LoopCount_, label_k_remainder_loop_begin);

    // Main k loops, broken into three parts to time C prefetching.
    add_imm(LoopCount_, LoopCount_, - stage1 - stage2, X_TMP_0);
    bge(zero, LoopCount_, label_k_main_loop_2);

    kernel_loop(unroll_m, unroll_n, false);

    L_aligned(label_k_main_loop_2);
    add_imm(CO2_, CO1_, size_ * 7, X_TMP_0);
    addi_d(LoopCount_, LoopCount_, stage1);
    bge(zero, LoopCount_, label_k_main_loop_3);

    kernel_loop(unroll_m, unroll_n, true);

    L_aligned(label_k_main_loop_3);
    addi_d(LoopCount_, LoopCount_, stage2);
    bge(zero, LoopCount_, label_k_remainder_loop_begin);

    kernel_loop(unroll_m, unroll_n, false);

    // k remainder handling
    L_aligned(label_k_remainder_loop_begin);
    add_d(LoopCount_, K_, zero);
    andi(X_TMP_0, LoopCount_, 8);
    beq(X_TMP_0, zero, label_k_rem_3);

    remainder_kernel(unroll_m, unroll_n, 2, 4);

    L_aligned(label_k_rem_3);
    add_d(LoopCount_, K_, zero);
    andi(X_TMP_0, LoopCount_, 4);
    beq(X_TMP_0, zero, label_k_rem_2);

    remainder_kernel(unroll_m, unroll_n, 1, 4);

    L_aligned(label_k_rem_2);
    add_d(LoopCount_, K_, zero);
    andi(X_TMP_0, LoopCount_, 2);
    beq(X_TMP_0, zero, label_k_rem_1);

    for (int i = 0; i < um_vecs; i++) {
        XVReg a = a_regs_[i];
        uni_xvld(make_xmm(a), AO_, 16 * i - offset_a_);
        xvreplve0_q(a, a);
        uni_xvld(xr31, sp, 0);
        xvshuf_b(a, a, a, xr31);
        // delete some data
        xvxor_v(xr30, xr30, xr30);
        xvpackev_h(a, xr30, a);
    }

    remainder_kernel(unroll_m, unroll_n, 1, 2);

    L_aligned(label_k_rem_1);
    add_d(LoopCount_, K_, zero);
    andi(X_TMP_0, LoopCount_, 1);
    beq(X_TMP_0, zero, label_k_rem_0);

    for (int i = 0; i < um_vecs; i++) {
        XVReg a = a_regs_[i];
        uni_xvldrepl_d(a, AO_, 8 * i - offset_a_);
        uni_xvld(xr31, sp, 32);
        xvshuf_b(a, a, a, xr31);
        //delete some data
        xvxor_v(xr30, xr30, xr30);
        xvpackev_h(a, xr30, a);
        xvpackev_b(a, xr30, a);
    }

    remainder_kernel(unroll_m, unroll_n, 1, 1);

    L_aligned(label_k_rem_0);

    // Add offsets and update C.
    if (enable_offset_r_) {
        // Add row offsets.
        ld_d(rax, coffset_ry_.getXReg(), coffset_ry_.getOffset());
        for (int j = 0; j < unroll_n; j++) {
            XVReg row_offset = xr0;

            uni_xvldrepl_w(row_offset, rax, size_ * j);

            for (int i = 0; i < um_vecs; i++){
                xvadd_w(c_regs_[i][j], c_regs_[i][j], row_offset);
            }
        }
        add_imm(rax, rax, size_ * unroll_n, X_TMP_0);
        st_d(rax, coffset_ry_.getXReg(), coffset_ry_.getOffset());
    }

    if (enable_offset_c_) {
        // Add column offsets.
        ld_d(rax, coffset_cy_.getXReg(), coffset_cy_.getOffset());
        for (int i = 0; i < um_vecs; i++) {
            XVReg col_offset = xr0;

            c_load(col_offset, rax, size_ * 8 * i, unroll_m);

            for (int j = 0; j < unroll_n; j++){
                xvadd_w(c_regs_[i][j], c_regs_[i][j], col_offset);
            }
        }
    }

    XReg LDC3 = rax;
    add_d(LDC3, LDC_, LDC_);
    add_d(LDC3, LDC3, LDC_);

    // C updates.
    int c_off_j = 0;
    for (int j = 0; j < unroll_n; j++) {
        if (j > 0 && (j & 3) == 0) {
            add_d(CO1_, CO1_, LDC3);
            add_d(CO1_, CO1_, LDC_);
            c_off_j += 4;
        }

        int jj = j - c_off_j;

        for (int i = 0; i < um_vecs; i++) {
            XVReg c = c_regs_[i][j];
            XVReg c_old = xr0;

            mov_imm(X_TMP_1, jj);
            mul_d(X_TMP_1, LDC_, X_TMP_1);
            add_d(X_TMP_1, X_TMP_1, CO1_);

            if (beta_zero_) {
                c_store(c, X_TMP_1, size_ * 8 * i, unroll_m);
            } else {
                c_load(c_old, X_TMP_1, size_ * 8 * i, unroll_m);
                xvadd_w(c_old, c, c_old);
                c_store(c_old, X_TMP_1, size_ * 8 * i, unroll_m);
            }

            xvxor_v(c, c, c);
        }
    }

    mov_imm(X_TMP_0, unroll_n - c_off_j);
    mul_d(X_TMP_0, LDC_, X_TMP_0);
    add_d(CO1_, CO1_, X_TMP_0);
}

// Outer loop.
void jit_lasx_gemm_s8u8s32_kern::outerloop(
        int unroll_x, int unroll_y, Label *&cur_outerloop_label) {
    Label label_m_loop, label_n_loop;
    std::vector<Label> label_n_remainder_loops(6);

    L(*cur_outerloop_label);
    cur_outerloop_label++;
    if (unroll_x >= unroll_m_) {
        add_d(J_, M_, zero);
        mov_imm(X_TMP_0, unroll_x);
        blt(J_, X_TMP_0, *cur_outerloop_label);
    } else {
        andi(X_TMP_0, J_, unroll_x);
        bge(zero, X_TMP_0, *cur_outerloop_label);
    }

    L_aligned(label_m_loop);
    {
        add_d(CO1_, C_, zero);
        add_imm(C_, C_, unroll_x * size_, X_TMP_0);

        add_d(BO_, B_, zero);

        mov_imm(X_TMP_1, unroll_x);
        mul_d(AA_, K_, X_TMP_1);
        mov_imm(X_TMP_1, isize_);
        mul_d(AA_, AA_, X_TMP_1);
        add_d(AA_, A_, AA_);
        add_imm(AA_, AA_, prefetch_size_a_ - offset_a_, X_TMP_0);

        if (enable_offset_c_) {
            ld_d(rax, coffset_cx_.getXReg(), coffset_cx_.getOffset());
            st_d(rax, coffset_cy_.getXReg(), coffset_cy_.getOffset());
            add_imm(rax, rax, unroll_x * size_, X_TMP_0);
            st_d(rax, coffset_cx_.getXReg(), coffset_cx_.getOffset());
        }

        if (enable_offset_r_) {
            ld_d(rax, coffset_rx_.getXReg(), coffset_rx_.getOffset());
            st_d(rax, coffset_ry_.getXReg(), coffset_ry_.getOffset());
        }

        add_d(I_, N_, zero);
        mov_imm(X_TMP_0, unroll_y);
        blt(I_, X_TMP_0, label_n_remainder_loops[0]);

        L_aligned(label_n_loop);
        {
            innerloop(unroll_x, unroll_y);
            add_imm(I_, I_, - unroll_y, X_TMP_0);
            mov_imm(X_TMP_0, unroll_y);
            bge(I_, X_TMP_0, label_n_loop);
        }


        int label_idx = 0;
        for (int uy = 2; uy > 0; uy >>= 1) {
            L(label_n_remainder_loops[label_idx++]);
            if (unroll_y > uy) {
                andi(X_TMP_0, I_, uy);
                bge(zero, X_TMP_0, label_n_remainder_loops[label_idx]);

                innerloop(unroll_x, uy);
            }
        }
        L(label_n_remainder_loops[label_idx]);

        add_d(A_, AO_, zero);
        if (unroll_x >= unroll_m_) {
            add_imm(J_, J_, - unroll_x, X_TMP_0);
            mov_imm(X_TMP_0, unroll_x);
            bge(J_, X_TMP_0, label_m_loop);
        }
    }

}

void jit_lasx_gemm_s8u8s32_kern::generate() {
    // Prologue
    preamble();
    addi_d(sp, sp, - stack_alloc_size_);

    addi_d(A_, A_, offset_a_);
    addi_d(B_, B_, offset_b_);

    ld_d(M_, M_, 0);
    ld_d(N_, N_, 0);
    ld_d(K_, K_, 0);

    mov_imm(X_TMP_0, size_);
    mul_d(LDC_, LDC_, X_TMP_0);

    if (enable_offset_c_) {
        ld_d(rax, arg_coffset_c_.getXReg(), arg_coffset_c_.getOffset());
        st_d(rax, coffset_cx_.getXReg(), coffset_cx_.getOffset());
    }

    if (enable_offset_r_) {
        ld_d(rax, arg_coffset_r_.getXReg(), arg_coffset_r_.getOffset());
        st_d(rax, coffset_rx_.getXReg(), coffset_rx_.getOffset());
    }

    // bcast_k2_
    for (int i = 0; i < 8; i++){
        mov_imm(X_TMP_0, 0x80800100 + 0x202 * i);
        st_w(X_TMP_0, sp, 4 * i);
    }

    // bcast_k1_
    for (int i = 0; i < 8; i++){
        mov_imm(X_TMP_0, 0x80808000 + i);
        st_w(X_TMP_0, sp, 32 + 4 * i);
    }

    for (int i = 0; i < (unroll_m_ >> 3); i++) {
        for (int j = 0; j < max_unroll_n_; j++) {
            auto &c = c_regs_[i][j];
            xvxor_v(c, c, c);
        }
    }

    std::vector<Label> outerloop_labels(8);
    Label *cur_outerloop_label = &outerloop_labels[0];

    // Main m loop.
    outerloop(unroll_m_, IGEMM_UNROLL_N_, cur_outerloop_label);

    // m remainder loops.
    for (int um = 16; um > 0; um >>= 1)
        if (unroll_m_ > um) outerloop(um, IGEMM_UNROLL_N_, cur_outerloop_label);

    L(*cur_outerloop_label);

    // Epilogue.
    addi_d(sp, sp, stack_alloc_size_);
    postamble();
}

jit_lasx_gemm_s8u8s32_kern::jit_lasx_gemm_s8u8s32_kern(bool beta_zero,
        bool enable_offset_c, bool enable_offset_r, int unroll_m)
    : jit_generator(nullptr, 100000) {

    beta_zero_ = beta_zero;
    enable_offset_c_ = enable_offset_c;
    enable_offset_r_ = enable_offset_r;
    vnni_ = false;
    unroll_m_ = unroll_m;

    assert(utils::one_of(unroll_m, 24, 16, 8, 4, 2, 1));

    // Assign integer registers

    // Assign vector registers

    for (int i = 0; i < (unroll_m_ >> 3); i++)
            a_regs_[i] = XVReg(i);


    int um_vecs = unroll_m_ >> 3;
    int rn = 0;
    for (int i = 0; i < nstl::min(um_vecs, 2); i++)
        for (int j = 0; j < max_unroll_n_; j++){
            c_regs_[i][j] = XVReg(8 + rn++);
        }
    for (int j = 0; j < max_unroll_n_; j++){
        c_regs_[2][j] = XVReg(4 + j);
    }

    // Assign stack variables.
    // auto args_offset = stack_alloc_size_ + get_size_of_abi_save_regs() + 8

}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
