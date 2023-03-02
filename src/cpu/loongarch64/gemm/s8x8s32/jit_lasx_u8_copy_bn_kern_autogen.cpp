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

#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

jit_lasx_u8_copy_bn_kern::jit_lasx_u8_copy_bn_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_bn_kern::generate() {

#define M a0//rdi
#define N a1//rsi
#define A a2//rdx
#define LDA a3//rcx
#define ALPHA a4//r8
#define B a5//r9

#define I t4//rax
#define A1 t5//r10
#define A2 a4//r8
#define LDA3 t6//r11
#define TM t7
#define TM0 t8

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch64::Label> labels(24);

        preamble();

        ld_d(N, N, 0);
        ld_d(M, M, 0);
        ld_d(LDA, LDA, 0);
        addi_d(A, A, 128);
        addi_d(B, B, 128);
        add_d(LDA3, LDA, LDA);
        add_d(LDA3, LDA3, LDA);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[3]);

        L(labels[6]);
        add_d(A1, A, zero);
        add_d(TM, A1, LDA);
        add_d(A2, TM, LDA);
        add_d(I, TM, LDA3);
        add_d(A, I, zero);
        srai_d(I, M, 0x4);
        bge(zero, I, labels[22]);

        L(labels[19]);
        vld(vr0, A1, -0x80);
        add_d(TM, A1, LDA);
        vld(vr1, TM, -0x80);
        addi_d(A1, A1, 16);
        vld(vr2, A2, -0x80);
        add_d(TM, A2, LDA);
        vld(vr3, TM, -0x80);
        addi_d(A2, A2, 16);
        vbsll_v(vr4, vr0, 0);
        vilvl_w(vr0, vr1, vr0);
        vilvh_w(vr4, vr1, vr4);
        vbsll_v(vr5, vr2, 0);
        vilvl_w(vr2, vr3, vr2);
        vilvh_w(vr5, vr3, vr5);
        vbsll_v(vr1, vr0, 0);
        vilvl_d(vr0, vr2, vr0);
        vilvh_d(vr1, vr2, vr1);
        vbsll_v(vr3, vr4, 0);
        vilvl_d(vr4, vr5, vr4);
        vilvh_d(vr3, vr5, vr3);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        vst(vr4, B, -0x60);
        vst(vr3, B, -0x50);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[19]);

        L(labels[22]);
        andi(TM, M, 0x8);
        bge(zero, TM, labels[23]);
        vld(vr0, A1, -0x80);
        add_d(TM, A1, LDA);
        vld(vr1, TM, -0x80);
        addi_d(A1, A1, 8);
        vld(vr2, A2, -0x80);
        add_d(TM, A2, LDA);
        vld(vr3, TM, -0x80);
        addi_d(A2, A2, 8);
        vilvl_w(vr0, vr1, vr0);
        vilvl_w(vr2, vr3, vr2);
        vbsll_v(vr1, vr0, 0);
        vilvl_d(vr0, vr2, vr0);
        vilvh_d(vr1, vr2, vr1);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        addi_d(B, B, 32);

        L(labels[23]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[0]);
        vld(vr0, A1, -0x80);
        add_d(TM, A1, LDA);
        vld(vr1, TM, -0x80);
        addi_d(A1, A1, 4);
        vld(vr2, A2, -0x80);
        add_d(TM, A2, LDA);
        vld(vr3, TM, -0x80);
        addi_d(A2, A2, 4);
        vilvl_w(vr0, vr1, vr0);
        vilvl_w(vr2, vr3, vr2);
        vilvl_d(vr0, vr2, vr0);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[0]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[1]);
        ld_h(TM0, A1, -0x80);
        vinsgr2vr_h(vr0, TM0, 0x0);
        add_d(TM, A1, LDA);
        ld_h(TM0, TM, -0x80);
        addi_d(A1, A1, 2);
        vinsgr2vr_h(vr0, TM0, 0x1);
        ld_h(TM0, A2, -0x80);
        vinsgr2vr_h(vr0, TM0, 0x2);
        add_d(TM, A2, LDA);
        ld_h(TM0, TM, -0x80);
        addi_d(A2, A2, 2);
        vinsgr2vr_h(vr0, TM0, 0x3);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);

        L(labels[1]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[2]);
        ld_b(TM0, A1, -0x80);
        vinsgr2vr_b(vr0, TM0, 0x0);
        add_d(TM, A1, LDA);
        ld_b(TM0, TM, -0x80);
        vinsgr2vr_b(vr0, TM0, 0x1);
        ld_b(TM0, A2, -0x80);
        vinsgr2vr_b(vr0, TM0, 0x2);
        add_d(TM, A2, LDA);
        ld_b(TM0, TM, -0x80);
        vinsgr2vr_b(vr0, TM0, 0x3);
        vstelm_w(vr0, B, -0x80, 0);
        addi_d(B, B, 4);

        L(labels[2]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[6]);

        L(labels[3]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[12]);

        L(labels[4]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA);
        add_d(I, A2, LDA);
        add_d(A, I, zero);
        srai_d(I, M, 0x4);
        bge(zero, I, labels[7]);

        L(labels[5]);
        vld(vr0, A1, -0x80);
        addi_d(A1, A1, 16);
        vld(vr1, A2, -0x80);
        addi_d(A2, A2, 16);
        vbsll_v(vr2, vr0, 0);
        vilvl_w(vr0, vr1, vr0);
        vilvh_w(vr2, vr1, vr2);
        vst(vr0, B, -0x80);
        vst(vr2, B, -0x70);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[5]);

        L(labels[7]);
        andi(TM, M, 0x8);
        bge(zero, TM, labels[8]);
        vld(vr0, A1, -0x80);
        addi_d(A1, A1, 8);
        vld(vr1, A2, -0x80);
        addi_d(A2, A2, 8);
        vilvl_w(vr0, vr1, vr0);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[8]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[9]);
        vld(vr0, A1, -0x80);
        addi_d(A1, A1, 4);
        vld(vr1, A2, -0x80);
        addi_d(A2, A2, 4);
        vilvl_w(vr0, vr1, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);

        L(labels[9]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[10]);
        ld_h(TM0, A1, -0x80);
        addi_d(A1, A1, 2);
        vinsgr2vr_h(vr0, TM0, 0x0);
        ld_h(TM0, A2, -0x80);
        addi_d(A2, A2, 2);
        vinsgr2vr_h(vr0, TM0, 0x1);
        vstelm_w(vr0, B, -0x80, 0);
        addi_d(B, B, 4);

        L(labels[10]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[11]);
        ld_b(TM0, A1, -0x80);
        st_b(TM0, B, -0x80);
        ld_b(TM0, A2, -0x80);
        st_b(TM0, B, -0x7f);
        addi_d(B, B, 2);

        L(labels[11]);
        addi_d(N, N, -0x2);
        mov_imm(TM, 0x2);
        bge(N, TM, labels[4]);

        L(labels[12]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[21]);

        L(labels[13]);
        add_d(A1, A, zero);
        add_d(A, A, LDA);
        srai_d(I, M, 0x4);
        bge(zero, I, labels[15]);

        L(labels[14]);
        vld(vr0, A1, -0x80);
        addi_d(A1, A1, 16);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);
        addi_d(I, I, -1);
        blt(zero, I, labels[14]);

        L(labels[15]);
        andi(TM, M, 0x8);
        bge(zero, TM, labels[16]);
        ld_d(TM, A1, -0x80);
        addi_d(A1, A1, 8);
        st_d(TM, B, -0x80);
        addi_d(B, B, 8);

        L(labels[16]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[17]);
        ld_w(TM, A1, -0x80);
        addi_d(A1, A1, 4);
        st_w(TM, B, -0x80);
        addi_d(B, B, 4);

        L(labels[17]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[18]);
        ld_h(TM, A1, -0x80);
        st_h(TM, B, -0x80);
        addi_d(A1, A1, 2);
        addi_d(B, B, 2);

        L(labels[18]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[20]);
        ld_b(TM, A1, -0x80);
        st_b(TM, B, -0x80);
        addi_d(B, B, 1);

        L(labels[20]);
        addi_d(N, N, -0x1);
        mov_imm(TM, 0x1);
        bge(N, TM, labels[13]);

        L(labels[21]);

        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef A
#undef LDA
#undef ALPHA
#undef B
#undef I
#undef A1
#undef A2
#undef LDA3
}

} //namespace loongarch64
} //namespace cpu
} //namespace impl
} //namespace dnnl
