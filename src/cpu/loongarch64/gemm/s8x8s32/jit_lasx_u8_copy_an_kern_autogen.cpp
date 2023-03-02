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

jit_lasx_u8_copy_an_kern::jit_lasx_u8_copy_an_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_an_kern::generate() {

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
        std::vector<Xbyak_loongarch64::Label> labels(34);
        preamble();

        ld_d(M, M, 0);
        ld_d(N, N, 0);
        ld_d(LDA, LDA, 0);
        add_d(LDA3, LDA, LDA);
        add_d(LDA3, LDA3, LDA);
        addi_d(A, A, 128);
        addi_d(B, B, 128);
        mov_imm(TM, 0x10);
        blt(N, TM, labels[0]);

        L(labels[4]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x10);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[31]);

        L(labels[12]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, -0x80);
        add_d(A1, A1, LDA);
        vbsll_v(vr4, vr0, 0);
        vilvl_b(vr0, vr1, vr0);
        vilvh_b(vr4, vr1, vr4);
        vbsll_v(vr1, vr2, 0);
        vilvl_b(vr2, vr3, vr2);
        vilvh_b(vr1, vr3, vr1);
        vbsll_v(vr3, vr0, 0);
        vilvl_h(vr0, vr2, vr0);
        vilvh_h(vr3, vr2, vr3);
        vbsll_v(vr2, vr4, 0);
        vilvl_h(vr4, vr1, vr4);
        vilvh_h(vr2, vr1, vr2);
        vst(vr0, B, -0x80);
        vst(vr3, B, -0x70);
        vst(vr4, B, -0x60);
        vst(vr2, B, -0x50);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[12]);

        L(labels[31]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[32]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vbsll_v(vr2, vr0, 0);
        vilvl_b(vr0, vr1, vr0);
        vilvh_b(vr2, vr1, vr2);
        vst(vr0, B, -0x80);
        vst(vr2, B, -0x70);
        addi_d(B, B, 32);

        L(labels[32]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[33]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[33]);
        addi_d(N, N, -0x10);
        mov_imm(TM, 0x10);
        bge(N, TM, labels[4]);

        L(labels[0]);
        mov_imm(TM, 0x8);
        blt(N, TM, labels[8]);

        L(labels[1]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x8);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[3]);

        L(labels[2]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vbsll_v(vr1, vr0, 0);
        vilvl_h(vr0, vr2, vr0);
        vilvh_h(vr1, vr2, vr1);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vbsll_v(vr1, vr0, 0);
        vilvl_h(vr0, vr2, vr0);
        vilvh_h(vr1, vr2, vr1);
        vst(vr0, B, -0x60);
        vst(vr1, B, -0x50);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[2]);

        L(labels[3]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[5]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vbsll_v(vr1, vr0, 0);
        vilvl_h(vr0, vr2, vr0);
        vilvh_h(vr1, vr2, vr1);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        addi_d(B, B, 32);

        L(labels[5]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[6]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[6]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[7]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);

        L(labels[7]);
        addi_d(N, N, -0x8);
        mov_imm(TM, 0x8);
        bge(N, TM, labels[1]);

        L(labels[8]);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[16]);

        L(labels[9]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x4);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[11]);

        L(labels[10]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vst(vr0, B, -0x80);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vst(vr0, B, -0x70);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[10]);

        L(labels[11]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[13]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[13]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[14]);
        vld(vr0, A1, -0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, -0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);

        L(labels[14]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[15]);
        vld(vr0, A1, -0x80);
        vstelm_w(vr0, B, -0x80, 0);
        addi_d(B, B, 4);

        L(labels[15]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[9]);

        L(labels[16]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[23]);

        L(labels[17]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x2);
        srai_d(LDA3, M, 0x3);
        bge(zero, LDA3, labels[19]);

        L(labels[18]);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr0, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr2, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr3, TM0, 0x0);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr2, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr3, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr4, TM0, 0x0);
        vilvl_b(vr1, vr2, vr1);
        vilvl_b(vr3, vr4, vr3);
        vilvl_h(vr1, vr3, vr1);
        vilvl_d(vr0, vr1, vr0);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);
        addi_d(LDA3, LDA3, -1);
        blt(zero, LDA3, labels[18]);

        L(labels[19]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[20]);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr0, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr2, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr3, TM0, 0x0);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);

        L(labels[20]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[21]);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr0, TM0, 0x0);
        ld_h(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM0, 0x0);
        vilvl_b(vr0, vr1, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        addi_d(B, B, 4);

        L(labels[21]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[22]);
        ld_h(TM0, A1, -0x80);
        st_h(TM0, B, -0x80);
        addi_d(B, B, 2);

        L(labels[22]);
        addi_d(N, N, -0x2);
        mov_imm(TM, 0x2);
        bge(N, TM, labels[17]);

        L(labels[23]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[30]);

        L(labels[24]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x1);
        srai_d(LDA3, M, 0x3);
        bge(zero, LDA3, labels[26]);

        L(labels[25]);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x0);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x1);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x2);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x3);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x4);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x5);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x6);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x7);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);
        addi_d(LDA3, LDA3, -1);
        blt(zero, LDA3, labels[25]);

        L(labels[26]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[27]);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x0);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x1);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x2);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM0, 0x3);
        vstelm_w(vr0, B, -0x80, 0);
        addi_d(B, B, 4);

        L(labels[27]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[28]);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        st_b(TM0, B, -0x80);
        ld_b(TM0, A1, -0x80);
        add_d(A1, A1, LDA);
        st_b(TM0, B, -0x7f);
        addi_d(B, B, 2);

        L(labels[28]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[29]);
        ld_b(TM0, A1, -0x80);
        st_b(TM0, B, -0x80);
        addi_d(B, B, 1);

        L(labels[29]);
        addi_d(N, N, -0x1);
        mov_imm(TM, 0x1);
        bge(N, TM, labels[24]);

        L(labels[30]);

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
