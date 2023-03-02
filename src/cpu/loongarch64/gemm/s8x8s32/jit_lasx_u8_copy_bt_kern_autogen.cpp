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

jit_lasx_u8_copy_bt_kern::jit_lasx_u8_copy_bt_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_bt_kern::generate() {

#define M a0    //rdi
#define N a1    //rsi
#define A a2    //rdx
#define LDA a3  //rcx
#define ALPHA a4    //r8
#define B a5    //r9

#define I t4    //rax
#define A1 t5   //r10
#define A2 a4   //r8
#define LDA3 t6 //r11
#define TM t7

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch64::Label> labels(21);

        preamble();

        ld_d(M, M, 0);
        ld_d(N, N, 0);
        ld_d(LDA, LDA, 0);
        add_d(LDA3, LDA, LDA);
        add_d(LDA3, LDA3, LDA);
        addi_d(A, A, 128);
        addi_d(B, B, 128);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[2]);

        L(labels[6]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x4);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[19]);

        L(labels[13]);
        vld(vr0, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, - 0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vst(vr0, B, - 0x80);
        vld(vr0, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, - 0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vst(vr0, B, - 0x70);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[13]);

        L(labels[19]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[20]);
        vld(vr0, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr2, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr3, A1, - 0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vst(vr0, B, - 0x80);
        addi_d(B, B, 16);

        L(labels[20]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[0]);
        vld(vr0, A1, - 0x80);
        add_d(A1, A1, LDA);
        vld(vr1, A1, - 0x80);
        add_d(A1, A1, LDA);
        vilvl_b(vr0, vr1, vr0);
        vstelm_d(vr0, B, - 0x80, 0);
        addi_d(B, B, 8);

        L(labels[0]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[1]);
        ld_d(TM, A1, - 0x80);
        st_d(TM, B, - 0x80);
        addi_d(B, B, 4);

        L(labels[1]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[6]);

        L(labels[2]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[10]);

        L(labels[3]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x2);
        srai_d(LDA3, M, 0x3);
        bge(zero, LDA3, labels[5]);

        L(labels[4]);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr0, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr2, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr3, TM, 0x0);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr2, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr3, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr4, TM, 0x0);
        vilvl_b(vr1, vr2, vr1);
        vilvl_b(vr3, vr4, vr3);
        vilvl_h(vr1, vr3, vr1);
        vilvl_d(vr0, vr1, vr0);
        vst(vr0, B, - 0x80);
        addi_d(B, B, 16);
        addi_d(LDA3, LDA3, -1);
        blt(zero, LDA3, labels[4]);

        L(labels[5]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[7]);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr0, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr2, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr3, TM, 0x0);
        vilvl_b(vr0, vr1, vr0);
        vilvl_b(vr2, vr3, vr2);
        vilvl_h(vr0, vr2, vr0);
        vstelm_d(vr0, B, - 0x80, 0);
        addi_d(B, B, 8);

        L(labels[7]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[8]);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr0, TM, 0x0);
        ld_h(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_h(vr1, TM, 0x0);
        vilvl_b(vr0, vr1, vr0);
        vstelm_w(vr0, B, - 0x80, 0);
        addi_d(B, B, 4);

        L(labels[8]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[9]);
        ld_h(TM, A1, - 0x80);
        st_h(TM, B, - 0x80);
        addi_d(B, B, 2);

        L(labels[9]);
        addi_d(N, N, -0x2);
        mov_imm(TM, 0x2);
        bge(N, TM, labels[3]);

        L(labels[10]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[18]);

        L(labels[11]);
        add_d(A1, A, zero);
        addi_d(A, A, 0x1);
        srai_d(LDA3, M, 0x3);
        bge(zero, LDA3, labels[14]);

        L(labels[12]);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x0);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x1);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x2);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x3);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x4);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x5);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x6);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x7);
        vstelm_d(vr0, B, - 0x80, 0);
        addi_d(B, B, 8);
        addi_d(LDA3, LDA3, -1);
        blt(zero, LDA3, labels[12]);

        L(labels[14]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[15]);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x0);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x1);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x2);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        vinsgr2vr_b(vr0, TM, 0x3);
        vstelm_w(vr0, B, - 0x80, 0);
        addi_d(B, B, 4);

        L(labels[15]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[16]);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        st_b(TM, B, - 0x80);
        ld_b(TM, A1, - 0x80);
        add_d(A1, A1, LDA);
        st_b(TM, B, - 0x7f);
        addi_d(B, B, 2);

        L(labels[16]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[17]);
        ld_b(TM, A1, - 0x80);
        st_b(TM, B, - 0x80);
        addi_d(B, B, 1);

        L(labels[17]);
        addi_d(N, N, -0x1);
        mov_imm(TM, 0x1);
        bge(N, TM, labels[11]);

        L(labels[18]);

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
#undef TM
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
