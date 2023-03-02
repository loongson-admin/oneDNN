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

#include "cpu/loongarch64/gemm/f32/common_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

jit_lasx_f32_copy_bn_kern::jit_lasx_f32_copy_bn_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_f32_copy_bn_kern::generate() {

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
#define TM s1
#define TM0 s0
#define LDA2 t7
#define LDA4 t8

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch64::Label> labels(50);

        preamble();

        ld_d(M, M, 0);
        ld_d(N, N, 0);
        ld_d(LDA, LDA, 0);
        addi_d(B, B, 128);
        slli_d(LDA, LDA, 0x2);
        add_d(LDA2, LDA, LDA);
        add_d(LDA3, LDA2, LDA);
        add_d(LDA4, LDA3, LDA);
        xvldrepl_w(xr6, ALPHA, 0);
        vseq_b(vr3, vr3, vr3);
        vsrli_w(vr3, vr3, 0x17);
        vslli_w(vr3, vr3, 0x19);
        vsrli_w(vr3, vr3, 0x2);
        vseq_b(vr4, vr4, vr4);
        vslli_w(vr4, vr4, 0x1f);
        xvpermi_q(xr4, xr4, 0x20);
        vfcmp_cle_s(vr31, vr3, vr6);
        vpickve2gr_w(TM, vr31, 0);
        bnez(TM, labels[36]);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[47]);

        L(labels[23]);
        add_d(A1, A, zero);
        add_d(A, A, LDA4);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[0]);

        L(labels[14]);
        vld(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vldx(vr2, A1, LDA2);
        vldx(vr3, A1, LDA3);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr5, vr1, vr0);
        vilvl_w(vr1, vr3, vr2);
        vilvh_w(vr3, vr3, vr2);
        vilvl_d(vr0, vr1, vr4);
        vilvh_d(vr1, vr1, vr4);
        vilvl_d(vr2, vr3, vr5);
        vilvh_d(vr3, vr3, vr5);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        vst(vr2, B, -0x60);
        vst(vr3, B, -0x50);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 16);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[14]);

        L(labels[0]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[49]);
        vldrepl_d(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        ldx_d(TM0, A1, LDA2);
        vinsgr2vr_d(vr0, TM0, 1);
        ldx_d(TM0, A1, LDA3);
        vinsgr2vr_d(vr1, TM0, 1);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr1, vr1, vr0);
        vilvl_d(vr0, vr1, vr4);
        vilvh_d(vr1, vr1, vr4);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 8);
        addi_d(B, B, 32);

        L(labels[49]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[48]);
        vldrepl_w(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vldx(vr2, A1, LDA2);
        vldx(vr3, A1, LDA3);
        vilvl_w(vr2, vr3, vr2);
        vilvl_d(vr0, vr2, vr0);
        vst(vr0, B, -0x80);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 4);
        addi_d(B, B, 16);

        L(labels[48]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[23]);

        L(labels[47]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[42]);
        add_d(A1, A, zero);
        add_d(A, A, LDA2);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[45]);

        L(labels[46]);
        vld(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr1, vr1, vr0);
        vbsll_v(vr0, vr4, 0);
        vstelm_d(vr0, B, -0x80, 0);
        vstelm_d(vr0, B, -0x78, 1);
        vstelm_d(vr1, B, -0x70, 0);
        vstelm_d(vr1, B, -0x68, 1);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 16);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[46]);

        L(labels[45]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[44]);
        vldrepl_d(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        vstelm_d(vr0, B, -0x78, 1);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 8);
        addi_d(B, B, 16);

        L(labels[44]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[43]);
        vldrepl_w(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 4);
        addi_d(B, B, 8);

        L(labels[43]);
        addi_d(N, N, -0x2);

        L(labels[42]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[37]);
        add_d(A1, A, zero);
        add_d(A, A, LDA);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[40]);

        L(labels[41]);
        vld(vr0, A1, 0);
        vstelm_w(vr0, B, -0x80, 0);
        vstelm_w(vr0, B, -0x7c, 1);
        vstelm_w(vr0, B, -0x78, 2);
        vstelm_w(vr0, B, -0x74, 3);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 16);
        addi_d(B, B, 16);
        addi_d(I, I, -1);
        blt(zero, I, labels[41]);

        L(labels[40]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[39]);
        vldrepl_d(vr0, A1, 0);
        vstelm_w(vr0, B, -0x80, 0);
        vstelm_w(vr0, B, -0x7c, 1);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 8);
        addi_d(B, B, 8);

        L(labels[39]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[38]);
        vldrepl_w(vr0, A1, 0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 4);
        addi_d(B, B, 4);

        L(labels[38]);
        addi_d(N, N, -0x1);

        L(labels[37]);
        b(labels[1]);

        L(labels[36]);
        vxor_v(vr3, vr3, vr4);
        vfcmp_cle_s(vr31, vr3, vr6);
        vpickve2gr_w(TM, vr31, 0);
        bnez(TM, labels[18]);
        xvbsll_v(xr6, xr4, 0);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[30]);

        L(labels[35]);
        add_d(A1, A, zero);
        add_d(A, A, LDA4);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[33]);

        L(labels[34]);
        vld(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vldx(vr2, A1, LDA2);
        vldx(vr3, A1, LDA3);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr5, vr1, vr0);
        vilvl_w(vr1, vr3, vr2);
        vilvh_w(vr3, vr3, vr2);
        vilvl_d(vr0, vr1, vr4);
        vilvh_d(vr1, vr1, vr4);
        vilvl_d(vr2, vr3, vr5);
        vilvh_d(vr3, vr3, vr5);
        vxor_v(vr0, vr6, vr0);
        vxor_v(vr1, vr6, vr1);
        vxor_v(vr2, vr6, vr2);
        vxor_v(vr3, vr6, vr3);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        vst(vr2, B, -0x60);
        vst(vr3, B, -0x50);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 16);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[34]);

        L(labels[33]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[32]);
        vldrepl_d(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        ldx_d(TM0, A1, LDA2);
        vinsgr2vr_d(vr0, TM0, 1);
        ldx_d(TM0, A1, LDA3);
        vinsgr2vr_d(vr1, TM0, 1);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr1, vr1, vr0);
        vilvl_d(vr0, vr1, vr4);
        vilvh_d(vr1, vr1, vr4);
        vxor_v(vr0, vr6, vr0);
        vxor_v(vr1, vr6, vr1);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 8);
        addi_d(B, B, 32);

        L(labels[32]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[31]);
        vldrepl_w(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vldx(vr2, A1, LDA2);
        vldx(vr3, A1, LDA3);
        vilvl_w(vr2, vr3, vr2);
        vilvl_d(vr0, vr2, vr0);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 4);
        addi_d(B, B, 16);

        L(labels[31]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[35]);

        L(labels[30]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[25]);
        add_d(A1, A, zero);
        add_d(A, A, LDA2);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[28]);

        L(labels[29]);
        vld(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr1, vr1, vr0);
        vbsll_v(vr0, vr4, 0);
        vxor_v(vr0, vr6, vr0);
        vxor_v(vr1, vr6, vr1);
        vstelm_d(vr0, B, -0x80, 0);
        vstelm_d(vr0, B, -0x78, 1);
        vstelm_d(vr1, B, -0x70, 0);
        vstelm_d(vr1, B, -0x68, 1);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 16);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[29]);

        L(labels[28]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[27]);
        vldrepl_d(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        vstelm_d(vr0, B, -0x78, 1);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 8);
        addi_d(B, B, 16);

        L(labels[27]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[26]);
        vldrepl_w(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 4);
        addi_d(B, B, 8);

        L(labels[26]);
        addi_d(N, N, -0x2);

        L(labels[25]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[19]);
        add_d(A1, A, zero);
        add_d(A, A, LDA);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[22]);

        L(labels[24]);
        vld(vr0, A1, 0);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        vstelm_w(vr0, B, -0x7c, 1);
        vstelm_w(vr0, B, -0x78, 2);
        vstelm_w(vr0, B, -0x74, 3);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 16);
        addi_d(B, B, 16);
        addi_d(I, I, -1);
        blt(zero, I, labels[24]);

        L(labels[22]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[21]);
        vldrepl_d(vr0, A1, 0);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        vstelm_w(vr0, B, -0x7c, 1);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 8);
        addi_d(B, B, 8);

        L(labels[21]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[20]);
        vldrepl_w(vr0, A1, 0);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 4);
        addi_d(B, B, 4);

        L(labels[20]);
        addi_d(N, N, -0x1);

        L(labels[19]);
        b(labels[1]);

        L(labels[18]);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[11]);

        L(labels[17]);
        add_d(A1, A, zero);
        add_d(A, A, LDA4);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[15]);

        L(labels[16]);
        vld(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vldx(vr2, A1, LDA2);
        vldx(vr3, A1, LDA3);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr5, vr1, vr0);
        vilvl_w(vr1, vr3, vr2);
        vilvh_w(vr3, vr3, vr2);
        vilvl_d(vr0, vr1, vr4);
        vilvh_d(vr1, vr1, vr4);
        vilvl_d(vr2, vr3, vr5);
        vilvh_d(vr3, vr3, vr5);
        vfmul_s(vr0, vr6, vr0);
        vfmul_s(vr1, vr6, vr1);
        vfmul_s(vr2, vr6, vr2);
        vfmul_s(vr3, vr6, vr3);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        vst(vr2, B, -0x60);
        vst(vr3, B, -0x50);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 16);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[16]);

        L(labels[15]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[13]);
        vldrepl_d(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        ldx_d(TM0, A1, LDA2);
        vinsgr2vr_d(vr0, TM0, 1);
        ldx_d(TM0, A1, LDA3);
        vinsgr2vr_d(vr1, TM0, 1);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr1, vr1, vr0);
        vilvl_d(vr0, vr1, vr4);
        vilvh_d(vr1, vr1, vr4);
        vfmul_s(vr0, vr6, vr0);
        vfmul_s(vr1, vr6, vr1);
        vst(vr0, B, -0x80);
        vst(vr1, B, -0x70);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 8);
        addi_d(B, B, 32);

        L(labels[13]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[12]);
        vldrepl_w(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vldx(vr2, A1, LDA2);
        vldx(vr3, A1, LDA3);
        vilvl_w(vr2, vr3, vr2);
        vilvl_d(vr0, vr2, vr0);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(A2, A1, LDA4);
        addi_d(A1, A1, 4);
        addi_d(B, B, 16);

        L(labels[12]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[17]);

        L(labels[11]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[6]);
        add_d(A1, A, zero);
        add_d(A, A, LDA2);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[9]);

        L(labels[10]);
        vld(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr4, vr1, vr0);
        vilvh_w(vr1, vr1, vr0);
        vbsll_v(vr0, vr4, 0);
        vfmul_s(vr0, vr6, vr0);
        vfmul_s(vr1, vr6, vr1);
        vstelm_d(vr0, B, -0x80, 0);
        vstelm_d(vr0, B, -0x78, 1);
        vstelm_d(vr1, B, -0x70, 0);
        vstelm_d(vr1, B, -0x68, 1);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 16);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[10]);

        L(labels[9]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[8]);
        vldrepl_d(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        vstelm_d(vr0, B, -0x78, 1);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 8);
        addi_d(B, B, 16);

        L(labels[8]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[7]);
        vldrepl_w(vr0, A1, 0);
        vldx(vr1, A1, LDA);
        vilvl_w(vr0, vr1, vr0);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(A2, A1, LDA2);
        addi_d(A1, A1, 4);
        addi_d(B, B, 8);

        L(labels[7]);
        addi_d(N, N, -0x2);

        L(labels[6]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[1]);
        add_d(A1, A, zero);
        add_d(A, A, LDA);
        srai_d(I, M, 0x2);
        bge(zero, I, labels[4]);

        L(labels[5]);
        vld(vr0, A1, 0);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        vstelm_w(vr0, B, -0x7c, 1);
        vstelm_w(vr0, B, -0x78, 2);
        vstelm_w(vr0, B, -0x74, 3);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 16);
        addi_d(B, B, 16);
        addi_d(I, I, -1);
        blt(zero, I, labels[5]);

        L(labels[4]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[3]);
        vldrepl_d(vr0, A1, 0);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        vstelm_w(vr0, B, -0x7c, 1);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 8);
        addi_d(B, B, 8);

        L(labels[3]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[2]);
        vldrepl_w(vr0, A1, 0);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(A2, A1, LDA);
        addi_d(A1, A1, 4);
        addi_d(B, B, 4);

        L(labels[2]);
        addi_d(N, N, -0x1);

        L(labels[1]);

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
#undef TM0
#undef LDA2
#undef LDA4
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
