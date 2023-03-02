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

jit_lasx_f32_copy_bt_kern::jit_lasx_f32_copy_bt_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_f32_copy_bt_kern::generate() {

#define M a0    //rdi
#define N a1    //rsi
#define A a2    //rdx
#define LDA a3  //rcx
#define ALPHA a4//r8
#define B a5    //r9

#define I t5    //rax
#define A1 t6   //r10
#define A2 a4   //r8
#define LDA3 t7 //r11
#define LDA4 t4 //new add
#define LDA8 t8 //new add
#define TM s1
#define TM0 s0

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch64::Label> labels(59);
        preamble();

        ld_d(M, M, 0);
        ld_d(N, N, 0);
        ld_d(LDA, LDA, 0);
        addi_d(A, A, 128);
        addi_d(B, B, 128);
        slli_d(LDA, LDA, 0x2);
        slli_d(LDA4, LDA, 2);
        sub_d(LDA3, LDA4, LDA);
        add_d(LDA8, LDA4, LDA4);
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
        bnez(TM, labels[42]);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[55]);

        L(labels[36]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x10);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[0]);

        L(labels[30]);
        vld(vr0, A1, -0x80);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x70);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x60);
        add_d(TM, A1, LDA3);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x50);
        vld(vr0, A2, -0x80);
        vst(vr0, B, -0x40);
        add_d(TM, A2, LDA);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x30);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x20);
        add_d(TM, A2, LDA3);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x10);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 128);
        addi_d(I, I, -1);
        blt(zero, I, labels[30]);

        L(labels[0]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[58]);
        vld(vr0, A1, -0x80);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x70);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x60);
        add_d(TM, A1, LDA3);
        vld(vr0, TM, -0x80);
        vst(vr0, B, -0x50);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 64);

        L(labels[58]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[57]);
        vld(vr0, A1, -0x80);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0,TM, -0x80);
        vst(vr0, B, -0x70);
        add_d(A1, TM, LDA);
        addi_d(B, B, 32);

        L(labels[57]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[56]);
        vld(vr0, A1, -0x80);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[56]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[36]);

        L(labels[55]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[49]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x8);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[53]);

        L(labels[54]);
        ld_d(TM0, A1, -0x80);
        st_d(TM0, B, -0x80);
        add_d(TM, A1, LDA);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x78);
        add_d(TM, TM, LDA);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x70);
        add_d(TM, A1, LDA3);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x68);
        ld_d(TM0, A2, -0x80);
        st_d(TM0, B, -0x60);
        add_d(TM, A2, LDA);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x58);
        add_d(TM, TM, LDA);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x50);
        add_d(TM, A2, LDA3);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x48);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[54]);

        L(labels[53]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[52]);
        ld_d(TM0, A1, -0x80);
        st_d(TM0, B, -0x80);
        add_d(TM, A1, LDA);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x78);
        add_d(TM, TM, LDA);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x70);
        add_d(TM, A1, LDA3);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x68);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 32);

        L(labels[52]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[51]);
        ld_d(TM0, A1, -0x80);
        st_d(TM0, B, -0x80);
        add_d(TM, A1, LDA);
        ld_d(TM0, TM, -0x80);
        st_d(TM0, B, -0x78);
        add_d(A1, TM, LDA);
        addi_d(B, B, 16);

        L(labels[51]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[50]);
        ld_d(TM0, A1, -0x80);
        st_d(TM0, B, -0x80);
        addi_d(B, B, 8);

        L(labels[50]);
        addi_d(N, N, -0x2);

        L(labels[49]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[43]);
        add_d(A1, zero, A);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x4);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[47]);

        L(labels[48]);
        ld_w(TM0, A1, -0x80);
        st_w(TM0, B, -0x80);
        add_d(TM, A1, LDA);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x7c);
        add_d(TM, TM, LDA);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x78);
        add_d(TM, A1, LDA3);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x74);
        ld_w(TM0, A2, -0x80);
        st_w(TM0, B, -0x70);
        add_d(TM, A2, LDA);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x6c);
        add_d(TM, TM, LDA);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x68);
        add_d(TM, A2, LDA3);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x64);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[48]);

        L(labels[47]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[46]);
        ld_w(TM0, A1, -0x80);
        st_w(TM0, B, -0x80);
        add_d(TM, A1, LDA);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x7c);
        add_d(TM, TM, LDA);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x78);
        add_d(TM, A1, LDA3);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x74);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 16);

        L(labels[46]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[45]);
        ld_w(TM0, A1, -0x80);
        st_w(TM0, B, -0x80);
        add_d(TM, A1, LDA);
        ld_w(TM0, TM, -0x80);
        st_w(TM0, B, -0x7c);
        add_d(A1, TM, LDA);
        addi_d(B, B, 8);

        L(labels[45]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[44]);
        ld_w(TM0, A1, -0x80);
        st_w(TM0, B, -0x80);
        addi_d(B, B, 4);

        L(labels[44]);
        addi_d(N, N, -0x1);

        L(labels[43]);
        b(labels[1]);

        L(labels[42]);
        vxor_v(vr3, vr3, vr4);
        vfcmp_cle_s(vr31, vr3, vr6);
        vpickve2gr_w(TM, vr31, 0);
        bnez(TM, labels[20]);
        xvbsll_v(xr6, xr4, 0);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[34]);

        L(labels[41]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x10);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[39]);

        L(labels[40]);
        vld(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x70);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x60);
        add_d(TM, A1, LDA3);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x50);
        vld(vr0, A2, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x40);
        add_d(TM, A2, LDA);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x30);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x20);
        add_d(TM, A2, LDA3);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x10);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 128);
        addi_d(I, I, -1);
        blt(zero, I, labels[40]);

        L(labels[39]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[38]);
        vld(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x70);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x60);
        add_d(TM, A1, LDA3);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x50);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 64);

        L(labels[38]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[37]);
        vld(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x70);
        add_d(A1, TM, LDA);
        addi_d(B, B, 32);

        L(labels[37]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[35]);
        vld(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[35]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[41]);

        L(labels[34]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[27]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x8);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[32]);

        L(labels[33]);
        vldrepl_d(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x78, 0);
        add_d(TM, TM, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x70, 0);
        add_d(TM, A1, LDA3);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x68, 0);
        vldrepl_d(vr0, A2, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x60, 0);
        add_d(TM, A2, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x58, 0);
        add_d(TM, TM, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x50, 0);
        add_d(TM, A2, LDA3);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x48, 0);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[33]);

        L(labels[32]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[31]);
        vldrepl_d(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x78, 0);
        add_d(TM, TM, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x70, 0);
        add_d(TM, A1, LDA3);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x68, 0);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 32);

        L(labels[31]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[29]);
        vldrepl_d(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x78, 0);
        add_d(A1, TM, LDA);
        addi_d(B, B, 16);

        L(labels[29]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[28]);
        vldrepl_d(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);

        L(labels[28]);
        addi_d(N, N, -0x2);

        L(labels[27]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[21]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x4);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[25]);

        L(labels[26]);
        vldrepl_w(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x7c, 0);
        add_d(TM, TM, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x78, 0);
        add_d(TM, A1, LDA3);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x74, 0);
        vldrepl_w(vr0, A2, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x70, 0);
        add_d(TM, A2, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x6c, 0);
        add_d(TM, TM, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x68, 0);
        add_d(TM, A2, LDA3);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x64, 0);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[26]);

        L(labels[25]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[24]);
        vldrepl_w(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x7c, 0);
        add_d(TM, TM, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x78, 0);
        add_d(TM, A1, LDA3);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x74, 0);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 16);

        L(labels[24]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[23]);
        vldrepl_w(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x7c, 0);
        add_d(A1, TM, LDA);
        addi_d(B, B, 8);

        L(labels[23]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[22]);
        vldrepl_w(vr0, A1, -0x80);
        vxor_v(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        addi_d(B, B, 4);

        L(labels[22]);
        addi_d(N, N, -0x1);

        L(labels[21]);
        b(labels[1]);

        L(labels[20]);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[13]);

        L(labels[19]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x10);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[17]);

        L(labels[18]);
        vld(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, - 0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x70);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x60);
        add_d(TM, A1, LDA3);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x50);
        vld(vr0, A2, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x40);
        add_d(TM, A2, LDA);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x30);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x20);
        add_d(TM, A2, LDA3);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x10);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 128);
        addi_d(I, I, -1);
        blt(zero, I, labels[18]);

        L(labels[17]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[16]);
        vld(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x70);
        add_d(TM, TM, LDA);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x60);
        add_d(TM, A1, LDA3);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x50);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 64);

        L(labels[16]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[15]);
        vld(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        add_d(TM, A1, LDA);
        vld(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x70);
        add_d(A1, TM, LDA);
        addi_d(B, B, 32);

        L(labels[15]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[14]);
        vld(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[14]);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[19]);

        L(labels[13]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[7]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x8);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[11]);

        L(labels[12]);
        vldrepl_d(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x78, 0);
        add_d(TM, TM, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x70, 0);
        add_d(TM, A1, LDA3);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x68, 0);
        vldrepl_d(vr0, A2, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x60, 0);
        add_d(TM, A2, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x58, 0);
        add_d(TM, TM, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x50, 0);
        add_d(TM, A2, LDA3);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x48, 0);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[12]);

        L(labels[11]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[10]);
        vldrepl_d(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x78, 0);
        add_d(TM, TM, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x70, 0);
        add_d(TM, A1, LDA3);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x68, 0);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 32);

        L(labels[10]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[9]);
        vldrepl_d(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_d(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x78, 0);
        add_d(A1, TM, LDA);
        addi_d(B, B, 16);

        L(labels[9]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[8]);
        vldrepl_d(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_d(vr0, B, -0x80, 0);
        addi_d(B, B, 8);

        L(labels[8]);
        addi_d(N, N, -0x2);

        L(labels[7]);
        addi_d(TM, zero, 0x1);
        blt(N, TM, labels[1]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA4);
        addi_d(A, A, 0x4);
        srai_d(I, M, 0x3);
        bge(zero, I, labels[5]);

        L(labels[6]);
        vldrepl_w(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x7c, 0);
        add_d(TM, TM, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x78, 0);
        add_d(TM, A1, LDA3);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x74, 0);
        vldrepl_w(vr0, A2, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x70, 0);
        add_d(TM, A2, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x6c, 0);
        add_d(TM, TM, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x68, 0);
        add_d(TM, A2, LDA3);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x64, 0);
        add_d(A1, A1, LDA8);
        add_d(A2, A2, LDA8);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[6]);

        L(labels[5]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[4]);
        vldrepl_w(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x7c, 0);
        add_d(TM, TM, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x78, 0);
        add_d(TM, A1, LDA3);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x74, 0);
        add_d(A1, A1, LDA4);
        addi_d(B, B, 16);

        L(labels[4]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[3]);
        vldrepl_w(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
        add_d(TM, A1, LDA);
        vldrepl_w(vr0, TM, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x7c, 0);
        add_d(A1, TM, LDA);
        addi_d(B, B, 8);

        L(labels[3]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[2]);
        vldrepl_w(vr0, A1, -0x80);
        vfmul_s(vr0, vr6, vr0);
        vstelm_w(vr0, B, -0x80, 0);
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
#undef LDA4
#undef LDA8
#undef TM
#undef TM0
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
