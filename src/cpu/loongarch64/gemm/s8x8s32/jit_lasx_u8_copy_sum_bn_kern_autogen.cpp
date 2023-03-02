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

jit_lasx_u8_copy_sum_bn_kern::jit_lasx_u8_copy_sum_bn_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_lasx_u8_copy_sum_bn_kern::generate() {

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

#define ARG_BIAS (stacksize)

    inLocalLabel();
    {
        std::vector<Xbyak_loongarch64::Label> labels(24);

        preamble();
        auto stacksize = get_size_of_abi_save_regs();

        ld_d(N, N, 0);
        ld_d(M, M, 0);
        ld_d(LDA, LDA, 0);
        addi_d(A, A, 128);
        addi_d(B, B, 128);
        add_d(LDA3, LDA, LDA);
        add_d(LDA3, LDA3, LDA);
        mov_imm(TM, 0x4);
        blt(N, TM, labels[6]);

        L(labels[2]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA);
        add_d(A2, A2, LDA);
        add_d(I, A2, LDA);
        add_d(I, I, LDA);
        add_d(A, I, zero);
        vxor_v(vr7, vr7, vr7);
        srai_d(I, M, 0x4);
        bge(zero, I, labels[0]);

        L(labels[11]);
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
        vext2xv_hu_bu(xr5, xr0);
        vextrins_d(vr6, vr0, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr0, B, -0x80);
        vext2xv_hu_bu(xr5, xr1);
        vextrins_d(vr6, vr1, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr1, B, -0x70);
        vext2xv_hu_bu(xr5, xr4);
        vextrins_d(vr6, vr4, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr4, B, -0x60);
        vext2xv_hu_bu(xr5, xr3);
        vextrins_d(vr6, vr3, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr3, B, -0x50);
        addi_d(B, B, 64);
        addi_d(I, I, -1);
        blt(zero, I, labels[11]);

        L(labels[0]);
        andi(TM, M, 0x8);
        bge(zero, TM, labels[1]);
        load_bytes(vr0, A1, -0x80, 8);
        add_d(TM, A1, LDA);
        load_bytes(vr1, TM, -0x80, 8);
        addi_d(A1, A1, 8);
        load_bytes(vr2, A2, -0x80, 8);
        add_d(TM, A2, LDA);
        load_bytes(vr3, TM, -0x80, 8);
        addi_d(A2, A2, 8);
        vilvl_w(vr0, vr1, vr0);
        vilvl_w(vr2, vr3, vr2);
        vbsll_v(vr1, vr0, 0);
        vilvl_d(vr0, vr2, vr0);
        vilvh_d(vr1, vr2, vr1);
        vext2xv_hu_bu(xr5, xr0);
        vextrins_d(vr6, vr0, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr0, B, -0x80);
        vext2xv_hu_bu(xr5, xr1);
        vextrins_d(vr6, vr1, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr1, B, -0x70);
        addi_d(B, B, 32);

        L(labels[1]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[3]);
        load_bytes(vr0, A1, -0x80, 4);
        add_d(TM, A1, LDA);
        load_bytes(vr1, TM, -0x80, 4);
        addi_d(A1, A1, 4);
        load_bytes(vr2, A2, -0x80, 4);
        add_d(TM, A2, LDA);
        load_bytes(vr3, TM, -0x80, 4);
        addi_d(A2, A2, 4);
        vilvl_w(vr0, vr1, vr0);
        vilvl_w(vr2, vr3, vr2);
        vilvl_d(vr0, vr2, vr0);
        vext2xv_hu_bu(xr5, xr0);
        vextrins_d(vr6, vr0, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[3]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[4]);
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
        vext2xv_hu_bu(xr5, xr0);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        store_bytes(vr0, B, -0x80, 8);
        addi_d(B, B, 8);

        L(labels[4]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[5]);
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
        vext2xv_wu_bu(xr5, xr0);
        vadd_w(vr7, vr7, vr5);
        store_bytes(vr0, B, -0x80, 4);
        addi_d(B, B, 4);

        L(labels[5]);
        ld_d(A1, sp, ARG_BIAS);
        vst(vr7, A1, 0);
        addi_d(A1, A1, 0x10);
        st_d(A1, sp, ARG_BIAS);
        addi_d(N, N, -0x4);
        mov_imm(TM, 0x4);
        bge(N, TM, labels[2]);

        L(labels[6]);
        mov_imm(TM, 0x2);
        blt(N, TM, labels[15]);

        L(labels[7]);
        add_d(A1, A, zero);
        add_d(A2, A1, LDA);
        add_d(I, A2, LDA);
        add_d(A, I, zero);
        vxor_v(vr7, vr7, vr7);
        srai_d(I, M, 0x4);
        bge(zero, I, labels[9]);

        L(labels[8]);
        vld(vr0, A1, -0x80);
        addi_d(A1, A1, 16);
        vld(vr1, A2, -0x80);
        addi_d(A2, A2, 16);
        vbsll_v(vr2, vr0, 0);
        vilvl_w(vr0, vr1, vr0);
        vilvh_w(vr2, vr1, vr2);
        vshuf4i_w(vr6, vr0, 0xd8);
        vext2xv_hu_bu(xr5, xr6);
        vextrins_d(vr6, vr6, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr0, B, -0x80);
        vshuf4i_w(vr6, vr2, 0xd8);
        vext2xv_hu_bu(xr5, xr6);
        vextrins_d(vr6, vr6, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr2, B, -0x70);
        addi_d(B, B, 32);
        addi_d(I, I, -1);
        blt(zero, I, labels[8]);

        L(labels[9]);
        andi(TM, M, 0x8);
        bge(zero, TM, labels[10]);
        load_bytes(vr0, A1, -0x80, 8);
        addi_d(A1, A1, 8);
        load_bytes(vr1, A2, -0x80, 8);
        addi_d(A2, A2, 8);
        vilvl_w(vr0, vr1, vr0);
        vshuf4i_w(vr6, vr0, 0xd8);
        vext2xv_hu_bu(xr5, xr6);
        vextrins_d(vr6, vr6, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);

        L(labels[10]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[12]);
        load_bytes(vr0, A1, -0x80, 4);
        addi_d(A1, A1, 4);
        load_bytes(vr1, A2, -0x80, 4);
        addi_d(A2, A2, 4);
        vilvl_w(vr0, vr1, vr0);
        vext2xv_hu_bu(xr5, xr0);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        store_bytes(vr0, B, -0x80, 8);
        addi_d(B, B, 8);

        L(labels[12]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[13]);
        ld_h(TM0, A1, -0x80);
        addi_d(A1, A1, 2);
        vinsgr2vr_h(vr0, TM0, 0x0);
        ld_h(TM0, A2, -0x80);
        addi_d(A2, A2, 2);
        vinsgr2vr_h(vr0, TM0, 0x1);
        vext2xv_hu_bu(xr5, xr0);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        store_bytes(vr0, B, -0x80, 4);
        addi_d(B, B, 4);

        L(labels[13]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[14]);
        ld_b(TM0, A1, -0x80);
        vinsgr2vr_b(vr0, TM0, 0x0);
        st_b(TM0, B, -0x80);
        ld_b(TM0, A2, -0x80);
        vinsgr2vr_b(vr0, TM0, 0x1);
        st_b(TM0, B, -0x7f);
        addi_d(B, B, 2);
        vext2xv_wu_bu(xr5, xr0);
        vadd_w(vr7, vr7, vr5);

        L(labels[14]);
        ld_d(A1, sp, ARG_BIAS);
        store_bytes(vr7, A1, 0, 8);
        addi_d(A1, A1, 0x8);
        st_d(A1, sp, ARG_BIAS);
        addi_d(N, N, -0x2);
        mov_imm(TM, 0x2);
        bge(N, TM, labels[7]);

        L(labels[15]);
        mov_imm(TM, 0x1);
        blt(N, TM, labels[23]);

        L(labels[16]);
        add_d(A1, A, zero);
        add_d(A, A, LDA);
        vxor_v(vr7, vr7, vr7);
        srai_d(I, M, 0x4);
        bge(zero, I, labels[18]);

        L(labels[17]);
        vld(vr0, A1, -0x80);
        addi_d(A1, A1, 16);
        vext2xv_hu_bu(xr5, xr0);
        vextrins_d(vr6, vr0, 0x01);
        vext2xv_hu_bu(xr6, xr6);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        vst(vr0, B, -0x80);
        addi_d(B, B, 16);
        addi_d(I, I, -1);
        blt(zero, I, labels[17]);

        L(labels[18]);
        andi(TM, M, 0x8);
        bge(zero, TM, labels[19]);
        load_bytes(vr0, A1, -0x80, 8);
        addi_d(A1, A1, 8);
        vext2xv_hu_bu(xr5, xr0);
        vpickev_h(vr30, vr6, vr5);
        vpickod_h(vr31, vr6, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        store_bytes(vr0, B, -0x80, 8);
        addi_d(B, B, 8);

        L(labels[19]);
        andi(TM, M, 0x4);
        bge(zero, TM, labels[20]);
        load_bytes(vr0, A1, -0x80, 4);
        addi_d(A1, A1, 4);
        vext2xv_hu_bu(xr5, xr0);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        store_bytes(vr0, B, -0x80, 4);
        addi_d(B, B, 4);

        L(labels[20]);
        andi(TM, M, 0x2);
        bge(zero, TM, labels[21]);
        ld_h(TM0, A1, -0x80);
        vinsgr2vr_h(vr0, TM0, 0x0);
        vext2xv_hu_bu(xr5, xr0);
        vpickev_h(vr30, vr5, vr5);
        vpickod_h(vr31, vr5, vr5);
        vadd_h(vr5, vr30, vr31);
        vext2xv_wu_hu(xr5, xr5);
        vadd_w(vr7, vr7, vr5);
        st_h(TM0, B, -0x80);
        addi_d(A1, A1, 2);
        addi_d(B, B, 2);

        L(labels[21]);
        andi(TM, M, 0x1);
        bge(zero, TM, labels[22]);
        ld_b(TM0, A1, -0x80);
        vinsgr2vr_b(vr0, TM0, 0x0);
        vext2xv_wu_bu(xr5, xr0);
        vadd_w(vr7, vr7, vr5);
        st_b(TM0, B, -0x80);
        addi_d(B, B, 1);

        L(labels[22]);
        ld_d(A1, sp, ARG_BIAS);
        store_bytes(vr7, A1, 0, 4);
        addi_d(A1, A1, 0x4);
        st_d(A1, sp, ARG_BIAS);
        addi_d(N, N, -0x1);
        mov_imm(TM, 0x1);
        bge(N, TM, labels[16]);

        L(labels[23]);
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
#undef ARG_BIAS
}

} //namespace x64
} //namespace cpu
} //namespace impl
} //namespace dnnl
