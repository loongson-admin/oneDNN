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

#ifndef CPU_LOONGARCH64_GEMM_S8X8S32_JIT_LASX_GEMM_S8U8S32_KERN_HPP
#define CPU_LOONGARCH64_GEMM_S8X8S32_JIT_LASX_GEMM_S8U8S32_KERN_HPP

#include "cpu/loongarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

class jit_lasx_gemm_s8u8s32_kern : public jit_generator {
public:
    jit_lasx_gemm_s8u8s32_kern(bool beta_zero, bool enable_offset_c,
            bool enable_offset_r, int unroll_m);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lasx_gemm_s8u8s32_kern);

protected:
    bool beta_zero_;
    bool enable_offset_c_, enable_offset_r_;
    bool vnni_;
    int unroll_m_;

    // use uni_preld instead in loongarch

    void c_load(const Xbyak_loongarch64::XVReg &dst, const Xbyak_loongarch64::XReg &src,
                int64_t offset, int nelems);
    void c_store(const Xbyak_loongarch64::XVReg &dst, const Xbyak_loongarch64::XReg &src,
                int64_t offset, int nelems);
    void dot_product(const Xbyak_loongarch64::XVReg &dst,
                    const Xbyak_loongarch64::XVReg &b,
                    const Xbyak_loongarch64::XVReg &a);

    void kernel_loop(int unroll_m, int unroll_n, bool cfetch);
    void remainder_kernel(int unroll_m, int unroll_n, int unroll_k, int bwidth);
    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y, Xbyak_loongarch64::Label *&outerloop_label);

    void generate() override ATTRIBUTE_OPTIMIZE;

private:
    static const int IGEMM_UNROLL_N_ = 4;

    static const int size_ = 4;
    static const int isize_ = 1;

    // Prefetch configuration
    static const int prefetch_size_a_ = 704;
    static const int prefetch_size_b_ = 384;

    static const int offset_a_ = 128, offset_b_ = 128;
    static const int max_unroll_m_ = 24, max_unroll_n_ = 4;

    // Integer register assignments
    Xbyak_loongarch64::XReg M_ = abi_param1;  //rdi
    Xbyak_loongarch64::XReg N_ = abi_param2;  //rsi
    Xbyak_loongarch64::XReg K_ = abi_param3;  //rdx
    Xbyak_loongarch64::XReg A_ = abi_param5;  //r8
    Xbyak_loongarch64::XReg B_ = abi_param6;  //r9
    Xbyak_loongarch64::XReg C_ = abi_param7;  //r10
    Xbyak_loongarch64::XReg LDC_ = abi_param8;    //r11

    Xbyak_loongarch64::XReg rax = t1;     // for calc in loongarch
    Xbyak_loongarch64::XReg I_ = t2;  // r12
    Xbyak_loongarch64::XReg J_ = t3;  //r13
    Xbyak_loongarch64::XReg LoopCount_ = t4;  // rax
    Xbyak_loongarch64::XReg AO_ = t5;     // r14
    Xbyak_loongarch64::XReg BO_ = t6;     // r15
    Xbyak_loongarch64::XReg CO1_ = t7;    // rbx
    Xbyak_loongarch64::XReg CO2_ = t8;    // rbp
    Xbyak_loongarch64::XReg AA_ = abi_param4; // rcx

    // Vector register assignments
    Xbyak_loongarch64::XVReg a_regs_[3] = {xr0, xr1, xr2};
    Xbyak_loongarch64::XVReg b_regs_ = xr3;
    Xbyak_loongarch64::XVReg c_regs_[3][4] = {xr8, xr9, xr10, xr11,
                                            xr12, xr13, xr14, xr15,
                                            xr4, xr5, xr6, xr7};

    // Stack variable assignments
    int stack_alloc_size_ = 96;
    int args_offset = stack_alloc_size_ + get_size_of_abi_save_regs();

    Xbyak_loongarch64::Address arg_coffset_c_ = ptr_a(sp, args_offset + 0);
    Xbyak_loongarch64::Address arg_coffset_r_ = ptr_a(sp, args_offset + 8);
    Xbyak_loongarch64::Address coffset_cx_ = ptr_a(sp, 64);
    Xbyak_loongarch64::Address coffset_cy_ = ptr_a(sp, 72);
    Xbyak_loongarch64::Address coffset_rx_ = ptr_a(sp, 80);
    Xbyak_loongarch64::Address coffset_ry_ = ptr_a(sp, 88);
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_LOONGARCH64_GEMM_S8X8S32_JIT_LASX_GEMM_S8U8S32_KERN_HPP
