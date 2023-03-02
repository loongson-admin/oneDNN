/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_LOONGARCH64_JIT_LASX_1X1_CONV_KERNEL_F32_HPP
#define CPU_LOONGARCH64_JIT_LASX_1X1_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/loongarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/loongarch64/jit_generator.hpp"
#include "cpu/loongarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

struct jit_lasx_1x1_conv_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lasx_1x1_conv_kernel_f32)

    jit_lasx_1x1_conv_kernel_f32(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<lasx>>
            postops_injector_;

    constexpr static int isa_simd_width_
            = cpu_isa_traits<lasx>::vlen / sizeof(float);

    Xbyak_loongarch64::XReg reg_bcast_data = t0;
    Xbyak_loongarch64::XReg reg_load_data = t1;
    Xbyak_loongarch64::XReg reg_output_data = t2;
    Xbyak_loongarch64::XReg aux_reg_bcast_data = t3;
    Xbyak_loongarch64::XReg aux1_reg_bcast_data = abi_not_param1;
    Xbyak_loongarch64::XReg aux_reg_load_data = abi_param1;
    Xbyak_loongarch64::XReg aux_reg_output_data = t4;
    Xbyak_loongarch64::XReg reg_load_loop_work = t5;
    Xbyak_loongarch64::XReg reg_bcast_loop_work = t6;
    Xbyak_loongarch64::XReg reg_reduce_loop_work = t7;
    Xbyak_loongarch64::XReg load_loop_iter = a6;
    Xbyak_loongarch64::XReg bcast_loop_iter = a2;
    Xbyak_loongarch64::XReg reduce_loop_iter = a3;
    Xbyak_loongarch64::XReg imm_addr64 = reduce_loop_iter;
    Xbyak_loongarch64::XReg reg_reduce_pos_flag = a4;
    Xbyak_loongarch64::XReg reg_output_stride = a5;
    Xbyak_loongarch64::XReg reg_bias_data = a5;
    Xbyak_loongarch64::XReg reg_diff_bias_data = bcast_loop_iter;
    Xbyak_loongarch64::XReg reg_tmp_output_stride = reg_bcast_data;
    Xbyak_loongarch64::XReg reg_tmp = aux_reg_bcast_data;
    Xbyak_loongarch64::XReg reg_output_stride_scale = load_loop_iter;

    constexpr static int reg64_size_ = sizeof(int64_t);
    constexpr static int reg_diff_bias_data_stack_offt = 0;
    constexpr static int reg_binary_post_op_acc_off = 1 * reg64_size_;
    constexpr static int reg_abi_param1_backup = 2 * reg64_size_;
    constexpr static int stack_space_needed = 3 * reg64_size_;

    Xbyak_loongarch64::XVReg vreg_bcast = Xbyak_loongarch64::XVReg(15);
    Xbyak_loongarch64::XVReg vtmp = Xbyak_loongarch64::XVReg(14);

    void apply_postops(
            const int load_loop_blk, const int ur, const int load_dim_tail);
    void generate_bcast_loop(int load_loop_blk);
    void generate_reduce_loop(int load_loop_blk, int ur);
    void generate_diff_bias_loop(int load_loop_blk);

    void generate() override;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
