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

#ifndef CPU_LOONGARCH64_JIT_UNI_DW_CONV_KERNEL_F32_HPP
#define CPU_LOONGARCH64_JIT_UNI_DW_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/loongarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/loongarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/loongarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/loongarch64/jit_generator.hpp"
#include "cpu/loongarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

template <cpu_isa_t isa>
struct jit_uni_dw_conv_fwd_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_fwd_kernel_f32)

    jit_uni_dw_conv_fwd_kernel_f32(
            const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md);

    jit_conv_conf_t jcp;

private:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    const int vlen = cpu_isa_traits<isa>::vlen;

    // dw convolution
    Xbyak_loongarch64::XReg reg_input = a3;//r8;
    Xbyak_loongarch64::XReg aux_reg_input = t1;//r9;
    Xbyak_loongarch64::XReg reg_kernel = t3;//r10;
    Xbyak_loongarch64::XReg aux_reg_kernel = t4;//r11;
    Xbyak_loongarch64::XReg reg_ch_blocks = t5;//r12;
    Xbyak_loongarch64::XReg reg_output = a4;//r13;
    Xbyak_loongarch64::XReg reg_bias = a5;//r14;
    Xbyak_loongarch64::XReg reg_kh = a6;//r15;
    Xbyak_loongarch64::XReg iter_kh = t0;//rax;
    Xbyak_loongarch64::XReg reg_oi = t6;//rbx;
    Xbyak_loongarch64::XReg aux_reg_ch_blocks = t7;//rsi;
    // fused convolution
    Xbyak_loongarch64::XReg reg_input_buffer_ptr = t8;//rdx;
    Xbyak_loongarch64::XReg aux_reg_input_buffer_ptr = a7;//rbp;
    Xbyak_loongarch64::XReg reg_iw_offset = reg_input; //Hack: clear reg_input early in kernel

    Xbyak_loongarch64::XReg reg_tail = t0;//rax;

    inline void load_src(int ur_ch_blocks, int ur_w, bool is_ch_tail);
    inline void compute_loop(int ur_w, int ur_ch_blocks, int pad_l, int pad_r);
    inline void ow_loop(int ur_ch_blocks);
    inline void apply_filter_unrolled(
            int ur_ch_blocks, int ur_w, int pad_l, int pad_r, bool is_ch_tail);
    inline void apply_postops(
            const int ur_ch_blocks, const int ur_w, const bool is_ch_tail);
    inline void store_dst(int ur_ch_blocks, int ur_w, bool is_ch_tail);

    int max_repeats() { return jcp.isa == lsx ? 2 : 1; }

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline int get_acc_reg_idx(int idx) {
        const int max_regs = 16;
        return idx + (max_regs - jcp.ur_w * jcp.nb_ch_blocking * max_repeats());
    }
    inline Vmm get_acc_reg(int idx) { return Vmm(get_acc_reg_idx(idx)); }

    void load_tail(
            Vmm &vmm, const Xbyak_loongarch64::XReg &reg, int64_t offset, int load_size);
    void add_tail_from_mem(Vmm &vmm_acc, Vmm &vmm_tmp, const Xbyak_loongarch64::XReg &reg,
            int64_t offset, int load_size);
    void store_tail(
            Vmm &vmm, const Xbyak_loongarch64::XReg &reg, int64_t offset, int store_size);

    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }

    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_dst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }

    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

    void generate() override;
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_data_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_data_kernel_f32)

    jit_uni_dw_conv_bwd_data_kernel_f32(const jit_conv_conf_t &ajcp)
        : jcp(ajcp) {}
    jit_conv_conf_t jcp;

private:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const int reg_repeats_ = (isa == lsx) ? 2 : 1;
    const int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);
    using reg64_t = const Xbyak_loongarch64::XReg;

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_ddst_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    reg64_t reg_ddst = a3; //rax;
    reg64_t aux_reg_ddst = t1; //r8;
    reg64_t aux1_reg_ddst = abi_not_param1;
    reg64_t reg_kernel = a6; //rdx;
    reg64_t aux_reg_kernel = t3; //r10;
    reg64_t aux1_reg_kernel = a7; //rbp;
    reg64_t reg_dsrc = a5; //rsi;

    reg64_t reg_ur_str_w = t2; //r9;
    reg64_t reg_ch_blocks = a4; //rbx;

    reg64_t iter_kh = t4; //r11;
    reg64_t iter_kw = t5; //r12;
    reg64_t reg_kh = t6; //r13;
    reg64_t reg_kw = t7; //r14;

    reg64_t aux_reg_ch_blocks = a2; //r15;
    reg64_t reg_tmp = a2; //r15;
    Xbyak_loongarch64::XReg k_ch_tail_mask = Xbyak_loongarch64::XReg(1);


    void load_vmm(Vmm &vmm, const Xbyak_loongarch64::XReg &addr, int offset, bool tail);
    void store_vmm(Vmm &vmm, const Xbyak_loongarch64::XReg &addr, int offset, bool tail);

    inline void ch_loop_body(int ur_ch_blocks, int unroll_w);
    inline void unroll_width_body(int ur_ch_blocks);
    inline void load_ddst(int ur_ch_blocks, int ur_str_w);
    inline void apply_filter(int ur_ch_blocks, int ur_str_w, bool is_last_ch);
    inline void store_dsrc(int ur_ch_blocks, int ur_str_w, bool is_last_ch);

    void generate() override;

    inline bool tail_simd_overlap(int reg_repeat) {
        return reg_repeat * simd_w_ >= jcp.ch_tail;
    }

    inline bool is_dsrc_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_weights_kernel_f32 : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_weights_kernel_f32)

    jit_uni_dw_conv_bwd_weights_kernel_f32(const jit_conv_conf_t &ajcp)
        : jcp(ajcp) {}

    jit_conv_conf_t jcp;

private:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int reg_repeats_ = (isa == lsx) ? 2 : 1;
    const int req_aux_vmm = (isa == lsx) ? 1 : 0;
    const int max_unroll_w_ = 30;
    const int block_size_ = 15;

    /* Offset between input and accummulators is 3, therefore, assume 'kw'
     * is no larger than 3*/
    inline Vmm get_bias_reg(int idx = 0) { return Vmm(idx); }
    inline Vmm get_output_reg(int idx) {
        int vmm_idx = jcp.is_fast_depthwise
                ? idx + 2 * jcp.kw * jcp.nb_ch_blocking
                : idx + req_aux_vmm;
        return Vmm(vmm_idx);
    }
    inline Vmm get_input_reg(int idx) {
        int vmm_idx = jcp.is_fast_depthwise
                ? idx + jcp.kw * jcp.nb_ch_blocking
                : idx + 4 * reg_repeats_ + req_aux_vmm;
        return Vmm(vmm_idx);
    }
    inline Vmm get_acc_reg(int idx) {
        int vmm_idx = jcp.is_fast_depthwise
                ? idx
                : idx + 1 * reg_repeats_ + req_aux_vmm;
        return Vmm(vmm_idx);
    }
    inline Vmm get_aux_reg() { return Vmm(0); }


    using reg64_t = const Xbyak_loongarch64::XReg;  //const Xbyak::Reg64;
    reg64_t reg_tmp_input = t2; //r9;
    reg64_t reg_tmp_output = t3; //r10;
    reg64_t reg_tmp_filter = t6; //r13;
    reg64_t reg_kh_offset = a3; //rax;

    /* parameter passed by driver into kernel */
    Xbyak_loongarch64::XReg reg_exec_flags = t0;

    reg64_t reg_oh_worksize = t7; //r14;
    reg64_t reg_oh = a3; //rax;

    reg64_t reg_iter_ow_blk = t4; //r11;

    reg64_t reg_kh_aux = a5; //rsi;
    reg64_t reg_kh = a6; //rdx;

    /* Base addresses for convolution parameters. */
    reg64_t reg_input_baddr = a2; //r15;
    reg64_t reg_output_baddr = t5; //r12;
    reg64_t reg_filter_baddr = abi_not_param1;
    reg64_t reg_bias_baddr = t6; //r13;

    reg64_t reg_tmp = t1; //r8;

    Xbyak_loongarch64::XReg k_ch_tail_mask = Xbyak_loongarch64::XReg(1);

    void addps_xmm(Vmm &vmm_dst, Vmm &vmm_src, const Xbyak_loongarch64::XReg &addr,
            int offset, bool compute_tail);
    void load_xmm(
            Vmm &vmm, const Xbyak_loongarch64::XReg &addr,
            int offset, bool compute_tail = false);
    void store_xmm(
            Vmm &vmm, const Xbyak_loongarch64::XReg &addr,
            int offset, bool compute_tail = false);

    void dispatch_ow_step_unroll(int unroll_w, int l_pad, int pad_offset,
            int ow_block, int nb_ch_blocking, bool is_last_ch);

    /* Micro-kernel JIT'ing, fusing 'kw' and 'ow_block' loops into unrolled FMAs
     */
    void compute_unroll_ow_step(int unroll_w, int l_pad, int pad_offset,
            int ow_block, bool is_last_ch);

    /* Micro-kernel JIT'ing, fusing 'kw', 'ow_block' and 'nb_ch_blocking' loops
     * into unrolled FMAs. */
    void compute_unroll_ow_step_nxc(int unroll_w, int l_pad, int pad_offset,
            int ow_block, int nb_ch_blocking, bool is_last_ch);

    /* JIT'ing the outer loops for the micro-kernel -> {kh, oh_block} */
    void compute_kh_step(int unroll_w, int l_pad, int pad_offset, int ow_block,
            int nb_ch_blocking, bool is_last_ch);
    /* Channel loop for 'nxc' format */
    void compute_ch_loop(int unroll_w, int l_pad, int pad_offset, int ow_block);
    void compute_h_loop(int unroll_w, int l_pad, int pad_offset, int ow_block);

    /* Write 'width' micro-kernel JITs; depending on the padding and convolution
     * size, write a micro-kernel for the left ow-block, middle ow-block(s), and
     * right ow-block.*/
    void compute_ow_block_unroll();

    void deploy_zero_filter();
    void zero_filter_ch_loop();
    void zero_filter_kh_loop(int nb_ch_blocking = 1);
    void load_filter(int nb_ch_blocking, bool is_last_ch = false);
    void zero_filter();
    void load_bias(int nb_ch_blocking, bool is_last_ch);
    void zero_bias();
    void compute_bias_step_unroll(
            const int unroll_w, int nb_ch_blocking, bool is_last_ch);
    void compute_ch_loop_bias(bool do_load_bias);
    void deploy_ch_loop_bias();
    void compute_single_ch_block_bias();
    void compute_spatial_loop_bias(int nb_ch_blocking, bool is_last_ch);
    void store_filter(int nb_ch_blocking, bool is_last_ch = false);
    void store_bias(int nb_ch_blocking, bool is_last_ch);
    void compute_bias();
    void calculate_w_unrolling(
            int &unroll_trips, int &unroll_w, int &unroll_w_tail);

    void generate() override;

    inline bool is_layout_nxc() {
        return utils::everyone_is(
                true, is_src_layout_nxc(), is_ddst_layout_nxc());
    }
    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
