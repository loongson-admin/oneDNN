/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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
#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/loongarch64/cpu_barrier.hpp"
#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/jit_uni_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

namespace {

using namespace memory_tracking::names;

using namespace Xbyak_loongarch64;
namespace barrier = simple_barrier;

using acc_data_t = float;

template <cpu_isa_t isa>
struct jit_bnorm_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        size_t N_ithr, N_nthr;
        size_t coff_max, soff_max;
        size_t mb_stride_Bc, spat_size, spat_size_loc;
        size_t S_s, S_tail;
        size_t is_cblk_tail;
        acc_data_t chan_size, eps, one;
        const acc_data_t *scale;
        const acc_data_t *shift;
        const acc_data_t *mean, *var;
        const acc_data_t *diff_scale;
        const acc_data_t *diff_shift;
        const void *src, *dst;
        const void *diff_src, *diff_dst;
        const acc_data_t *rbuf1, *rbuf2;
        const uint8_t *ws;
        barrier::ctx_64_t *barrier;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    /* cpu specific part */
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    XReg rsp = XReg(sp.getIdx());  //use rsp as sp
    const int vlen = cpu_isa_traits<isa>::vlen;
    int vlen_spat_data_; // set by ctor depending on data type (BF16 or FP32);

    const batch_normalization_pd_t *bdesc_;
    bool is_spatial_thr_;
    bool is_nspc_;
    bool is_bf16_;
    uint32_t tail_ = 0;

    XReg reg_param = abi_param1;

    XReg reg_scale = a3;
    XReg reg_rbuf1 = a1;
    XReg reg_rbuf2 = a2;
    XReg reg_coff_max_fwd_copy = reg_rbuf2;

    XReg reg_mean = a5;
    XReg reg_var = reg_param;
    XReg reg_diff_scale = a7;
    XReg reg_coff_max_bwd_copy = reg_diff_scale;
    XReg reg_shift = reg_rbuf1;

    XReg reg_coff = t0;
    XReg reg_coff_max = t1;
    XReg reg_soff = t2;
    XReg reg_soff_max = t3;
    XReg reg_diff_shift = reg_soff_max;
    XReg reg_ctr = t4;
    XReg reg_roff = t5;

    XReg reg_mb_stride_Bc = t6;
    XReg reg_soff_nspc = reg_mb_stride_Bc;
    XReg reg_src = t7;
    XReg reg_diff_src = reg_rbuf1;
    XReg reg_dst = a6;
    XReg reg_diff_dst = reg_dst;
    XReg reg_tmp_off = reg_roff;

    // Reuse loop counters
    XReg reg_bar = reg_coff;
    XReg reg_nnthr = reg_soff; // must be usable w/ loops over coff
    XReg reg_tmp = reg_ctr;

    // Relu section
    bool with_relu, with_relu_inf_only;
    Vmm vzero = Vmm(0); // is_fwd() ? vdiff_beta : vbeta
    XReg reg_ws = reg_roff;
    Label l_relu_mask_avx2;

    size_t unroll_blocks;
    size_t unroll_regs;

    Vmm vbuf = Vmm(5);
    Vmm vdiff_beta = Vmm(6);
    Vmm vdiff_gamma = Vmm(7);
    Vmm vsqrtvar = Vmm(8);
    Vmm vone = Vmm(9);
    Vmm vmean = Vmm(10);
    Vmm vgamma = Vmm(11);
    Vmm vbeta = Vmm(12);
    Vmm veps = Vmm(13);
    Vmm vchan_size = Vmm(14);
    Vmm vtail_mask = Vmm(15);
    Vmm v_tmp0 = Vmm(16);

    size_t t0_pf_offt;
    size_t t1_pf_offt;
    size_t spat_size;
    size_t chan_data_offt;
    size_t spat_step;
    size_t mb_offt;
    size_t ws_mb_offt;

    enum {
        stack_off_N_nthr = 0,
        stack_off_N_ithr = 8,
        stack_off_src = 16,
        stack_off_dst = 24,
        stack_off_diff_src = 32,
        stack_off_diff_dst = 40,
        stack_off_diff_scale = 48,
        stack_off_ws = 56,
        stack_off_barrier = 64,
        stack_off_spat_size_loc = 72,
        stack_off_s_s = 80,
        stack_off_s_tail = 88,
        stack_off_is_cblk_tail = 96,
        stack_off_ws_off_copy = 104,
        stack_off_shift = 112,
        stack_off_diff_shift = 120,
        stack_off_soff_max = 128,
        stack_size_required = 136,
    };

    int bit_shift() { return 5 - is_bf16_; }

    bool stream_store_supported() { return !is_bf16_; }

    bool is_c_padded() const {
        const memory_desc_wrapper data_d(bdesc_->src_md());
        return bdesc_->C() != data_d.padded_dims()[1];
    }

    void compute_static_strides() {
        spat_size = bdesc_->D() * bdesc_->W() * bdesc_->H();
        chan_data_offt = bdesc_->C() * sizeof(acc_data_t);
        spat_step
                = is_nspc_ ? chan_data_offt / (1 + is_bf16_) : vlen_spat_data_;
        mb_offt = spat_step * spat_size;
        ws_mb_offt = (spat_step / (is_bf16_ ? 16 : 32)) * spat_size;

        if (false) {
            t0_pf_offt = 4096;
            t1_pf_offt = 0;
        } else {
            t0_pf_offt = 0;
            t1_pf_offt = 0;
        }
    }

    void load_common_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
        ld_d(reg_rbuf1, reg_param, PARAM_OFF(rbuf1));
        if (bdesc_->is_bwd()) ld_d(reg_rbuf2, reg_param, PARAM_OFF(rbuf2));

        ld_d(reg_coff_max, reg_param, PARAM_OFF(coff_max));
        ld_d(reg_soff_max, reg_param, PARAM_OFF(soff_max));
        ld_d(reg_mb_stride_Bc, reg_param, PARAM_OFF(mb_stride_Bc));
        slli_d(reg_coff_max, reg_coff_max, 2);

        ld_d(reg_mean, reg_param, PARAM_OFF(mean));
        ld_d(reg_scale, reg_param, PARAM_OFF(scale));

        xvldrepl_w(vchan_size, reg_param, PARAM_OFF(chan_size));
        xvldrepl_w(vone, reg_param, PARAM_OFF(one));
        xvldrepl_w(veps, reg_param, PARAM_OFF(eps));

        ld_d(reg_tmp, reg_param, PARAM_OFF(N_nthr));
        st_d(reg_tmp, rsp, stack_off_N_nthr);
        ld_d(reg_tmp, reg_param, PARAM_OFF(N_ithr));
        st_d(reg_tmp, rsp, stack_off_N_ithr);
        ld_d(reg_tmp, reg_param, PARAM_OFF(src));
        st_d(reg_tmp, rsp, stack_off_src);
        ld_d(reg_tmp, reg_param, PARAM_OFF(dst));
        st_d(reg_tmp, rsp, stack_off_dst);
        ld_d(reg_tmp, reg_param, PARAM_OFF(diff_src));
        st_d(reg_tmp, rsp, stack_off_diff_src);
        ld_d(reg_tmp, reg_param, PARAM_OFF(diff_dst));
        st_d(reg_tmp, rsp, stack_off_diff_dst);
        ld_d(reg_tmp, reg_param, PARAM_OFF(ws));
        st_d(reg_tmp, rsp, stack_off_ws);
        ld_d(reg_tmp, reg_param, PARAM_OFF(barrier));
        st_d(reg_tmp, rsp, stack_off_barrier);
        if (is_spatial_thr_) {
            ld_d(reg_tmp, reg_param, PARAM_OFF(spat_size_loc));
            st_d(reg_tmp, rsp, stack_off_spat_size_loc);
            ld_d(reg_tmp, reg_param, PARAM_OFF(S_s));
            st_d(reg_tmp, rsp, stack_off_s_s);
            ld_d(reg_tmp, reg_param, PARAM_OFF(S_tail));
            st_d(reg_tmp, rsp, stack_off_s_tail);
        }
        if (is_c_padded()) {
            ld_d(reg_tmp, reg_param, PARAM_OFF(is_cblk_tail));
            st_d(reg_tmp, rsp, stack_off_is_cblk_tail);
        }

        if (bdesc_->is_fwd()) {
            ld_d(reg_tmp, reg_param, PARAM_OFF(shift));
            st_d(reg_tmp, rsp, stack_off_shift);
            ld_d(reg_tmp, reg_param, PARAM_OFF(var));
            add_d(reg_var, reg_tmp, zero);
        } else {
            ld_d(reg_tmp, reg_param, PARAM_OFF(diff_scale));
            st_d(reg_tmp, rsp, stack_off_diff_scale);
            ld_d(reg_tmp, reg_param, PARAM_OFF(diff_shift));
            st_d(reg_tmp, rsp, stack_off_diff_shift);
            ld_d(reg_tmp, reg_param, PARAM_OFF(soff_max));
            st_d(reg_tmp, rsp, stack_off_soff_max);
            ld_d(reg_tmp, reg_param, PARAM_OFF(var));
            add_d(reg_var, reg_tmp, zero);
        }
#undef PARAM_OFF
    }

    void prepare_tail_mask_lasx_common() {
        if (!is_c_padded()) return;

        //const int tail = bdesc_->C() % (int)(vlen / sizeof(float));
        tail_ = bdesc_->C() % (int)(vlen / sizeof(float));
        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};

        //std::cout << "tail: " << tail_ << std::endl;  //for DEBUG

        mov_imm(reg_tmp, reinterpret_cast<size_t>(&mask[8 - tail_]));
        xvld(vtail_mask, reg_tmp, 0);
    }

    void prepare_relu() {
        with_relu = bdesc_->is_fwd()
                ? bdesc_->with_relu_post_op() || bdesc_->fuse_norm_relu()
                : bdesc_->fuse_norm_relu();
        with_relu_inf_only = with_relu && bdesc_->is_fwd()
                && !(bdesc_->fuse_norm_relu() && bdesc_->is_training());

        vzero = bdesc_->is_fwd() ? vdiff_beta : vbeta;
        if (with_relu) {
            xvxor_v(vzero, vzero, vzero);
            if (!bdesc_->is_fwd() && isa == lasx) prepare_l_relu_mask_lasx();
        }
    }

    void prepare_l_relu_mask_lasx() {
        Label l_mask_after;
        b(l_mask_after);
        align(32);
        L(l_relu_mask_avx2); /* [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01] */
        for (int i = 0; i < 8; ++i)
            dd(1 << i);
        L(l_mask_after);
    }

    void fwd_process_relu_lasx(Vmm vdst, int offt, Vmm vstore_mask) {
        srli_d(reg_soff, reg_soff, bit_shift());
        xvfcmp_clt_s(vstore_mask, vzero, vdst);
        xvpickev_h(v_tmp0, vstore_mask, vstore_mask);
        xvpermi_d(v_tmp0, v_tmp0, 0xD8);
        xvpickev_b(v_tmp0, v_tmp0, v_tmp0);
        add_imm(X_TMP_0, reg_soff, offt / (1 << bit_shift()), X_TMP_1);
        add_d(X_TMP_0, X_TMP_0, reg_ws);
        xvstelm_d(v_tmp0, X_TMP_0, 0, 0);
        xvbitsel_v(vdst, vzero, vdst, vstore_mask);
        slli_d(reg_soff, reg_soff, bit_shift());
    }

    void bwd_process_relu_lasx(Vmm vdiff_dst, int offt, Vmm vstore_mask) {
        srli_d(reg_soff, reg_soff, bit_shift());
        add_imm(X_TMP_0, reg_soff, offt / (1 << bit_shift()), X_TMP_1);
        add_d(X_TMP_0, X_TMP_0, reg_ws);
        xvldrepl_d(vstore_mask, X_TMP_0, 0);  //load ziped mask
        vext2xv_w_b(vstore_mask, vstore_mask);
        xvseq_d(vstore_mask, vstore_mask, v_tmp0);
        xvbitsel_v(vdiff_dst, vzero, vdiff_dst, vstore_mask);
        slli_d(reg_soff, reg_soff, bit_shift());
    }

    #define FLAG_XVLD 1
    #define FLAG_XVST 0

    void uni_xvldst_spat_data(const XVReg &xvreg, const XReg &r1, const XReg &r2, int32_t off, bool ldFlag=FLAG_XVLD) {
        add_imm(X_TMP_0, r2, off, X_TMP_1);
        if(ldFlag == FLAG_XVLD) {
            xvldx(xvreg, r1, X_TMP_0);
        } else {
            xvstx(xvreg, r1, X_TMP_0);
        }
    }

    void uni_xvldst_tail_lasx_common(const XVReg &xvreg, const XReg &r1, const XReg &r2, bool ldFlag, Label &l_ret) {
        add_d(X_TMP_0, r1, r2);

        if(ldFlag == FLAG_XVLD) {
            xvld(xvreg, X_TMP_0, 0);  //load directly, as x64 load high mem to high xvreg directly using mask
            xvand_v(xvreg, xvreg, vtail_mask);
        } else {
            switch(tail_) {
                case 1: xvstelm_w(xvreg, X_TMP_0, 0, 0); break;
                case 2: xvstelm_d(xvreg, X_TMP_0, 0, 0); break;
                case 3: xvstelm_d(xvreg, X_TMP_0, 0, 0);
                        xvstelm_w(xvreg, X_TMP_0, 4*2, 2);
                        break;
                case 4: xvstelm_d(xvreg, X_TMP_0, 0, 0);
                        xvstelm_d(xvreg, X_TMP_0, 4*2, 1);
                        break;
                case 5: xvstelm_d(xvreg, X_TMP_0, 0, 0);
                        xvstelm_d(xvreg, X_TMP_0, 4*2, 1);
                        xvstelm_w(xvreg, X_TMP_0, 4*4, 4);
                        break;
                case 6: xvstelm_d(xvreg, X_TMP_0, 0, 0);
                        xvstelm_d(xvreg, X_TMP_0, 4*2, 1);
                        xvstelm_d(xvreg, X_TMP_0, 4*4, 2);
                        break;
                case 7: xvstelm_d(xvreg, X_TMP_0, 0, 0);
                        xvstelm_d(xvreg, X_TMP_0, 4*2, 1);
                        xvstelm_d(xvreg, X_TMP_0, 4*4, 2);
                        xvstelm_w(xvreg, X_TMP_0, 4*6, 6);
                        break;
                default: break;
            }
        }

        b(l_ret);
    }


    void uni_xvldst_maybe_tail(const XVReg &xvreg, const XReg &r1, bool ldFlag=FLAG_XVLD, const XReg &r2=XReg(0), int32_t off=0) {
        Label l_no_mask, l_ret;
        XReg r2_tmp = r2;
        if(0 != off)
        {
           r2_tmp = X_TMP_1;
           add_imm(r2_tmp, r2, off, X_TMP_0);
        }

        if (is_c_padded()) {
            ld_d(reg_tmp, rsp, stack_off_is_cblk_tail);
            beqz(reg_tmp, l_no_mask);

            add_imm(reg_tmp, reg_coff, vlen, X_TMP_0);
            blt(reg_tmp, reg_coff_max, l_no_mask);

            uni_xvldst_tail_lasx_common(xvreg, r1, r2_tmp, ldFlag, l_ret);
        }
        L(l_no_mask);

        if(ldFlag == FLAG_XVLD)
            xvldx(xvreg, r1, r2_tmp);
        else
            xvstx(xvreg, r1, r2_tmp);

        L(l_ret);
    }

    void barrier() {
        ld_d(reg_nnthr, rsp, stack_off_N_nthr);
        ld_d(reg_bar, rsp, stack_off_barrier);
        simple_barrier::generate(*this, reg_bar, reg_nnthr);
    }

    XReg xreg_addr(const XReg &base, const XReg &off = XReg(DUMMY_IDX),
            const int disp = 0) {
        XReg x_addr = base;
        uint32_t offIdx = off.getIdx();

        if (offIdx <= 31) {
            add_d(X_DEFAULT_ADDR, base, off);
            x_addr = X_DEFAULT_ADDR;
        }
        if (disp) {
            add_imm(X_DEFAULT_ADDR, x_addr, disp, X_TMP_0);
            x_addr = X_DEFAULT_ADDR;
        }

        return x_addr;
    }

    XReg mean_ptr(size_t offt = 0) {
        return xreg_addr(reg_mean, reg_coff, offt);
    }

    XReg var_ptr(size_t offt = 0) { return xreg_addr(reg_var, reg_coff, offt); }

    XReg diff_gamma_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_scale, reg_coff, offt);
    }

    XReg diff_beta_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_shift, reg_coff, offt);
    }

    XReg gamma_ptr(size_t offt = 0) {
        return xreg_addr(reg_scale, reg_coff, offt);
    }

    XReg beta_ptr(size_t offt = 0) {
        return xreg_addr(reg_shift, reg_coff, offt);
    }

    template <typename init_t, typename body_t, typename fini_t>
    void spat_loop(size_t len, size_t blocks, size_t regs, init_t init,
            body_t body, fini_t fini) {
        size_t factor = regs * blocks;
        size_t loop_unroll = len / factor * factor;
        size_t loop_tail = len - loop_unroll;
        size_t num_active_regs = (len < regs) ? len : regs;
        for (size_t i = 0; i < num_active_regs; i++)
            init(i);
        if (loop_unroll) {
            if (is_spatial_thr_) {
                ld_d(reg_ctr, rsp, stack_off_spat_size_loc);
                ld_d(X_TMP_0, rsp, stack_off_s_s);
                add_d(reg_soff, reg_soff, X_TMP_0);
            } else {
                mov_imm(reg_ctr, loop_unroll);
            }
            Label label;
            L(label);
            {
                for (size_t i = 0; i < factor; i++) {
                    size_t base_reg = i % regs;
                    body(base_reg, i);
                }

                add_imm(reg_soff, reg_soff, factor * spat_step, X_TMP_0);
                sub_imm(reg_ctr, reg_ctr, factor, X_TMP_0);
                bnez(reg_ctr, label);
            }
            if (is_spatial_thr_) {
                ld_d(X_TMP_0, rsp, stack_off_s_tail);
                add_d(reg_soff, reg_soff, X_TMP_0);
            }
        }

        for (size_t i = 0; i < loop_tail; i++) {
            size_t base_reg = i % regs;
            body(base_reg, i);
        }
        if (loop_tail) add_imm(reg_soff, reg_soff, loop_tail * spat_step, X_TMP_0);

        for (size_t i = 0; i < num_active_regs; i++)
            fini(i);
    }

    void mean_channels() {
        Label ch_label;
        L(ch_label);
        {
            xvldx(Vmm(0), reg_rbuf1, reg_coff);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg) xvxor_v(v, v, v);
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm v0 = Vmm(base_reg * 2 + 0);
                        Vmm v1 = Vmm(base_reg * 2 + 1);
                        size_t offt = i * vlen_spat_data_;
                        uni_xvldst_spat_data(v1, reg_src, reg_soff, offt, FLAG_XVLD);
                        xvfadd_s(v0, v0, v1);
                        add_imm(X_TMP_0, reg_soff, offt + t0_pf_offt, X_TMP_1);
                        preldx(0, reg_src, X_TMP_0);
                        add_imm(X_TMP_0, reg_soff, offt + t1_pf_offt, X_TMP_1);
                        preldx(0, reg_src, X_TMP_0);
                    },
                    [=](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg) xvfadd_s(b, b, v);
                    });
            xvstx(Vmm(0), reg_rbuf1, reg_coff);

            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            blt(reg_coff, reg_coff_max, ch_label);
        }
    }

    void mean_variance_nspc(
            const int num_ch_blks, int num_spat_pts, bool compute_mean) {

        auto mean_compute = [=](int num_ch_blks, int num_spat_pts) {
            int sp_idx = num_ch_blks;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                int offt = 0;
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    uni_xvldst_spat_data(Vmm(sp_idx), reg_src, reg_soff_nspc, offt, FLAG_XVLD);

                    xvfadd_s(Vmm(ch_idx), Vmm(ch_idx), Vmm(sp_idx++));

                    offt += vlen_spat_data_;
                }
                add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
            }
        };

        auto variance_compute = [=](int num_ch_blks, int num_spat_pts) {
            int sp_idx = num_ch_blks;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                int coff = 0, offt = 0;
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    uni_xvldst_maybe_tail(vmean, mean_ptr(coff));

                    uni_xvldst_spat_data(Vmm(sp_idx), reg_src, reg_soff_nspc, offt, FLAG_XVLD);

                    xvfsub_s(Vmm(30), vmean, Vmm(sp_idx++));
                    xvfmadd_s(Vmm(ch_idx), Vmm(30), Vmm(30), Vmm(ch_idx));

                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
            }
        };

        for (int idx = 0, offt = 0; idx < num_ch_blks; ++idx, offt += vlen)
        {
            add_imm(X_TMP_0, reg_coff, offt, X_TMP_1);
            xvldx(Vmm(idx), reg_rbuf1, X_TMP_0);
        }

        xor_(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);

        if (is_spatial_thr_) {
            ld_d(reg_ctr, rsp, stack_off_spat_size_loc);
            ld_d(X_TMP_0, rsp, stack_off_s_s);
            add_d(reg_soff_nspc, reg_soff_nspc, X_TMP_0);
            // TODO: need a better heuristic for num_spat_pts
            num_spat_pts = 1;
        } else {
            mov_imm(reg_ctr, spat_size);
            num_spat_pts = nstl::min((size_t)num_spat_pts, spat_size);
            // TODO: unroll by spatial
            if (spat_size % num_spat_pts != 0) num_spat_pts = 1;
        }

        Label spatial;
        L(spatial);
        {
            compute_mean ? mean_compute(num_ch_blks, num_spat_pts)
                         : variance_compute(num_ch_blks, num_spat_pts);
            sub_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
            bnez(reg_ctr, spatial);
        }

        for (int idx = 0, offt = 0; idx < num_ch_blks; ++idx, offt += vlen)
        {
            add_imm(X_TMP_0, reg_coff, offt, X_TMP_1);
            xvstx(Vmm(idx), reg_rbuf1, X_TMP_0);
        }
    }

    void forward_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            // Overwritten during mean and variance computation
            xvxor_v(vzero, vzero, vzero);
            xor_(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);

            if (is_spatial_thr_) {
                ld_d(reg_ctr, rsp, stack_off_spat_size_loc);
                ld_d(X_TMP_0, rsp, stack_off_s_s);
                add_d(reg_soff_nspc, reg_soff_nspc, X_TMP_0);
            } else {
                mov_imm(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            Label spatial;
            L(spatial);
            {
                int coff = 0, offt = 0;
                for (int idx = 0; idx < num_ch_blks; ++idx) {
                    uni_xvldst_maybe_tail(vmean, mean_ptr(coff));
                    uni_xvldst_maybe_tail(vsqrtvar, var_ptr(coff));
                    xvfadd_s(vsqrtvar, vsqrtvar, veps);
                    xvfsqrt_s(vsqrtvar, vsqrtvar);

                    if (bdesc_->use_scaleshift()) {
                        uni_xvldst_maybe_tail(vgamma, gamma_ptr(coff));
                        uni_xvldst_maybe_tail(vbeta, beta_ptr(coff));
                    }
                    if (bdesc_->use_scale()) {
                        uni_xvldst_maybe_tail(vgamma, gamma_ptr(coff));
                    }
                    if (bdesc_->use_shift()) {
                        uni_xvldst_maybe_tail(vbeta, beta_ptr(coff));
                    }

                    Vmm vscale
                            = (bdesc_->use_scaleshift() || bdesc_->use_scale())
                            ? vgamma
                            : vone;
                    Vmm vdiv = (bdesc_->use_scaleshift() || bdesc_->use_scale())
                            ? vgamma
                            : vsqrtvar;

                    xvfdiv_s(vdiv, vscale, vsqrtvar);
                    uni_xvldst_spat_data(Vmm(idx), reg_src, reg_soff_nspc, offt);
                    xvfsub_s(Vmm(idx), Vmm(idx), vmean);

                    if (bdesc_->use_scaleshift()
                            || (bdesc_->use_scale() && bdesc_->use_shift())) {
                        // --flags=S,CH
                        xvfmadd_s(Vmm(idx), vgamma, Vmm(idx), vbeta);
                    } else if (bdesc_->use_scale()) {
                        // --flags=C
                        xvfmul_s(Vmm(idx), Vmm(idx), vgamma);
                    } else if (bdesc_->use_shift()) {
                        // --flags=H
                        xvfmadd_s(Vmm(idx), vsqrtvar, Vmm(idx), vbeta);
                    } else {
                        xvfmul_s(Vmm(idx), Vmm(idx), vsqrtvar);
                    }

                    if (with_relu_inf_only) { // --attr=post_ops='relu'
                        xvfmax_s(Vmm(idx), Vmm(idx), vzero);
                    } else if (with_relu) { // --flags=R
                    }

                    if (stream_store_allowed) {
                        add_imm(X_TMP_0, reg_soff_nspc, offt, X_TMP_1);
                        xvstx(Vmm(idx), reg_dst, X_TMP_0);
                    } else {
                        uni_xvldst_spat_data(Vmm(idx), reg_dst, reg_soff_nspc, offt, FLAG_XVST);
                    }

                    addi_d(reg_ws, reg_ws, 2);
                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
                sub_imm(reg_ws, reg_ws, 2 * num_ch_blks, X_TMP_0);
                sub_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
                bnez(reg_ctr, spatial);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            sub_imm(X_TMP_0, reg_dst, vlen - 1, X_TMP_1);
            bnez(X_TMP_0, normal_store);
            compute(true);
            b(end_store);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // no NT store for BF16
        }
    }

    void compute_mean_variance_nspc(bool compute_mean = true) {
        xor_(reg_coff, reg_coff, reg_coff);
        add_d(reg_coff_max_fwd_copy, reg_coff_max, zero);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = is_bf16_ ? 3 : 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll, sp_idx = 1; ch_idx > 0;
                --ch_idx, ++sp_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                mov_imm(X_TMP_0, vlen * ch_blk_size);
                blt(reg_coff_max, X_TMP_0, ch_unroll_label[ch_idx - 1]);

                const int spat_blk_size = (1 << sp_idx);
                mean_variance_nspc(ch_blk_size, spat_blk_size, compute_mean);

                add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size, X_TMP_0);
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size, X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        add_d(reg_coff_max, reg_coff_max_fwd_copy, zero);

        if (is_bf16_) srli_d(reg_coff_max, reg_coff_max, 1);
        sub_d(reg_src, reg_src, reg_coff_max);
        if (is_bf16_) slli_d(reg_coff_max, reg_coff_max, 1);
    }

    void var_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_xvldst_maybe_tail(vmean, mean_ptr());
            xvldx(Vmm(0), reg_rbuf1, reg_coff);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg > 0) xvxor_v(v, v, v);
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm v = Vmm(3 * base_reg);
                        Vmm vtmp0 = Vmm(3 * base_reg + 1);
                        Vmm vtmp1 = Vmm(3 * base_reg + 2);
                        size_t offt = i * vlen_spat_data_;
                        uni_xvldst_spat_data(vtmp0, reg_src, reg_soff, offt);
                        xvfsub_s(vtmp1, vmean, vtmp0);
                        xvfmadd_s(v, vtmp1, vtmp1, v);

                        add_imm(X_TMP_0, reg_soff, offt + t0_pf_offt, X_TMP_1);
                        preldx(0, reg_src, X_TMP_0);
                        add_imm(X_TMP_0, reg_soff, offt + t1_pf_offt, X_TMP_1);
                        preldx(0, reg_src, X_TMP_0);
                    },
                    [=](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg) xvfadd_s(b, b, v);
                    });
            xvstx(Vmm(0), reg_rbuf1, reg_coff);
            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            blt(reg_coff, reg_coff_max, ch_label);
        }
    }

    void compute_mean_variance() {
        xvxor_v(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff, reg_coff);
        Label zero_rbuf;
        L(zero_rbuf);
        {
            xvstx(Vmm(0), reg_rbuf1, reg_coff);
            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            bne(reg_coff, reg_coff_max, zero_rbuf);
        }

        ld_d(reg_src, rsp, stack_off_src);

        xor_(reg_soff, reg_soff, reg_soff);
        Label mean_spatial;
        L(mean_spatial);
        {
            xor_(reg_coff, reg_coff, reg_coff);
            is_nspc_ ? compute_mean_variance_nspc() : mean_channels();

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
            } else {
                add_d(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            blt(reg_soff, reg_soff_max, mean_spatial);
        }

        if (is_nspc_) ld_d(reg_src, rsp, stack_off_src); // comeback

        Label no_mean_reduction;
        barrier();
        {
            ld_d(reg_tmp, rsp, stack_off_N_ithr);
            bnez(reg_tmp, no_mean_reduction);
            ld_d(reg_nnthr, rsp, stack_off_N_nthr);
            xor_(reg_coff, reg_coff, reg_coff);
            Label mean_reduction_channels;
            L(mean_reduction_channels);
            {
                add_d(reg_roff, reg_coff, zero);
                xvxor_v(Vmm(0), Vmm(0), Vmm(0));
                xvxor_v(Vmm(1), Vmm(1), Vmm(1));
                add_d(reg_ctr, reg_nnthr, zero);
                Label mean_reduction_thrs;
                L(mean_reduction_thrs);
                {
                    xvldx(v_tmp0, reg_rbuf1, reg_roff);
                    xvfadd_s(Vmm(1), Vmm(1), v_tmp0);
                    xvstx(Vmm(0), reg_rbuf1, reg_roff);
                    add_d(reg_roff, reg_roff, reg_coff_max);
                    sub_imm(reg_ctr, reg_ctr, 1, X_TMP_0);
                    bnez(reg_ctr, mean_reduction_thrs);
                }
                xvfdiv_s(Vmm(1), Vmm(1), vchan_size);
                uni_xvldst_maybe_tail(Vmm(1), mean_ptr(), FLAG_XVST);

                add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
                blt(reg_coff, reg_coff_max, mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        barrier();

        xor_(reg_soff, reg_soff, reg_soff);
        Label var_spatial;
        L(var_spatial);
        {
            xor_(reg_coff, reg_coff, reg_coff);
            is_nspc_ ? compute_mean_variance_nspc(false) : var_channels();

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
            } else {
                add_d(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            blt(reg_soff, reg_soff_max, var_spatial);
        }

        if (is_nspc_) ld_d(reg_src, rsp, stack_off_src); // comeback

        Label no_var_reduction;
        barrier();
        {
            ld_d(reg_tmp, rsp, stack_off_N_ithr);
            bnez(reg_tmp, no_var_reduction);

            ld_d(reg_nnthr, rsp, stack_off_N_nthr);
            xor_(reg_coff, reg_coff, reg_coff);
            Label var_reduction_channels;
            L(var_reduction_channels);
            {
                add_d(reg_roff, reg_coff, zero);
                xvxor_v(Vmm(1), Vmm(1), Vmm(1));
                add_d(reg_ctr, reg_nnthr, zero);
                Label var_reduction_thrs;
                L(var_reduction_thrs);
                { // TODO: unroll (?)
                    xvldx(v_tmp0, reg_rbuf1, reg_roff);
                    xvfadd_s(Vmm(1), Vmm(1), v_tmp0);
                    add_d(reg_roff, reg_roff, reg_coff_max);
                    sub_imm(reg_ctr, reg_ctr, 1, X_TMP_0);
                    bnez(reg_ctr, var_reduction_thrs);
                }
                xvfdiv_s(Vmm(1), Vmm(1), vchan_size);
                uni_xvldst_maybe_tail(Vmm(1), var_ptr(), FLAG_XVST);
                add_imm(reg_coff, reg_coff, vlen, X_TMP_0);

                bne(reg_coff, reg_coff_max, var_reduction_channels);
            }
        }
        L(no_var_reduction);
        barrier();
    }

    void forward_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_xvldst_maybe_tail(vmean, mean_ptr());
            uni_xvldst_maybe_tail(vsqrtvar, var_ptr());
            xvfadd_s(vsqrtvar, vsqrtvar, veps);
            xvfsqrt_s(vsqrtvar, vsqrtvar);

            if (bdesc_->use_scaleshift()) {
                uni_xvldst_maybe_tail(vgamma, gamma_ptr());
                uni_xvldst_maybe_tail(vbeta, beta_ptr());
            }
            if (bdesc_->use_scale()) {
                uni_xvldst_maybe_tail(vgamma, gamma_ptr());
            }
            if (bdesc_->use_shift()) {
                uni_xvldst_maybe_tail(vbeta, beta_ptr());
            }

            Vmm vscale = (bdesc_->use_scaleshift() || bdesc_->use_scale())
                    ? vgamma
                    : vone;
            Vmm vdiv = (bdesc_->use_scaleshift() || bdesc_->use_scale())
                    ? vgamma
                    : vsqrtvar;

            xvfdiv_s(vdiv, vscale, vsqrtvar);

            auto compute = [=](bool stream_store_allowed) {
                spat_loop(
                        spat_size, unroll_blocks, unroll_regs,
                        [](size_t base_reg) { UNUSED(base_reg); },
                        [=](size_t base_reg, size_t i) {
                            Vmm v = Vmm(base_reg);
                            size_t offt = i * vlen_spat_data_;
                            uni_xvldst_spat_data(v, reg_src, reg_soff, offt);
                            add_imm(X_TMP_0, reg_soff, offt + t0_pf_offt, X_TMP_1);
                            preldx(0, reg_src, X_TMP_0);
                            add_imm(X_TMP_0, reg_soff, offt + t1_pf_offt, X_TMP_1);
                            preldx(0, reg_src, X_TMP_0);
                            xvfsub_s(v, v, vmean);
                            if (bdesc_->use_scaleshift()
                                    || (bdesc_->use_scale()
                                            && bdesc_->use_shift())) {
                                // --flags=S,CH
                                xvfmadd_s(v, vgamma, v, vbeta);
                            } else if (bdesc_->use_scale()) {
                                // --flags=C
                                xvfmul_s(v, v, vgamma);
                            } else if (bdesc_->use_shift()) {
                                // --flags=H
                                xvfmadd_s(v, vsqrtvar, v, vbeta);
                            } else {
                                xvfmul_s(v, v, vsqrtvar);
                            }
                            if (with_relu_inf_only) {
                                xvfmax_s(v, v, vzero);
                            } else if (with_relu) {
                                fwd_process_relu_lasx(v, offt, Vmm(3));
                            }
                            if (stream_store_allowed) {
                                add_imm(X_TMP_0, reg_soff, offt, X_TMP_1);
                                xvstx(v, reg_dst, X_TMP_0);
                            } else {
                                uni_xvldst_spat_data(v, reg_dst, reg_soff, offt, FLAG_XVST);
                            }
                        },
                        [](size_t base_reg) { UNUSED(base_reg); });
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                sub_imm(X_TMP_0, reg_dst, vlen - 1, X_TMP_1);
                bnez(X_TMP_0, normal_store);
                compute(true);
                b(end_store);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            blt(reg_coff, reg_coff_max, ch_label);
        }
    }

    void forward_channels_nspc() {
        xor_(reg_coff, reg_coff, reg_coff);
        add_d(reg_coff_max_fwd_copy, reg_coff_max, zero);

        Label ch_unroll_label[5];
        const int max_ch_unroll = is_bf16_ ? 3 : 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                sub_imm(X_TMP_0, reg_coff_max, vlen * ch_blk_size, X_TMP_1);
                blt(X_TMP_0, zero, ch_unroll_label[ch_idx - 1]);

                forward_channels_nspc_compute(ch_blk_size);

                add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size, X_TMP_0);
                add_imm(reg_dst, reg_dst, vlen_spat_data_ * ch_blk_size, X_TMP_0);
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);
                add_imm(reg_ws, reg_ws, 2 * ch_blk_size, X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size, X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        add_d(reg_coff_max, reg_coff_max_fwd_copy, zero);

        if (is_bf16_) srli_d(reg_coff_max, reg_coff_max, 1);
        sub_d(reg_src, reg_src, reg_coff_max);
        sub_d(reg_dst, reg_dst, reg_coff_max);
        if (is_bf16_) slli_d(reg_coff_max, reg_coff_max, 1);

        srli_d(reg_coff_max, reg_coff_max, 5);
        sub_d(reg_ws, reg_ws, reg_coff_max);
        slli_d(reg_coff_max, reg_coff_max, 5);
    }

    void forward() {
        ld_d(reg_src, rsp, stack_off_src);
        ld_d(reg_dst, rsp, stack_off_dst);
        ld_d(reg_ws, rsp, stack_off_ws);
        ld_d(reg_shift, rsp, stack_off_shift);

        xor_(reg_soff, reg_soff, reg_soff);
        Label dst_spatial;
        L(dst_spatial);
        {
            xor_(reg_coff, reg_coff, reg_coff);

            is_nspc_ ? forward_channels_nspc() : forward_channels();

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                add_imm(reg_dst, reg_dst, mb_offt, X_TMP_0);
                add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                add_imm(reg_ws, reg_ws, ws_mb_offt, X_TMP_0);
            } else {
                add_d(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            blt(reg_soff, reg_soff_max, dst_spatial);
        }

        if (is_nspc_) {
            // comeback
            ld_d(reg_src, rsp, stack_off_src);
            ld_d(reg_dst, rsp, stack_off_dst);
            ld_d(reg_ws, rsp, stack_off_ws);
        }
    }

    void backward_sh_channels() {
        Label sh_channels;
        L(sh_channels);
        {
            uni_xvldst_maybe_tail(vmean, mean_ptr());
            xvldx(Vmm(0), reg_rbuf1, reg_coff);
            xvldx(Vmm(1), reg_rbuf2, reg_coff);
            spat_loop(
                    spat_size, 1, 1,
                    [=](size_t base_reg) {
                        if (base_reg > 0) {
                            for (int i = 0; i < 2; i++) {
                                Vmm v(base_reg * 5 + i);
                                xvxor_v(v, v, v);
                            }
                        }
                    },
                    [=](size_t base_reg, size_t i) {
                        Vmm o0 = Vmm(base_reg * 5 + 0);
                        Vmm o1 = Vmm(base_reg * 5 + 1);
                        Vmm t1 = Vmm(base_reg * 5 + 2);
                        Vmm t2 = Vmm(base_reg * 5 + 3);
                        Vmm t3 = Vmm(base_reg * 5 + 4);
                        size_t offt = i * vlen_spat_data_;
                        uni_xvldst_spat_data(
                                t1, reg_src, reg_soff, offt);
                        uni_xvldst_spat_data(
                                t2, reg_diff_dst, reg_soff, offt);
                        if (with_relu) {
                            if (isa == lasx)
                                bwd_process_relu_lasx(t2, offt, t3);
                            else
                                assert(false);
                        }
                        xvfsub_s(t3, vmean, t1);
                        xvfnmsub_s(o0, t3, t2, o0);
                        xvfadd_s(o1, o1, t2);

                        add_imm(X_TMP_0, reg_soff, offt + t0_pf_offt, X_TMP_1);
                        preldx(0, reg_diff_dst, X_TMP_0);
                        preldx(0, reg_src, X_TMP_0);
                        add_imm(X_TMP_0, reg_soff, offt + t1_pf_offt, X_TMP_1);
                        preldx(0, reg_diff_dst, X_TMP_0);
                        preldx(0, reg_src, X_TMP_0);
                    },
                    [=](size_t base_reg) {
                        Vmm b0 = Vmm(0);
                        Vmm b1 = Vmm(1);
                        if (base_reg) {
                            xvfadd_s(b0, b0, Vmm(base_reg * 5 + 0));
                            xvfadd_s(b1, b1, Vmm(base_reg * 5 + 1));
                        }
                    });

            xvstx(Vmm(0), reg_rbuf1, reg_coff);
            xvstx(Vmm(1), reg_rbuf2, reg_coff);

            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            blt(reg_coff, reg_coff_max, sh_channels);
        }
    }

    void backward_sh_channels_nspc_compute(const int num_ch_blks) {
        for (int idx = 0, offt = 0; idx < 2 * num_ch_blks; offt += vlen) {
            uni_xvld(Vmm(idx++), reg_rbuf1, reg_coff, offt);
            uni_xvld(Vmm(idx++), reg_rbuf2, reg_coff, offt);
        }

        xor_(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);

        if (is_spatial_thr_) {
            ld_d(reg_ctr, rsp, stack_off_spat_size_loc);
            ld_d(X_TMP_0, rsp, stack_off_s_s);
            add_d(reg_soff_nspc, reg_soff_nspc, X_TMP_0);

        } else {
            mov_imm(reg_ctr, spat_size);
        }

        // TODO: spatial blocking
        const int num_spat_pts = 1;

        Label spatial;
        L(spatial);
        {
            int coff = 0, offt = 0, sp_idx = 2 * num_ch_blks;
            for (int ch_idx = 0; ch_idx < 2 * num_ch_blks; ch_idx += 2) {
                uni_xvldst_maybe_tail(vmean, mean_ptr(coff));
                uni_xvldst_spat_data(Vmm(sp_idx), reg_src, reg_soff_nspc, offt);
                uni_xvldst_spat_data(Vmm(sp_idx + 1), reg_diff_dst, reg_soff_nspc, offt);

                if (with_relu) {
                    assert(false);
                }

                xvfsub_s(Vmm(sp_idx + 2), vmean, Vmm(sp_idx));
                xvfnmsub_s(Vmm(ch_idx), Vmm(sp_idx + 2), Vmm(sp_idx + 1), Vmm(ch_idx));
                xvfadd_s(Vmm(ch_idx + 1), Vmm(ch_idx + 1), Vmm(sp_idx + 1));

                coff += vlen;
                offt += vlen_spat_data_;
                sp_idx += 3;
            }
            add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
            sub_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
            bnez(reg_ctr, spatial);
        }

        for (int idx = 0, offt = 0; idx < 2 * num_ch_blks; offt += vlen) {
            uni_xvst(Vmm(idx++), reg_rbuf1, reg_coff, offt);
            uni_xvst(Vmm(idx++), reg_rbuf2, reg_coff, offt);
        }
    }

    void backward_sh_channels_nspc() {
        xor_(reg_coff, reg_coff, reg_coff);
        add_d(reg_coff_max_bwd_copy, reg_coff_max, zero);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = is_bf16_ ? 1 : 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1
                sub_imm(X_TMP_0, reg_coff_max, vlen * ch_blk_size, X_TMP_1);
                blt(X_TMP_0, zero, ch_unroll_label[ch_idx - 1]);

                backward_sh_channels_nspc_compute(ch_blk_size);

                add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size, X_TMP_0);
                add_imm(reg_diff_dst, reg_diff_dst, vlen_spat_data_ * ch_blk_size, X_TMP_0);

                // advance mean_ptr() and var_ptr()
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);
                add_imm(reg_ws, reg_ws, 2 * ch_blk_size,X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size, X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        add_d(reg_coff_max, reg_coff_max_bwd_copy, zero);
        ld_d(reg_diff_scale, rsp, stack_off_diff_scale);

        if (is_bf16_) srli_d(reg_coff_max, reg_coff_max, 1);
        sub_d(reg_src, reg_src, reg_coff_max);
        sub_d(reg_diff_dst, reg_diff_dst, reg_coff_max);
        if (is_bf16_) slli_d(reg_coff_max, reg_coff_max, 1);

        if (with_relu) {
            srli_d(reg_coff_max, reg_coff_max, 5);
            sub_d(reg_ws, reg_ws, reg_coff_max);
            slli_d(reg_coff_max, reg_coff_max, 5);
        }
    }

    void backward_diff_channels() {
        Label diff_channels;
        L(diff_channels);
        {
            uni_xvldst_maybe_tail(vmean, mean_ptr());
            uni_xvldst_maybe_tail(vsqrtvar, var_ptr());

            xvfadd_s(vsqrtvar, vsqrtvar, veps);
            xvfsqrt_s(vsqrtvar, vsqrtvar);
            xvfdiv_s(vsqrtvar, vone, vsqrtvar);
            if (bdesc_->use_scaleshift() || bdesc_->use_scale())
                uni_xvldst_maybe_tail(vgamma, gamma_ptr());

            uni_xvldst_maybe_tail(vdiff_gamma, diff_gamma_ptr());
            uni_xvldst_maybe_tail(vdiff_beta, diff_beta_ptr());

            xvfmul_s(vdiff_gamma, vdiff_gamma, vsqrtvar);
            xvfdiv_s(vdiff_beta, vdiff_beta, vchan_size);
            xvfdiv_s(vdiff_gamma, vdiff_gamma, vchan_size);

            auto compute = [=](bool stream_store_allowed) {
                spat_loop(
                        spat_size, unroll_blocks, unroll_regs,
                        [=](size_t base_reg) { UNUSED(base_reg); },
                        [=](size_t base_reg, size_t i) {
                            Vmm v(base_reg * 2 + 0);
                            Vmm t(base_reg * 2 + 1);
                            Vmm t1(base_reg * 2 + 2);
                            size_t offt = i * vlen_spat_data_;

                            uni_xvldst_spat_data(
                                    v, reg_diff_dst, reg_soff, offt);
                            if (with_relu) {
                                if (isa == lasx)
                                    bwd_process_relu_lasx(v, offt, t);
                                else
                                    assert(false);
                            }
                            if (!bdesc_->use_global_stats()) {
                                xvfsub_s(v, v, vdiff_beta);
                                uni_xvldst_spat_data(
                                        t, reg_src, reg_soff, offt);

                                xvfsub_s(t, vmean, t);
                                xvfmul_s(t, t, vdiff_gamma);
                                xvfadd_s(v, v, t);
                            }
                            xvfmul_s(v, v, vsqrtvar);
                            if (bdesc_->use_scaleshift()
                                    || bdesc_->use_scale()) {
                                xvfmul_s(v, v, vgamma);
                            }
                            if (stream_store_allowed) {
                                uni_xvst(v, reg_diff_src, reg_soff, offt);
                            } else {
                                uni_xvldst_spat_data(v, reg_diff_src, reg_soff, offt, FLAG_XVST);
                            }

                            add_imm(X_TMP_0, reg_soff, offt + t0_pf_offt, X_TMP_1);
                            preldx(0, reg_diff_dst, X_TMP_0);
                            preldx(0, reg_src, X_TMP_0);
                            add_imm(X_TMP_0, reg_soff, offt + t1_pf_offt, X_TMP_1);
                            preldx(0, reg_diff_dst, X_TMP_0);
                            preldx(0, reg_src, X_TMP_0);
                        },
                        [=](size_t base_reg) { UNUSED(base_reg); });
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                sub_imm(X_TMP_0, reg_diff_src, vlen - 1, X_TMP_1);
                bnez(X_TMP_0, normal_store);
                compute(true);
                b(end_store);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            blt(reg_coff, reg_coff_max, diff_channels);
        }
    }

    void backward_diff_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            xor_(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);
            if (is_spatial_thr_) {
                ld_d(reg_ctr, rsp, stack_off_spat_size_loc);
                ld_d(X_TMP_0, rsp, stack_off_s_s);
                add_d(reg_soff_nspc, reg_soff_nspc, X_TMP_0);
            } else {
                mov_imm(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            Label spatial;
            L(spatial);
            {
                int coff = 0, offt = 0;
                for (int idx = 0; idx < 3 * num_ch_blks; idx += 3) {
                    uni_xvldst_maybe_tail(vmean, mean_ptr(coff));
                    uni_xvldst_maybe_tail(vsqrtvar, var_ptr(coff));

                    xvfadd_s(vsqrtvar, vsqrtvar, veps);
                    xvfsqrt_s(vsqrtvar, vsqrtvar);
                    xvfdiv_s(vsqrtvar, vone, vsqrtvar);

                    if (bdesc_->use_scaleshift() || bdesc_->use_scale())
                        uni_xvldst_maybe_tail(vgamma, gamma_ptr(coff));

                    st_d(reg_ws, rsp, stack_off_ws_off_copy);
                    ld_d(reg_ws, rsp, stack_off_diff_scale);

                    uni_xvldst_maybe_tail(
                            vdiff_gamma, reg_ws, FLAG_XVLD, reg_coff, coff);
                    uni_xvldst_maybe_tail(vdiff_beta,
                            reg_diff_shift, FLAG_XVLD, reg_coff, coff);
                    ld_d(reg_ws, rsp, stack_off_ws_off_copy);

                    xvfmul_s(vdiff_gamma, vdiff_gamma, vsqrtvar);
                    xvfdiv_s(vdiff_beta, vdiff_beta, vchan_size);
                    xvfdiv_s(vdiff_gamma, vdiff_gamma, vchan_size);

                    uni_xvldst_spat_data(Vmm(idx),
                            reg_diff_dst, reg_soff_nspc, offt);

                    if (with_relu) {
                        assert(false);
                    }

                    if (!bdesc_->use_global_stats()) {
                        xvfsub_s(Vmm(idx), Vmm(idx), vdiff_beta);
                        uni_xvldst_spat_data(Vmm(idx + 1),
                                reg_src, reg_soff_nspc, offt);
                        xvfsub_s(Vmm(idx + 1), vmean, Vmm(idx + 1));
                        xvfmul_s(Vmm(idx + 1), Vmm(idx + 1), vdiff_gamma);
                        xvfadd_s(Vmm(idx), Vmm(idx), Vmm(idx + 1));
                    }

                    xvfmul_s(Vmm(idx), Vmm(idx), vsqrtvar);

                    if (bdesc_->use_scaleshift() || bdesc_->use_scale()) {
                        xvfmul_s(Vmm(idx), Vmm(idx), vgamma);
                    }

                    if (stream_store_allowed) {
                        uni_xvst(Vmm(idx), reg_diff_src, reg_soff_nspc, offt);
                    } else {
                        uni_xvldst_spat_data(Vmm(idx), reg_diff_src, reg_soff_nspc, offt, FLAG_XVST);
                    }

                    coff += vlen;
                    offt += vlen_spat_data_;
                }

                add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
                sub_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
                bnez(reg_ctr, spatial);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;

            sub_imm(X_TMP_0, reg_diff_src, vlen - 1, X_TMP_1);
            bnez(X_TMP_0, normal_store);
            compute(true);
            b(end_store);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // no NT store for BF16
        }
    }

    void backward_diff_channels_nspc() {
        xor_(reg_coff, reg_coff, reg_coff);
        add_d(reg_coff_max_bwd_copy, reg_coff_max, zero);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = is_bf16_ ? 1 : 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1

                sub_imm(X_TMP_0, reg_coff_max, vlen * ch_blk_size, X_TMP_1);
                blt(X_TMP_0, zero, ch_unroll_label[ch_idx - 1]);

                backward_diff_channels_nspc_compute(ch_blk_size);

                add_imm(reg_diff_dst, reg_diff_dst, vlen_spat_data_ * ch_blk_size, X_TMP_0);
                if (!bdesc_->use_global_stats())
                    add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size, X_TMP_0);
                add_imm(reg_diff_src, reg_diff_src, vlen_spat_data_ * ch_blk_size, X_TMP_0);

                // advance mean_ptr() and var_ptr()
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);
                add_imm(reg_ws, reg_ws, 2 * ch_blk_size, X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size, X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        add_d(reg_coff_max, reg_coff_max_bwd_copy, zero);
        ld_d(reg_diff_scale, rsp, stack_off_diff_scale);

        if (is_bf16_) srli_d(reg_coff_max, reg_coff_max, 1);
        sub_d(reg_diff_dst, reg_diff_dst, reg_coff_max);
        if (!bdesc_->use_global_stats()) sub_d(reg_src, reg_src, reg_coff_max);
        sub_d(reg_diff_src, reg_diff_src, reg_coff_max);
        if (is_bf16_) slli_d(reg_coff_max, reg_coff_max, 1);

        srli_d(reg_coff_max, reg_coff_max, 5);
        sub_d(reg_ws, reg_ws, reg_coff_max);
        slli_d(reg_coff_max, reg_coff_max, 5);
    }

    void backward() {
        xvxor_v(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff, reg_coff);
        Label zero_rbuf, sh_spatial;

        L(zero_rbuf);
        {
            uni_xvst(Vmm(0), reg_rbuf1, reg_coff, 0);
            uni_xvst(Vmm(0), reg_rbuf2, reg_coff, 0);

            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            bne(reg_coff, reg_coff_max, zero_rbuf);
        }

        ld_d(reg_src, rsp, stack_off_src);
        ld_d(reg_diff_dst, rsp, stack_off_diff_dst);
        if (with_relu) {
            assert(isa == lasx);
            ld_d(reg_ws, rsp, stack_off_ws);
        }

        xor_(reg_soff, reg_soff, reg_soff);
        L(sh_spatial);
        {
            xor_(reg_coff, reg_coff, reg_coff);
            is_nspc_ ? backward_sh_channels_nspc() : backward_sh_channels();

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                add_imm(reg_diff_dst, reg_diff_dst, mb_offt, X_TMP_0);
                add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                add_imm(reg_ws, reg_ws, ws_mb_offt, X_TMP_0);
            } else {
                add_d(reg_soff, reg_soff, reg_mb_stride_Bc);
            }
            blt(reg_soff, reg_soff_max, sh_spatial);
        }

        if (is_nspc_) {
            // comeback
            ld_d(reg_src, rsp, stack_off_src);
            ld_d(reg_diff_dst, rsp, stack_off_diff_dst);
        }

        ld_d(reg_diff_scale, rsp, stack_off_diff_scale);
        ld_d(reg_diff_shift, rsp, stack_off_diff_shift);

        Label no_sh_reduction;
        barrier();
        {
            ld_d(reg_tmp, rsp, stack_off_N_ithr);
            Label sh_reduction_channels;
            bnez(reg_tmp, no_sh_reduction);

            ld_d(reg_nnthr, rsp, stack_off_N_nthr);
            xor_(reg_coff, reg_coff, reg_coff);
            L(sh_reduction_channels);
            {
                add_d(reg_roff, reg_coff, zero);
                xvxor_v(Vmm(0), Vmm(0), Vmm(0));
                xvxor_v(Vmm(1), Vmm(1), Vmm(1));
                uni_xvldst_maybe_tail(vsqrtvar, var_ptr());

                xvfadd_s(vsqrtvar, vsqrtvar, veps);
                xvfsqrt_s(vsqrtvar, vsqrtvar);
                xvfdiv_s(vsqrtvar, vone, vsqrtvar);
                add_d(reg_ctr, reg_nnthr, zero);
                Label sh_reduction_thrs;
                L(sh_reduction_thrs);
                { // TODO: unroll (?)
                    xvldx(v_tmp0, reg_rbuf1, reg_roff);
                    xvfadd_s(Vmm(0), Vmm(0), v_tmp0);
                    xvldx(v_tmp0, reg_rbuf2, reg_roff);
                    xvfadd_s(Vmm(1), Vmm(1), v_tmp0);

                    add_d(reg_roff, reg_roff, reg_coff_max);
                    sub_imm(reg_ctr, reg_ctr, 1, X_TMP_0);
                    bnez(reg_ctr, sh_reduction_thrs);
                }
                xvfmul_s(Vmm(0), Vmm(0), vsqrtvar);
                uni_xvldst_maybe_tail(Vmm(0), diff_gamma_ptr(), FLAG_XVST);
                uni_xvldst_maybe_tail(Vmm(1), diff_beta_ptr(), FLAG_XVST);
                add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
                bne(reg_coff, reg_coff_max, sh_reduction_channels);
            }
        }
        L(no_sh_reduction);
        barrier();

        ld_d(reg_diff_src, rsp, stack_off_diff_src);
        if (with_relu) {
            assert(isa == lasx);
            ld_d(reg_ws, rsp, stack_off_ws);
        }

        xor_(reg_soff, reg_soff, reg_soff);
        Label diff_spatial;
        L(diff_spatial);
        {
            xor_(reg_coff, reg_coff, reg_coff);
            // diff_shift is shared with soff_max.
            ld_d(reg_diff_shift, rsp, stack_off_diff_shift);
            is_nspc_ ? backward_diff_channels_nspc() : backward_diff_channels();

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (!bdesc_->use_global_stats()) add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                add_imm(reg_diff_dst, reg_diff_dst, mb_offt, X_TMP_0);
                add_imm(reg_diff_src, reg_diff_src, mb_offt, X_TMP_0);
                add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                add_imm(reg_ws, reg_ws, ws_mb_offt, X_TMP_0);
            } else {
                add_d(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            // comeback soff_max. Shared with diff_shift.
            ld_d(reg_soff_max, rsp, stack_off_soff_max);
            blt(reg_soff, reg_soff_max, diff_spatial);
        }
        if (is_nspc_) {
            // comeback
            if (!bdesc_->use_global_stats())
                ld_d(reg_src, rsp, stack_off_src);

            ld_d(reg_diff_dst, rsp, stack_off_diff_dst);
            ld_d(reg_diff_src, rsp, stack_off_diff_src);
            if (with_relu) ld_d(reg_ws, rsp, stack_off_ws);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *bdesc) : bdesc_(bdesc) {
        static_assert(isa == lasx, "unsupported isa");

        const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(acc_data_t);
        is_bf16_ = bdesc_->desc()->data_desc.data_type == data_type::bf16;
        size_t dt_size
                = types::data_type_size(bdesc_->desc()->data_desc.data_type);
        const memory_desc_wrapper src_d(bdesc_->src_md());
        is_nspc_
                = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::ndhwc);
        is_spatial_thr_ = bnorm_utils::is_spatial_thr(
                bdesc_, is_nspc_, simd_w, dt_size);
        vlen_spat_data_ = vlen / (1 + is_bf16_); // 32B of BF16 -> 64B of FP32

        unroll_blocks = 1;
        unroll_regs = 1;
    }

    void generate() override {
        preamble();

        if (is_bf16_) {
            // init emulation of bfloat16 operations
        }

        prepare_tail_mask_lasx_common();

        compute_static_strides();
        sub_imm(rsp, rsp, (int)stack_size_required, X_TMP_0);
        load_common_params();
        prepare_relu();

        if (bdesc_->is_fwd()) {
            if (!bdesc_->stats_is_src()) { compute_mean_variance(); }
            forward();
        } else {
            backward();
        }
        addi_d(rsp, rsp, stack_size_required);
        postamble();
    }

    void operator()(const call_params_t *p) { jit_generator::operator()(p); }

    ~jit_bnorm_t() override { }
};
} // namespace

namespace bnorm_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
    driver_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc), ker_(bdesc_) {
        const dim_t C_PADDED = get_c_padded(bdesc_);

        const memory_desc_wrapper src_d(bdesc_->src_md());
        is_nspc_
                = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::ndhwc);

        dt_size_ = types::data_type_size(bdesc_->desc()->data_desc.data_type);
        size_t data_size = dt_size_ * bdesc_->MB() * C_PADDED * bdesc_->D()
                * bdesc_->H() * bdesc_->W();
        l3_size_ = platform::get_per_core_cache_size(3) * dnnl_get_max_threads()
                / 2; // XXX
        // TODO: cache balancing for nspc
        do_blocking_ = is_nspc_ ? false
                                : (data_size >= l3_size_ / 2 && l3_size_ > 0);
    }

    ~driver_t() = default;

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *bdesc) {
        dim_t C_PADDED = get_c_padded(bdesc);

        int sbuf_sz = use_tmp_stats(bdesc) * 2 * C_PADDED;
        int pbuf_sz = (use_tmp_diff_scale(bdesc) + use_tmp_diff_shift(bdesc))
                * C_PADDED;
        int rbuf_sz
                = (bdesc->is_fwd() ? 1 : 2) * C_PADDED * dnnl_get_max_threads();

        scratchpad.book<acc_data_t>(key_bnorm_tmp_stats, sbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_tmp_diff_ss, pbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_reduction, rbuf_sz);

        if (dnnl_thr_syncable()) {
            int n_barriers = C_PADDED / simd_w;
            scratchpad.book<barrier::ctx_64_t>(key_barrier, n_barriers);
        }
    }

    void exec(int ithr, int nthr, const void *src, void *diff_src, void *dst,
            const void *diff_dst, const acc_data_t *scale,
            acc_data_t *diff_scale, const acc_data_t *shift,
            acc_data_t *diff_shift, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto sbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_stats);
        auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);

        dim_t N = bdesc_->MB();
        dim_t C = bdesc_->C();
        dim_t C_PADDED = get_c_padded(bdesc_);
        dim_t D = bdesc_->D();
        dim_t H = bdesc_->H();
        dim_t W = bdesc_->W();
        dim_t SP = D * H * W;
        dim_t img_size = C_PADDED * D * H * W;
        const int vlen_spat_data = ker_.spat_step;

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = bdesc_->desc()->batch_norm_epsilon;
        p.one = 1.0f;
        p.spat_size = D * H * W;
        p.chan_size = 1.0f * N * p.spat_size;

        dim_t C_blks = C_PADDED / simd_w;

        int C_ithr {0}, C_nthr {0}, N_ithr {0}, N_nthr {0}, S_ithr {0},
                S_nthr {0};
        dim_t C_blk_s {0}, C_blk_e {0}, N_s {0}, N_e {0}, S_s {0}, S_e {0};

        dim_t C_blks_per_iter {1};
        int64_t iters {1};
        if (do_blocking_) {
            int num_tensors = bdesc_->is_fwd() ? 1 : 2;
            size_t working_set_size
                    = dt_size_ * (N * D * H * W * simd_w) * num_tensors;
            bnorm_utils::cache_balance(
                    working_set_size, C_blks, N, nthr, C_blks_per_iter, iters);
        }

        bool spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking_,
                true /* spatial_thr_allowed */, is_nspc_, ithr, nthr, N,
                do_blocking_ ? C_blks_per_iter : C_blks, SP,
                /* outputs */ C_ithr, C_nthr, C_blk_s, C_blk_e, N_ithr, N_nthr,
                N_s, N_e, S_ithr, S_nthr, S_s, S_e);

        int SP_N_ithr = N_ithr * S_nthr + S_ithr;
        int SP_N_nthr = N_nthr * S_nthr;
        assert(IMPLICATION(!dnnl_thr_syncable(), SP_N_nthr == 1));

        p.N_ithr = SP_N_ithr;
        p.N_nthr = SP_N_nthr;

        int last_iter_blks = C_blks - (iters - 1) * C_blks_per_iter;
        int global_C_blk_s;
        int global_barriers_per_iter = C_nthr;

        for (int64_t it = 0; it < iters; it++) {
            if (it == iters - 1 && iters > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking_,
                        spatial_thr_allowed, is_nspc_, ithr, nthr, N,
                        last_iter_blks, SP, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s, S_e);

                // Update call parameters for JIT, last iteration
                p.N_ithr = N_ithr * S_nthr + S_ithr;
                p.N_nthr = N_nthr * S_nthr;
            }

            global_C_blk_s = do_blocking_
                    ? (C_blk_s == -1) ? -1 : it * C_blks_per_iter + C_blk_s
                    : C_blk_s;

            int C_blks_thr = C_blk_e - C_blk_s;
            int N_thr = N_e - N_s;

            if (C_blks_thr == 0 || N_thr == 0) continue;

            size_t coff_base = global_C_blk_s * simd_w;
            size_t soff_base = is_nspc_
                    ? coff_base + N_s * img_size
                    : global_C_blk_s * p.spat_size * simd_w + N_s * img_size;
            size_t shift_off = use_tmp_diff_scale(bdesc_) ? bdesc_->C() : 0;

            p.spat_size_loc = S_e - S_s;
            p.S_s = S_s * vlen_spat_data;
            p.S_tail = (p.spat_size - S_e) * vlen_spat_data;
            p.coff_max = C_blks_thr * simd_w;
            const auto tmp_mean = use_tmp_stats(bdesc_) ? sbuf : mean;
            if (tmp_mean != nullptr) p.mean = tmp_mean + coff_base;
            const auto tmp_var = use_tmp_stats(bdesc_) ? sbuf + C_PADDED : var;
            if (tmp_var != nullptr) p.var = tmp_var + coff_base;
            if (scale != nullptr) p.scale = scale + coff_base;
            if (shift != nullptr) p.shift = shift + coff_base;
            const auto tmp_diff_scale
                    = use_tmp_diff_scale(bdesc_) ? pbuf : diff_scale;
            if (tmp_diff_scale != nullptr)
                p.diff_scale = tmp_diff_scale + coff_base;
            const auto tmp_diff_shift = use_tmp_diff_shift(bdesc_)
                    ? &pbuf[shift_off]
                    : diff_shift;
            if (tmp_diff_shift != nullptr)
                p.diff_shift = tmp_diff_shift + coff_base;

            p.soff_max = dt_size_ * N_thr * img_size;
            if (src != nullptr)
                p.src = (void *)((char *)src + soff_base * dt_size_);
            if (dst != nullptr)
                p.dst = (void *)((char *)dst + soff_base * dt_size_);
            if (diff_src != nullptr)
                p.diff_src = (void *)((char *)diff_src + soff_base * dt_size_);
            if (diff_dst != nullptr)
                p.diff_dst = (void *)((char *)diff_dst + soff_base * dt_size_);
            if (ws != nullptr) p.ws = ws + soff_base / 8;

            p.mb_stride_Bc = dt_size_ * (img_size - p.coff_max * p.spat_size);

            // use SP_N_nthr which is the same as p.N_nthr except maybe for
            // the last iteration.
            p.rbuf1 = rbuf
                    + ((it * C_blks_per_iter) * SP_N_nthr + C_blk_s * p.N_nthr
                              + p.N_ithr * C_blks_thr)
                            * simd_w;
            // rbuf1 and rbuf2 have to be disjoint
            p.rbuf2 = p.rbuf1 + C_PADDED * nthr;
            p.is_cblk_tail = (it * C_blks_per_iter + C_blk_e) * simd_w > C;

            size_t iter_bariers
                    = do_blocking_ ? it * global_barriers_per_iter : 0;
            p.barrier = barriers + C_ithr + iter_bariers;
            if (p.soff_max != 0 && p.coff_max != 0) ker_(&p);
        }
    }

    void init_barriers(const memory_tracking::grantor_t &scratchpad) {
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);
        if (barriers) {
            const int n_barriers = get_c_padded(bdesc_) / simd_w;
            for (int i = 0; i < n_barriers; ++i)
                barrier::ctx_init(&barriers[i]);
        }
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    enum {
        simd_w = cpu_isa_traits<isa>::vlen / sizeof(acc_data_t)
    };

    static bool use_tmp_stats(const batch_normalization_pd_t *bdesc) {
        return true && !bdesc->stats_is_src()
                && bdesc->desc()->prop_kind == prop_kind::forward_inference;
    }

    static bool use_tmp_diff_scale(const batch_normalization_pd_t *bdesc) {
        return false
                || (bdesc->is_bwd() && !bdesc->use_scaleshift()
                        && !bdesc->use_scale())
                || bdesc->desc()->prop_kind == prop_kind::backward_data;
    }

    static bool use_tmp_diff_shift(const batch_normalization_pd_t *bdesc) {
        return false
                || (bdesc->is_bwd() && !bdesc->use_scaleshift()
                        && !bdesc->use_shift())
                || bdesc->desc()->prop_kind == prop_kind::backward_data;
    }

    static dim_t get_c_padded(const batch_normalization_pd_t *bdesc) {
        return bdesc->src_md()->padded_dims[1];
    }

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_t<isa> ker_;
    bool do_blocking_;
    bool is_nspc_;
    size_t l3_size_;
    size_t dt_size_;
};
} // namespace bnorm_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = true
            /* the algorithm requires barriers for best performance so for TBB we use
             * jit_uni_tbb_batch_normalization instead */
            && dnnl_thr_syncable() && mayiuse(isa) && is_fwd()
            && !has_zero_dim_memory() && one_of(ndims(), 4, 5)
            && one_of(src_md()->data_type, f32, bf16)
            && IMPLICATION(src_md()->data_type == bf16, false)
            && check_scale_shift_data_type()
            && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    if (!src_d.matches_one_of_tag(nChw8c, nCdhw8c))
        return status::unimplemented;

    const bool isa_supports_lasx = (isa == lasx);
    if (is_training() && fuse_norm_relu()) {
        if (!isa_supports_lasx) return status::unimplemented;
        init_default_ws(1);
    }

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C()
            && !isa_supports_lasx)
        return status::unimplemented;

    // Only IC % 16 == 0 is supported for now
    if (src_d.matches_one_of_tag(nhwc, ndhwc)
            && src_d.padded_dims()[1] % 16 != 0) {
        return status::unimplemented;
    }

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::jit_uni_batch_normalization_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_, new bnorm_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {

    const memory_desc_wrapper ss_d(pd()->weights_md());

    const auto use_ss = pd()->use_scaleshift();
    const auto use_sc = pd()->use_scale();
    const auto use_sh = pd()->use_shift();

    const size_t shift_off
            = use_ss && !ss_d.has_zero_dim() ? ss_d.off(1, 0) : 0;

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto scale = CTX_IN_MEM(
            const acc_data_t *, use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT);
    auto shift = use_sh ? CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SHIFT)
                        : use_ss ? &CTX_IN_MEM(const acc_data_t *,
                                  DNNL_ARG_SCALE_SHIFT)[shift_off]
                                 : nullptr;

    auto mean = pd()->stats_is_src() ? const_cast<acc_data_t *>(
                        CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN))
                                     : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_MEAN);
    auto var = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(
                    CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_VARIANCE);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->init_barriers(scratchpad);

    parallel(0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, nullptr, dst, nullptr, scale,
                nullptr, shift, nullptr, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::~jit_uni_batch_normalization_fwd_t() {
    delete bnorm_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = true
            /* the algorithm requires barriers for best performance so for TBB we use
             * jit_uni_tbb_batch_normalization instead */
            && dnnl_thr_syncable() && mayiuse(isa) && is_bwd()
            && !has_zero_dim_memory() && one_of(ndims(), 4, 5)
            && set_default_formats_common()
            && one_of(true,
                    everyone_is(
                            f32, src_md()->data_type, diff_src_md()->data_type),
                    everyone_is(bf16, src_md()->data_type,
                            diff_src_md()->data_type))
            && IMPLICATION(src_md()->data_type == bf16, false)
            && check_scale_shift_data_type() && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper diff_src_d(diff_src_md());

    format_tag_t src_tag, diff_src_tag;
    src_tag = src_d.matches_one_of_tag(nChw8c, nCdhw8c);
    diff_src_tag = diff_src_d.matches_one_of_tag(nChw8c, nCdhw8c);

    ok = (src_tag != format_tag::undef && diff_src_tag != format_tag::undef
            && src_tag == diff_src_tag);
    if (!ok) return status::unimplemented;

    const bool isa_supports_lasx = (isa == lasx);
    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C()
            && !isa_supports_lasx)
        return status::unimplemented;

    // Only IC % 16 == 0 is supported for now
    if (src_d.matches_one_of_tag(nhwc, ndhwc)
            && src_d.padded_dims()[1] % 16 != 0) {
        return status::unimplemented;
    }

    if (fuse_norm_relu()) {
        if (!isa_supports_lasx) return status::unimplemented;
        init_default_ws(1);
        if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
    }

    /* TODO: extra checks required */

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::jit_uni_batch_normalization_bwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_, new bnorm_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper diff_ss_d(pd()->diff_weights_md());

    const auto use_ss = pd()->use_scaleshift();
    const auto use_sc = pd()->use_scale();
    const auto use_sh = pd()->use_shift();

    const size_t diff_shift_off
            = use_ss && !diff_ss_d.has_zero_dim() ? diff_ss_d.off(1, 0) : 0;

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto var = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto scale = CTX_IN_MEM(
            const acc_data_t *, use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
    auto diff_scale = CTX_OUT_MEM(acc_data_t *,
            use_sc ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT);
    auto diff_shift = use_sh ? CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SHIFT)
                             : use_ss ? &diff_scale[diff_shift_off] : nullptr;

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->init_barriers(scratchpad);

    parallel(0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, diff_src, nullptr, diff_dst, scale,
                diff_scale, nullptr, diff_shift, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::~jit_uni_batch_normalization_bwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_fwd_t<lasx>;
template struct jit_uni_batch_normalization_bwd_t<lasx>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
