/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/jit_uni_batch_normalization_s8.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

namespace {

using namespace Xbyak_loongarch64;

using data_t = int8_t;

struct call_params_t {
    // keep int sizes at 8 bytes -- jit code expects this
    size_t channel_offt_count, spat_offt_count;
    float eps;
    const float *scale_shift, *mean, *var;
    const data_t *src, *dst;
};

template <cpu_isa_t isa>
struct jit_bnorm_base_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    const int vlen = cpu_isa_traits<isa>::vlen;

    const batch_normalization_pd_t *pd_;

    XReg reg_param = abi_param1;
    XReg reg_scale_shift = a4;
    XReg reg_mean = a5;

    XReg reg_channel_offt_count = a6;
    XReg reg_spat_offt = a7;
    XReg reg_spat_offt_count = t3;
    XReg reg_tmp = t0;
    XReg reg_src = a1;
    XReg reg_dst = a2;
    XReg reg_var = a3;
    XReg reg_channel_offt_1byte = t1;
    XReg reg_channel_offt_4byte = t2;

    XVReg vzero = xr29;
    XVReg vone = xr30;
    XVReg veps = xr31;
    XVReg z_tmp0 = xr25;
    XVReg z_tmp1 = xr26;

    size_t c_in_xmm_ = 8;
    size_t chan_data_offt_;
    size_t num_c16_blocks_;
    size_t c_tail_;
    bool with_relu_;

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

    void compute_predefined_variables() {
        chan_data_offt_ = pd_->C() * sizeof(float);
        num_c16_blocks_ = pd_->C() / c_in_xmm_;
        c_tail_ = pd_->C() % c_in_xmm_;
        with_relu_ = (pd_->with_relu_post_op() || pd_->fuse_norm_relu())
                && pd_->is_fwd();
    }

    void load_common_params() {
        mov_imm(reg_tmp, float2int(1.0f));
        xvreplgr2vr_w(vone, reg_tmp);

#define PARAM_OFF(x) offsetof(call_params_t, x)
#define PARAM_OFF_DIFF(x, y) \
    (static_cast<int32_t>(PARAM_OFF(x)) - static_cast<int32_t>(PARAM_OFF(y)))
#define LDR_PARAM(r, x, y) \
    assert(-256 <= PARAM_OFF_DIFF(x, y) && PARAM_OFF_DIFF(x, y) <= 255); \
    addi_d(X_DEFAULT_ADDR, X_DEFAULT_ADDR, PARAM_OFF_DIFF(x, y)); \
    ld_d(r, X_DEFAULT_ADDR, 0)

        add_d(X_DEFAULT_ADDR, reg_param, zero);

	    addi_d(X_DEFAULT_ADDR, X_DEFAULT_ADDR, PARAM_OFF(eps));
        ld_w(W_TMP_0, X_DEFAULT_ADDR, 0);
        xvreplgr2vr_w(veps, W_TMP_0);
        xvxor_v(vzero, vzero, vzero);

        LDR_PARAM(reg_channel_offt_count, channel_offt_count, eps);
        LDR_PARAM(reg_spat_offt_count, spat_offt_count, channel_offt_count);
        LDR_PARAM(reg_src, src, spat_offt_count);
        LDR_PARAM(reg_dst, dst, src);
        LDR_PARAM(reg_mean, mean, dst);
        LDR_PARAM(reg_scale_shift, scale_shift, mean);
        LDR_PARAM(reg_var, var, scale_shift);

#undef PARAM_OFF
#undef PARAM_OFF_DIFF
#undef LDR_PARAM
    }

    XReg mean_ptr(size_t offt = 0) {
        return xreg_addr(reg_mean, reg_channel_offt_4byte, offt);
    }

    XReg var_ptr(size_t offt = 0) {
        return xreg_addr(reg_var, reg_channel_offt_4byte, offt);
    }

    XReg scale_ptr(size_t offt = 0) {
        return xreg_addr(reg_scale_shift, reg_channel_offt_4byte, offt);
    }

    XReg shift_ptr(size_t offt = 0) {
        return xreg_addr(reg_scale_shift, reg_channel_offt_4byte,
                offt + chan_data_offt_);
    }

    XReg src_ptr(size_t offt = 0) {
        return xreg_addr(reg_src, reg_spat_offt, offt);
    }

    XReg dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_dst, reg_spat_offt, offt);
    }

    virtual void prepare_tail_mask() {}
    virtual void load_mean_and_var(const XVReg &vmean, const XVReg &vsqrtvar,
            size_t offt, bool need_tail) {}
    virtual void load_scale_and_shift(const XVReg &vscale, const XVReg &vshift,
            size_t offt, bool need_tail) {}
    virtual void compute_dst(bool need_tail) {}

    // Precomputes vscale and vshift for following
    // `vdst = vscale * vsrc + vshift`
    void compute_vscaleshift(const XVReg &vscale, const XVReg &vshift,
            const XVReg &vmean, const XVReg &vsqrtvar, size_t offt,
            bool need_tail) {
        load_mean_and_var(vmean, vsqrtvar, offt, need_tail);
        xvfadd_s(vsqrtvar, vsqrtvar, veps);
        xvfsqrt_s(vsqrtvar, vsqrtvar);

        if (pd_->use_scaleshift()) {
            load_scale_and_shift(vscale, vshift, offt, need_tail);
            xvfdiv_s(vscale, vscale, vsqrtvar);
            xvfnmsub_s(vshift, vmean, vscale, vshift);
        } else {
            xvfdiv_s(vscale, vone, vsqrtvar);
            xvfmul_s(vmean, vmean, vscale);
            xvfsub_s(vshift, vzero, vmean); //TODO: xvfnmsub_s
        }
    }

    void forward() {
        xor_(reg_channel_offt_1byte, reg_channel_offt_1byte, reg_channel_offt_1byte);
        xor_(reg_channel_offt_4byte, reg_channel_offt_4byte, reg_channel_offt_4byte);
        mov_imm(reg_tmp, sizeof(data_t) * c_in_xmm_);

        if (num_c16_blocks_) compute_dst(false);
        if (c_tail_) compute_dst(true);
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        preamble();

        compute_predefined_variables();
        load_common_params();
        //prepare_tail_mask();
        forward();
        postamble();
    }

    jit_bnorm_base_t(const batch_normalization_pd_t *pd) : pd_(pd) {}
};

template <cpu_isa_t isa>
struct jit_bnorm_t;

template <>
struct jit_bnorm_t<lasx> : public jit_bnorm_base_t<lasx> {

    void load_mean_and_var(const XVReg &vmean, const XVReg &vsqrtvar, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            xvld(vmean, mean_ptr(offt), 0);
            xvld(vsqrtvar, var_ptr(offt), 0);
        } else {
            xvld(vmean, mean_ptr(offt), 0);
            xvld(vsqrtvar, var_ptr(offt), 0);
        }
    }

    void load_scale_and_shift(const XVReg &vscale, const XVReg &vshift,
            size_t offt, bool need_tail) override {
        if (need_tail) {
            xvld(vscale, scale_ptr(offt), 0);
            xvld(vshift, shift_ptr(offt), 0);
        } else {
            xvld(vscale, scale_ptr(offt), 0);
            xvld(vshift, shift_ptr(offt), 0);
        }
    }

    void compute_dst(bool need_tail = false) override {
        Label c_loop;
        L(c_loop);
        {
            XVReg v = XVReg(0);
            XVReg vscale = XVReg(1);
            XVReg vshift = XVReg(2);
            XVReg vmean = XVReg(3);
            XVReg vsqrtvar = XVReg(4);

            // compute single vscale and vshift vectors...
            compute_vscaleshift(vscale, vshift, vmean, vsqrtvar, 0, need_tail);

            // ... then process all spatial loop with it and move to the
            // next channel chunk
            add_d(reg_spat_offt, reg_channel_offt_1byte, zero);
            Label mb_sp_loop;
            L(mb_sp_loop);
            {
                if (need_tail) {
                    if (c_tail_ != 0) {
                        xvld(z_tmp0, src_ptr(), 0);
                    }

		            vext2xv_w_b(v, z_tmp0);
                } else {
                    xvldrepl_d(z_tmp0, src_ptr(), 0);
                    vext2xv_w_b(v, z_tmp0);
                }

                xvffint_s_w(v, v);
                xvfmadd_s(v, v, vscale, vshift);
                if (with_relu_) xvfmax_s(v, v, vzero);

                xvfrintrne_s(v, v);
                xvftintrz_w_s(v, v);
                if (need_tail) {
                    xvor_v(z_tmp0, v, v);

                    addi_w(X_TMP_0, zero, 127);
                    xvreplgr2vr_w(z_tmp1, X_TMP_0);
                    xvmin_w(z_tmp0, z_tmp0, z_tmp1);
                    addi_w(X_TMP_0, zero, -128);
                    xvreplgr2vr_w(z_tmp1, X_TMP_0);
                    xvmax_w(z_tmp0, z_tmp0, z_tmp1);
                    xvpickev_h(z_tmp0, z_tmp0, z_tmp0);
                    xvpermi_d(z_tmp0, z_tmp0, 0xD8);
                    xvpickev_b(v, z_tmp0, z_tmp0);

                    if (c_tail_ != 0) {
                        switch(c_tail_) {
                            case 1: xvstelm_b(v, dst_ptr(), 0, 0); break;
                            case 2: xvstelm_h(v, dst_ptr(), 0, 0); break;
                            case 3: xvstelm_h(v, dst_ptr(), 0, 0);
                                    xvstelm_b(v, dst_ptr(2), 0, 2);
                                break;
                            case 4: xvstelm_w(v, dst_ptr(), 0, 0); break;
                            case 5: xvstelm_w(v, dst_ptr(), 0, 0);
                                    xvstelm_b(v, dst_ptr(4), 0, 4);
                                break;
                            case 6: xvstelm_w(v, dst_ptr(), 0, 0);
                                    xvstelm_h(v, dst_ptr(4), 0, 2);
                                break;
                            case 7: xvstelm_w(v, dst_ptr(), 0, 0);
                                    xvstelm_h(v, dst_ptr(4), 0, 2);
                                    xvstelm_b(v, dst_ptr(6), 0, 6);
                                break;
                            case 8: xvstelm_d(v, dst_ptr(), 0, 0); break;
                        }
                    }
                } else {
                    xvor_v(z_tmp0, v, v);
                    addi_w(X_TMP_0, zero, 127);
                    xvreplgr2vr_w(z_tmp1, X_TMP_0);
                    xvmin_w(z_tmp0, z_tmp0, z_tmp1);
                    addi_w(X_TMP_0, zero, -128);
                    xvreplgr2vr_w(z_tmp1, X_TMP_0);
                    xvmax_w(z_tmp0, z_tmp0, z_tmp1);
                    xvpickev_h(z_tmp0, z_tmp0, z_tmp0);
                    xvpermi_d(z_tmp0, z_tmp0, 0xD8);
                    xvpickev_b(z_tmp0, z_tmp0, z_tmp0);
                    xvstelm_d(z_tmp0, dst_ptr(), 0, 0);
                }

                add_d(reg_spat_offt, reg_spat_offt, reg_channel_offt_count);
                blt(reg_spat_offt, reg_spat_offt_count, mb_sp_loop);
            }

            // reg_tmp checks c_in_xmm_ channels ahead for further tail process
            add_imm(reg_tmp, reg_tmp, sizeof(data_t) * c_in_xmm_, X_TMP_0);
            add_imm(reg_channel_offt_1byte, reg_channel_offt_1byte,
                    sizeof(data_t) * c_in_xmm_, X_TMP_0);
            add_imm(reg_channel_offt_4byte, reg_channel_offt_4byte,
                    sizeof(float) * c_in_xmm_, X_TMP_0);
            bge(reg_channel_offt_count, reg_tmp, c_loop);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *pd)
        : jit_bnorm_base_t<lasx>(pd) {}
};

} // namespace

namespace bnorm_s8_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
    driver_t(const batch_normalization_pd_t *pd) : pd_(pd), ker_(pd_) {}
    ~driver_t() = default;

    // TODO: for problems where thread pieces don't fit L2 cache, add spatial
    // re-balance using less pieces.
    void exec(int ithr, int nthr, const data_t *src, data_t *dst,
            const float *scale_shift, const float *mean, const float *var) {
        dim_t N = pd_->MB();
        dim_t C = pd_->C();
        dim_t D = pd_->D();
        dim_t H = pd_->H();
        dim_t W = pd_->W();
        dim_t SP = D * H * W;

        call_params_t p;

        p.eps = pd_->desc()->batch_norm_epsilon;

        p.scale_shift = scale_shift;
        p.mean = mean;
        p.var = var;

        dim_t work_amount {N * SP}, start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        p.channel_offt_count = C;
        p.spat_offt_count = (end - start) * p.channel_offt_count;
        p.src = src + start * p.channel_offt_count;
        p.dst = dst + start * p.channel_offt_count;

        if (p.spat_offt_count != 0) ker_(&p);
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    const batch_normalization_pd_t *pd_;

    jit_bnorm_t<isa> ker_;
};

} // namespace bnorm_s8_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::pd_t::init(
        engine_t *engine) {
    auto desired_fmt_tag = (ndims() == 4) ? nhwc : ndhwc;

    bool ok = true && mayiuse(isa) && is_fwd() && !has_zero_dim_memory()
            && one_of(ndims(), 4, 5) && stats_is_src()
            && src_md()->data_type == s8 && check_scale_shift_data_type()
            && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
            /* separate scale and shift are not supported */
            && !use_scale() && !use_shift()
            && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<isa>::jit_uni_batch_normalization_s8_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_s8_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto scale_shift = CTX_IN_MEM(const float *, DNNL_ARG_SCALE_SHIFT);
    auto mean = const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN));
    auto var
            = const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE));
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);

    // do sequential if the problem is less than one 4K memory page
    const bool force_sequential
            = pd()->MB() * pd()->C() * pd()->D() * pd()->H() * pd()->W()
            <= 4096;

    parallel(force_sequential ? 1 : 0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, dst, scale_shift, mean, var);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<
        isa>::~jit_uni_batch_normalization_s8_fwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_s8_fwd_t<lasx>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
