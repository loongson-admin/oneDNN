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
#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/loongarch64/jit_uni_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

namespace {

using namespace Xbyak_loongarch64;

template <cpu_isa_t isa>
struct jit_softmax_base_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const void *src, *dst, *diff_dst; // src dubs as diff_src
        size_t spat_offt_count;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_t)

    // cpu specific part
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const int vlen = cpu_isa_traits<isa>::vlen;

    const softmax_pd_t *pd_;
    const memory_desc_wrapper data_d_;

    virtual void operator()(const call_params_t *p) = 0;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> log_injector_;

    XReg reg_param = abi_param1;

    XReg reg_exp_injector_table = a2;
    XReg reg_log_injector_table = a3;
    XReg reg_src = t0;
    XReg reg_diff_src = reg_src;
    XReg reg_dst = t1;
    XReg reg_diff_dst = t2;
    XReg reg_spat_offt = t3;
    XReg reg_spat_offt_count = t4;
    XReg reg_reverse_spat_offt = t5;
    XReg reg_tmp = t6;

    XVReg injector_mask = xr29;

    Vmm vtmp = Vmm(31); // assigned at placed where used
    Vmm tail_vmask = Vmm(0);
    Vmm vneg_flt_max = Vmm(12);
    Vmm vone = Vmm(13);
    Vmm vsum = Vmm(14);
    Vmm vmax = Vmm(15);
    Vmm vsbr = vsum; // must be not equal to vmax
    Vmm vzeropad = Vmm(11);

    bool is_bf16_ = false;
    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();
    bool axis_is_blocked_;

    size_t data_type_size_ = 0;
    size_t simd_w_ = 0;
    size_t unroll_regs_ = 4;

    size_t axis_simd_full_;
    size_t axis_simd_tail_;
    size_t n_loops_;
    size_t loop_tail_;
    size_t axis_stride_;

    void compute_predefined_variables() {
        axis_simd_full_ = pd_->axis_size() / simd_w_;
        axis_simd_tail_ = pd_->axis_size() % simd_w_;
        n_loops_ = axis_simd_full_ / unroll_regs_;
        loop_tail_ = axis_simd_full_ - n_loops_ * unroll_regs_;
        axis_stride_ = compute_axis_stride();
        axis_is_blocked_ = (pd_->src_md(0)->padded_dims[pd_->axis()]
                != pd_->src_md(0)->dims[pd_->axis()]);
    }

    size_t compute_axis_stride() {
        const auto &bd = data_d_.blocking_desc();

        if (bd.inner_nblks) return data_type_size_ * bd.strides[pd_->axis()];
        return is_bf16_ ? vlen / 2 : vlen;
    }

    void load_common_params() {
        mov_imm(reg_tmp, float2int(1.0f));
        xvreplgr2vr_w(vone, reg_tmp);
        mov_imm(reg_tmp, float2int(-FLT_MAX));
        xvreplgr2vr_w(vneg_flt_max, reg_tmp);

#define PARAM_OFF(x) offsetof(call_params_t, x)
        ld_d(reg_spat_offt_count, reg_param, PARAM_OFF(spat_offt_count));
        ld_d(reg_dst, reg_param, PARAM_OFF(dst));

        if (pd_->is_fwd())
            ld_d(reg_src, reg_param, PARAM_OFF(src));
        else {
            ld_d(reg_diff_src, reg_param, PARAM_OFF(src));
            ld_d(reg_diff_dst, reg_param, PARAM_OFF(diff_dst));
        }
#undef PARAM_OFF
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

    XReg diff_src_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_src, reg_spat_offt, offt);
    }

    XReg src_ptr(size_t offt = 0) {
        return xreg_addr(reg_src, reg_spat_offt, offt);
    }

    XReg dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_dst, reg_spat_offt, offt);
    }

    XReg diff_dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_dst, reg_spat_offt, offt);
    }


    enum class op_t : unsigned { max, sum };

    void perform_op(Vmm v, Vmm vtmp, op_t op) {
        if (op == op_t::max)
            xvfmax_s(v, v, vtmp);
        else if (op == op_t::sum)
            xvfadd_s(v, v, vtmp);
    }

    template <typename body_t>
    void axis_loop(body_t body) {
        Label main_loop, tail_loop, tail_axis;

        // reverse_spat_offt to dispatch between labels
        add_d(reg_reverse_spat_offt, reg_spat_offt_count, zero);
        xor_(reg_spat_offt, reg_spat_offt, reg_spat_offt);

        L(main_loop);
        {
            if (n_loops_) {
                mov_imm(X_TMP_0, unroll_regs_ * axis_stride_);
                blt(reg_reverse_spat_offt, X_TMP_0, tail_loop);

                body(unroll_regs_, false);
                add_imm(reg_reverse_spat_offt, reg_reverse_spat_offt, -1 * unroll_regs_ * axis_stride_, X_TMP_0);
                add_imm(reg_spat_offt, reg_spat_offt, unroll_regs_ * axis_stride_, X_TMP_0);
                b(main_loop);
            }
        }

        L(tail_loop);
        {
            if (loop_tail_) {
                body(loop_tail_, false);
                add_imm(reg_spat_offt, reg_spat_offt, loop_tail_ * axis_stride_, X_TMP_0);
            }
        }

        L(tail_axis);
        {
            if (axis_simd_tail_) { body(1, true); }
        }
    }

    virtual void prepare_tail_mask() = 0;
    virtual void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) = 0;
    virtual void accumulate_vmax() = 0;
    virtual void accumulate_vsum() = 0;
    virtual void compute_dst() = 0;
    virtual void initialization_hook() {}
    virtual void accumulate_vsbr() {}
    virtual void compute_diff_src() {}

    void forward() {
        accumulate_vmax();
        accumulate_vsum();
        compute_dst();
    }

    void backward() {
        accumulate_vsbr();
        compute_diff_src();
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        if (pd_->is_fwd() || is_logsoftmax_)
            exp_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, true,
                    reg_exp_injector_table, injector_mask));
        if (pd_->is_fwd() && is_logsoftmax_) {
            log_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, true,
                    reg_log_injector_table, injector_mask));
        }

        compute_predefined_variables();
        preamble();
        initialization_hook();
        if (exp_injector_) exp_injector_->load_table_addr();
        if (log_injector_) log_injector_->load_table_addr();
        if (axis_simd_tail_) prepare_tail_mask();
        load_common_params();
        if (pd_->is_fwd())
            forward();
        else
            backward();
        postamble();
        if (exp_injector_) exp_injector_->prepare_table();
        if (log_injector_) log_injector_->prepare_table();
    }

    jit_softmax_base_t(const softmax_pd_t *pd)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
        , pd_(pd)
        , data_d_(pd_->dst_md()) {
        is_bf16_ = data_d_.data_type() == data_type::bf16;
        data_type_size_ = is_bf16_ ? sizeof(bfloat16_t) : sizeof(float);
        simd_w_ = vlen / sizeof(float); // bf16 works
    }
};

template <cpu_isa_t isa>
struct jit_softmax_t;

template <>
struct jit_softmax_t<lasx> : public jit_softmax_base_t<lasx> {
    Vmm tail_vmask = Vmm(0);

    void prepare_tail_mask() override {
        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

        mov_imm(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - axis_simd_tail_]));
        uni_xvld(tail_vmask, reg_tmp, 0);
    }

    void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) override {
        // 128/256-bit shuffle
        xvpermi_q(vtmp, v, 0x1);
        perform_op(v, vtmp, op);

        // 64/128-bit shuffle
        xvshuf4i_w(vtmp, v, 0x4E);
        perform_op(v, vtmp, op);

        // 32/64-bit shuffle
        xvshuf4i_w(vtmp, v, 0xB1);
        perform_op(v, vtmp, op);
    }

    void movups_tail(const Xbyak_loongarch64::XVReg &x, const int tail,
            const Xbyak_loongarch64::XReg &addr){
        for (int i = 0; i < tail; i++){
            ld_w(X_TMP_0, addr, i);
            xvinsgr2vr_w(x, X_TMP_0, i);
        }
    }

    void movups_tail(const Xbyak_loongarch64::XReg &addr, const int tail,
            const Xbyak_loongarch64::XVReg &x){
        for (int i = 0; i < tail; i++){
            xvpickve2gr_w(X_TMP_0, x, i);
            st_w(X_TMP_0, addr, i);
        }
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        xvbsll_v(vmax, vneg_flt_max, 0);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                if (!tail){
                    uni_xvld(XVReg(31), src_ptr(axis_stride_ * i), 0);
                    xvfmax_s(vmax, vmax, XVReg(31));
                }
                else {
                    vtmp = Vmm(i + 1);
                    load_bytes(vtmp, src_ptr(axis_stride_ * i), 0, axis_simd_tail_ * sizeof(float));

                    xvbitsel_v(vtmp, vneg_flt_max, vtmp, tail_vmask);
                    xvfmax_s(vmax, vmax, vtmp);
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_xvld(vreg_tmp_src, src_ptr(axis_stride_ * i), 0);
                    xvfsub_s(vreg_tmp_src, vreg_tmp_src, vmax);

                    if (is_logsoftmax_) // store before applying exp
                        uni_xvst(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0);

                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());

                    xvfadd_s(vsum, vsum, vreg_tmp_src);

                    if (is_softmax_) // store after applying exp
                        uni_xvst(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0);
                } else {
                    load_bytes(vreg_tmp_src, src_ptr(axis_stride_ * i), 0, axis_simd_tail_ * sizeof(float));
                    xvfsub_s(vreg_tmp_src, vreg_tmp_src, vmax);

                    if (is_logsoftmax_) // store before applying exp
                        store_bytes(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0, axis_simd_tail_ * sizeof(float));

                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    vtmp = Vmm(vreg_tmp_src.getIdx() + 1);
                    uni_vpxor(vtmp, vtmp, vtmp);
                    xvbitsel_v(vtmp, vtmp, vreg_tmp_src, tail_vmask);
                    xvfadd_s(vsum, vsum, vtmp);

                    if (is_softmax_) // store after applying exp
                        store_bytes(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0, axis_simd_tail_ * sizeof(float));
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_)
            xvfdiv_s(vsum, vone, vsum);

        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    if (is_softmax_) {
                        uni_xvld(XVReg(31), dst_ptr(axis_stride_ * i), 0);
                        xvfmul_s(vreg_tmp_src, vsum, XVReg(31));
                    }

                    if (is_logsoftmax_) {
                        uni_xvld(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0);
                        xvfsub_s(vreg_tmp_src, vreg_tmp_src, vsum);
                    }

                    uni_xvst(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0);
                } else {
                    load_bytes(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0, axis_simd_tail_ * sizeof(float));

                    if (is_softmax_)
                        xvfmul_s(vreg_tmp_src, vreg_tmp_src, vsum);

                    if (is_logsoftmax_)
                        xvfsub_s(vreg_tmp_src, vreg_tmp_src, vsum);

                    if (axis_is_blocked_) {
                        uni_vpxor(vzeropad, vzeropad, vzeropad);
                        xvbitsel_v(vzeropad, vzeropad, vreg_tmp_src, tail_vmask);
                        uni_xvst(vzeropad, dst_ptr(axis_stride_ * i), 0);
                    } else {
                        store_bytes(vreg_tmp_src, dst_ptr(axis_stride_ * i), 0, axis_simd_tail_ * sizeof(float));
                    }
                }
            }
        });
    }

    void operator()(const call_params_t *p) override {
        return jit_generator::operator()(p);
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {}
};

} // namespace

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(new softmax_impl::driver_t<isa>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto data_type_size = data_d.data_type() == data_type::bf16
            ? sizeof(bfloat16_t)
            : sizeof(float);
    const auto &bd = data_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto outer_stride = data_d.padded_dims()[axis] * inner_size;
    const auto outer_size = data_d.nelems(true) / outer_stride;

    parallel_nd(outer_size, inner_size, [&](dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride) * data_type_size;
        const char *src_ptr = src + offset;
        char *dst_ptr = dst + offset;
        softmax_driver_->exec(src_ptr, dst_ptr, outer_stride);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::jit_uni_softmax_bwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(new softmax_impl::driver_t<isa>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::~jit_uni_softmax_bwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const char *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->dst_md());
    const auto data_type_size = data_d.data_type() == data_type::bf16
            ? sizeof(bfloat16_t)
            : sizeof(float);
    const auto &bd = data_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto outer_stride = data_d.padded_dims()[axis] * inner_size;
    const auto outer_size = data_d.nelems(true) / outer_stride;

    parallel_nd(outer_size, inner_size, [&](dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride) * data_type_size;
        char *diff_src_ptr = diff_src + offset;
        const char *dst_ptr = dst + offset;
        const char *diff_dst_ptr = diff_dst + offset;
        softmax_driver_->exec(
                diff_src_ptr, dst_ptr, diff_dst_ptr, outer_stride);
    });

    return status::success;
}

namespace softmax_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {

    driver_t(const softmax_pd_t *pd) : pd_(pd), ker_(pd_) {}

    void exec(const void *src, void *dst, const dim_t outer_stride) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.spat_offt_count = outer_stride * ker_.data_type_size_;
        p.src = src;
        p.dst = dst;
        ker_(&p);
    }

    void exec(void *diff_src, const void *dst, const void *diff_dst,
            const dim_t outer_stride) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.spat_offt_count = outer_stride * ker_.data_type_size_;
        p.src = diff_src;
        p.dst = dst;
        p.diff_dst = diff_dst;
        ker_(&p);
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    const softmax_pd_t *pd_;
    jit_softmax_t<isa> ker_;
};

} // namespace softmax_impl

/* struct instantiation */
template struct jit_uni_softmax_fwd_t<lasx>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
