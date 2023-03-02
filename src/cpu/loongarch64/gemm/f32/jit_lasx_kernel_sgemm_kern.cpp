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

#include "cpu/loongarch64/gemm/f32/jit_lasx_kernel_sgemm_kern.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

int jit_lasx_kernel_sgemm_kern::next_acc(int idx, int um, int un) const {
    while (!(((idx / unroll_n_) < std::max(1, um / nelt_per_vecreg_))
            || ((idx % unroll_n_) < un)))
        idx++;
    return idx;
}

void jit_lasx_kernel_sgemm_kern::prefetchB_beforeBload(
        int um, int un, int k_idx, int n_idx) {
    {
        if ((n_idx == 0) && (k_idx == 0) && (un == unroll_n_) && (um != 16)) {
            uni_preld(0, BO_, elt_size_ * (PREFETCHSIZEB_ + offb_));
            offb_ += 16;
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchB_beforeFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    {
        if ((um == 16) || (un < unroll_n_)) {
            if ((k_idx + m_idx + n_idx) == 0) {
                uni_preld(0, BO_, elt_size_ * (PREFETCHSIZEB_ + offb_));
                offb_ += 16;
            }
            if ((um == 16) && (un == 4) && (k_idx == 2)
                    && ((m_idx + n_idx) == 0)) {
                uni_preld(0, BO_, elt_size_ * (PREFETCHSIZEB_ + offb_));
                offb_ += 16;
            }
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchA_afterFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
        if (un == unroll_n_) {
            if (((um < nelt_per_vecreg_) && (n_idx == 0)
                        && (k_idx == std::min(2, nelt_per_vecreg_ / um - 1)))
                    || ((um == nelt_per_vecreg_) && (un == unroll_n_)
                            && (n_idx == 1) && (k_idx == 0))) {
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            }
        }
}

void jit_lasx_kernel_sgemm_kern::prefetchA_afterBload(
        int um, int un, int k_idx, int n_idx) {
    {
        if ((um == unroll_m_) && (un == 2)) {
            if (k_idx % 3 == 0) {
                if (n_idx == 1) {
                    if (k_idx == 0) off_ += 16;
                    uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                    off_ += 16;
                }
                if ((k_idx == 0) && (n_idx == 0)) {
                    uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                    off_ += 16;
                }
            } else {
                if (n_idx == 1) {
                    uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                    off_ += 16;
                }
            }
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchB_afterFMA(
        int k_idx, int n_idx, int m_idx) {
}

void jit_lasx_kernel_sgemm_kern::prefetchA_beforeFMA(
        int um, int un, int k_idx, int n_idx, int m_idx) {
    {
        if ((um == unroll_m_) && (un == unroll_n_)) {
            if (((k_idx == 0) && (n_idx % 2 == 1) && (m_idx == 0))
                    || ((k_idx == 1) && (n_idx == 2) && (m_idx == 0))
                    || ((k_idx == 2) && (n_idx == 0) && (m_idx == 2))
                    || ((k_idx == 2) && (n_idx == 3) && (m_idx == 0))
                    || ((k_idx == 3) && (n_idx == 1) && (m_idx == 0))) {
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            }
        }
        if ((um == unroll_m_) && (un == 1)) {
            if (m_idx == 2) {
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            } else if ((m_idx == 0) && ((k_idx == 1) || (k_idx == 2))) {
                uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
                off_ += 16;
            }
        }
        if ((um == 16) && (un == unroll_n_) && (m_idx == 0) && (n_idx == 2)) {
            uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
            off_ += 16;
        }
        if ((um == 8) && (un == unroll_n_) && (m_idx == 0) && (n_idx == 1)
                && (k_idx == 2)) {
            uni_preld(0, AO_, elt_size_ * (PREFETCHSIZEA_ + off_));
            off_ += 16;
        }
    }
}

void jit_lasx_kernel_sgemm_kern::prefetchC_afterBload(
        int um, int un, int k_idx, int n_idx) {
}

void jit_lasx_kernel_sgemm_kern::prefetchC_beforeKloop(int um) {
        preld(2, AA_, -16 * elt_size_);

        preld(0, CO1_, 7 * elt_size_);
        add_d(X_TMP_1, CO1_, LDC_);
        preld(0, X_TMP_1, 7 * elt_size_);
        preld(0, CO2_, 7 * elt_size_);
        add_d(X_TMP_1, CO2_, LDC_);
        preld(0, X_TMP_1, 7 * elt_size_);

        preld(0, CO1_, 23 * elt_size_);
        add_d(X_TMP_1, CO1_, LDC_);
        preld(0, X_TMP_1, 23 * elt_size_);
        preld(0, CO2_, 23 * elt_size_);
        add_d(X_TMP_1, CO2_, LDC_);
        preld(0, X_TMP_1, 23 * elt_size_);

        addi_d(LL_, LL_, second_fetch_);

        preld(2, AA_, 0);
}

void jit_lasx_kernel_sgemm_kern::generate() {

    int i, unroll_x, unroll_y, uy_bin, ux_bin;
    int sepload = 0;

    std::vector<Xbyak_loongarch64::Label> unroll_x_label(MAX_UNROLL_M),
            unroll_y_label((MAX_UNROLL_N_BIN + 1) * MAX_UNROLL_M);
    std::vector<Xbyak_loongarch64::Label> end_n_loop_label(MAX_UNROLL_M);
    Xbyak_loongarch64::Label end_m_loop_label;

    preamble();

        ld_d(M_, M_, 0);
        ld_d(N_, N_, 0);
        ld_d(K_, K_, 0);

    add_imm(A_, A_, addr_off_ * elt_size_, X_TMP_0);
    add_imm(B_, B_, addr_off_ * elt_size_, X_TMP_0);

    slli_d(LDC_, LDC_, elt_size_bin_);

    for (unroll_x = unroll_m_, i = 0, ux_bin = unroll_m_bin_; unroll_x >= 1;
            unroll_x -= std::min(nelt_per_vecreg_, std::max(1, unroll_x / 2)),
        i++, ux_bin--) {

        if (unroll_x == unroll_m_) {
            add_d(J_, M_, zero);
            mov_imm(X_TMP_1, unroll_m_);
            blt(J_, X_TMP_1, unroll_x_label[i + 1]);
            L_aligned(unroll_x_label[i]);
        } else {
            L_aligned(unroll_x_label[i]);
            andi(X_TMP_1, J_, unroll_x);
            if (unroll_x > 1)
                bge(zero, X_TMP_1, unroll_x_label[i + 1]);
            else
                bge(zero, X_TMP_1, end_m_loop_label);
        }

        add_d(AA_, KK_, zero);

        if ((1 << ux_bin) > unroll_x) {
            mov_imm(X_TMP_1, unroll_x * elt_size_);
            mul_d(AA_, AA_, X_TMP_1);
        }
        else
            slli_d(AA_, AA_, elt_size_bin_ + ux_bin);

        add_d(AA_, AA_, A_);
        add_d(CO1_, C_, zero);

        slli_d(X_TMP_1, LDC_, 1);
        add_d(CO2_, C_, X_TMP_1);

        add_imm(C_, C_, unroll_x * elt_size_, X_TMP_0);
        add_d(BO_, B_, zero);

        for (unroll_y = unroll_n_, uy_bin = unroll_n_bin_; unroll_y >= 1;
                unroll_y /= 2, uy_bin--) {

            if (unroll_y == unroll_n_) {
                srai_d(I_, N_, uy_bin);
                bge(zero, I_, unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1]);
                L_aligned(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
            } else {
                L_aligned(unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
                andi(X_TMP_1, N_, unroll_y);
                if (uy_bin == 0)
                    bge(zero, X_TMP_1, end_n_loop_label[i]);
                else
                    bge(zero, X_TMP_1, unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin - 1]);
            }

            uni_preld(2, AA_, -1 * addr_off_ * elt_size_);

            switch (unroll_x) {
                case 8:
                        loop<Xbyak_loongarch64::XVReg, Xbyak_loongarch64::XVReg, Xbyak_loongarch64::XReg, Xbyak_loongarch64::XVReg,
                                Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                                &jit_generator::uni_xvld,
                                &jit_generator::uni_xvldrepl_w);
                        update<Xbyak_loongarch64::XVReg, Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                                1, beta_zero_, &jit_generator::uni_fadd_s,
                                &jit_generator::uni_xvst,
                                &jit_generator::uni_xvld);
                    break;
                case 4:
                        loop<Xbyak_loongarch64::VReg, Xbyak_loongarch64::VReg, Xbyak_loongarch64::XReg, Xbyak_loongarch64::VReg,
                                Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                                &jit_generator::uni_xvld,
                                &jit_generator::uni_xvldrepl_w);
                        sepload = 1;

                    update<Xbyak_loongarch64::VReg, Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                            sepload, beta_zero_, &jit_generator::uni_fadd_s,
                            &jit_generator::uni_xvst,
                            &jit_generator::uni_xvld);

                    break;
                case 2:
                    {
                        loop<Xbyak_loongarch64::VReg, Xbyak_loongarch64::VReg, Xbyak_loongarch64::XReg, Xbyak_loongarch64::VReg,
                                Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                                //&Xbyak::CodeGenerator::vmovddup,
                                &jit_generator::uni_xvld,
                                &jit_generator::uni_xvldrepl_w);
                    }
                    update<Xbyak_loongarch64::VReg, Xbyak_loongarch64::XReg>(unroll_x, unroll_y, 1,
                            beta_zero_, &jit_generator::uni_fadd_s,
                            &jit_generator::uni_xvstelm_d0,
                            &jit_generator::uni_xvldrepl_d);
                    break;
                case 1:
                    {
                        loop<Xbyak_loongarch64::VReg, Xbyak_loongarch64::VReg, Xbyak_loongarch64::XReg, Xbyak_loongarch64::VReg,
                                Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                                &jit_generator::uni_xvldrepl_w,
                                &jit_generator::uni_xvldrepl_w);
                        sepload = 1;
                    }
                    update<Xbyak_loongarch64::VReg, Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                            sepload, beta_zero_, &jit_generator::uni_fadd_s,
                            &jit_generator::uni_xvstelm_w0,
                            &jit_generator::uni_xvldrepl_w);

                    break;
                default:
                    {
                        loop<Xbyak_loongarch64::XVReg, Xbyak_loongarch64::XVReg, Xbyak_loongarch64::XReg, Xbyak_loongarch64::XVReg,
                                Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                                &jit_generator::uni_xvld,
                                &jit_generator::uni_xvldrepl_w);
                        update<Xbyak_loongarch64::XVReg, Xbyak_loongarch64::XReg>(unroll_x, unroll_y,
                                1, beta_zero_, &jit_generator::uni_fadd_s,
                                &jit_generator::uni_xvst,
                                &jit_generator::uni_xvld);
                    }

                    break;
            }

            {
                if ((unroll_y != unroll_n_) || (unroll_x <= 4)) {
                    if (unroll_x == unroll_m_)
                        addi_d(AA_, AA_, 16 * elt_size_);
                    else
                        addi_d(AA_, AA_, 32 * elt_size_);
                } else
                    addi_d(AA_, AA_, 48 * elt_size_);
            }

            if (unroll_y == unroll_n_) {
                addi_d(I_, I_, -1);
                blt(zero, I_, unroll_y_label[i * (unroll_n_bin_ + 1) + uy_bin]);
            }
        }

        L_aligned(end_n_loop_label[i]);

        add_d(A_, AO_, zero);

        if (unroll_x == unroll_m_) {
            add_imm(J_, J_, -1 * unroll_x, X_TMP_0);
            mov_imm(X_TMP_1, unroll_x);
            bge(J_, X_TMP_1, unroll_x_label[i]);
        }
    }

    L_aligned(end_m_loop_label);

    postamble();
}

jit_lasx_kernel_sgemm_kern::jit_lasx_kernel_sgemm_kern(bool beta_zero)
    : jit_generator(nullptr, 65536) {

    beta_zero_ = beta_zero;
}
} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
