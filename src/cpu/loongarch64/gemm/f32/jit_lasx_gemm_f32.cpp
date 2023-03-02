/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#include <atomic>
#include <cmath>
#include <mutex>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/f32/gemm_utils_f32.hpp"
#include "cpu/gemm/f32/ref_gemm_f32.hpp"
#include "cpu/gemm/gemm_msan_unpoison.hpp"

#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/gemm/gemm_driver.hpp"

#include "cpu/loongarch64/gemm/f32/jit_lasx_gemm_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

#define CACHE_LINE_SIZE 64

#define STACKSIZE get_size_of_abi_save_regs()
#define STACK_K_CAPACITY 8192
#define SIZE 4
#define OFFSET 32
#define BASE_SHIFT 2
#define SECOND_FETCH 14

namespace lasx_gemm_f32 {
using namespace gemm_utils;
using namespace Xbyak_loongarch64;

struct xbyak_gemm_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lasx_gemm_f32_xbyak_gemm)
    xbyak_gemm_t(char isTransA, char isTransB, float beta, bool hasBias = false,
            void *code_ptr = nullptr,
            size_t code_size = 80 * Xbyak_loongarch64::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , isTransA(isTransA)
        , isTransB(isTransB)
        , hasBias(hasBias)
        , is_lasx(mayiuse(lasx))
        , UNROLL_M(is_lasx ? 16 : 8)
        , UNROLL_N(6)
        , isBeta0(beta == 0.0)
        , isBetaN(!isBeta0 && beta != 1.0)
        , PREFETCHSIZEA(128)
        , PREFETCHSIZEB((!isTransB) ? -16 : 0) {}

    // Fused multiply add; may become one or two instructions
    void fma(bool useFma, const XVReg &reg0, const XVReg &reg1, const XVReg &reg2,
            bool overWrite = false) {
        if (useFma) {
            xvfmadd_s(reg2, reg1, reg0, reg2);
        } else {
            if (!overWrite) {
                xvfmul_s(xr15, reg1, reg0);
                xvfadd_s(reg2, reg2, xr15);
            } else {
                xvfmul_s(reg1, reg1, reg0);
                xvfadd_s(reg2, reg2, reg1);
            }
        }
    }

    // Inner kernel with k=8
    void innerkernel8(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        XVReg fmareg(0);

        if (!isDirect) {
            uni_preld(0, AO1, (PREFETCHSIZEA + 0) * SIZE);
        } else {
            preldx(0, AO1, LDA4);
        }

        for (int i = 0; i < 8; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    } else {
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                add_d(AO1, AO1, LDA);
            }

            if (!isTransB) {
                uni_xvldrepl_w(xr2, BO1, (i - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, xr0, xr2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, xr1, xr2, fmareg);
            }
            if (i == 0) {
                if (!isTransB) { uni_preld(0, BO1, PREFETCHSIZEB * SIZE); }
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    if (i == 1) {
                        add_d(X_TMP_1, BO1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (isCopy) {
                uni_xvst(xr0, LDA4, (unroll_m * i + 0 * 8 - OFFSET) * SIZE);
                if (unroll_m >= 16) {
                    uni_xvst(xr1, LDA4, (unroll_m * i + 1 * 8 - OFFSET) * SIZE);
                }
                if (i == 7) { add_imm(LDA4, LDA4, unroll_m * 8 * SIZE, X_TMP_0); }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        add_d(X_TMP_1, BO1, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO1, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (i == 7) {
                if (!isTransB) { addi_d(BO1, BO1, 8 * SIZE); }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    if (i == 3) { uni_preld(0, BO2, PREFETCHSIZEB * SIZE); }
                    uni_xvldrepl_w(xr2, BO2, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    if (i == 4) {
                        add_d(X_TMP_1, BO2, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO2, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    if (i == 5) {
                        add_d(X_TMP_1, BO2, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO2, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }
            if (isTransB) {
                preldx(0, BO1, BO2);
                add_d(BO1, BO1, LDB);
            }

            if (i == 0) {
                if (unroll_m >= 4) {
                    if (!isDirect) {
                        uni_preld(0, AO1, (PREFETCHSIZEA + 2 * 8) * SIZE);
                    } else {
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 1 || i == 2) {
                if (unroll_m >= 8) {
                    if (!isDirect) {
                        uni_preld(0, AO1, (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE);
                    } else {
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 3 || i == 4 || i == 5 || i == 6) {
                if (unroll_m >= 16) {
                    if (!isDirect) {
                        uni_preld(0, AO1, (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE);
                    } else {
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 7) {
                if (!isTransB) {
                    if (unroll_n >=4) { addi_d(BO2, BO2, 8 * SIZE); }
                }
                if (!isTransA) {
                    preld(2, AA, 0);
                    add_d(AA, AA, LDA);
                }
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                    } else {
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
            }
        }

        if (!isDirect) { add_imm(AO1, AO1, unroll_m * 8 * SIZE, X_TMP_0); }
        addi_d(LL, LL, -1);
    }

    // Inner kernel with k=4
    void innerkernel4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        XVReg fmareg(0);

        if (!isDirect) {
            uni_preld(0, AO1, (PREFETCHSIZEA + 0) * SIZE);
        } else {
            preldx(0, AO1, LDA4);
        }

        for (int i = 0; i < 4; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    } else {
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                add_d(AO1, AO1, LDA);
            }

            if (!isTransB) {
                uni_xvldrepl_w(xr2, BO1, (i - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, xr0, xr2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, xr1, xr2, fmareg);
            }
            if (i == 0) {
                if (!isTransB) { uni_preld(0, BO1, PREFETCHSIZEB * SIZE); }
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    if (i == 1) {
                        add_d(X_TMP_1, BO1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (isCopy) {
                uni_xvst(xr0, LDA4, (unroll_m * i + 0 * 8 - OFFSET) * SIZE);
                if (unroll_m >= 16) {
                    uni_xvst(xr1, LDA4, (unroll_m * i + 1 * 8 - OFFSET) * SIZE);
                }
                if (i == 3) { add_imm(LDA4, LDA4, unroll_m * 4 * SIZE, X_TMP_0); }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        add_d(X_TMP_1, BO1, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO1, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (i == 7) {
                if (!isTransB) { addi_d(BO1, BO1, 8 * SIZE); }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    if (i == 3) { uni_preld(0, BO2, PREFETCHSIZEB * SIZE); }
                    uni_xvldrepl_w(xr2, BO2, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    if (i == 4) {
                        add_d(X_TMP_1, BO2, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO2, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    if (i == 5) {
                        add_d(X_TMP_1, BO2, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO2, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (i - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }
            if (isTransB) {
                preldx(0, BO1, BO2);
                add_d(BO1, BO1, LDB);
            }

            if (i == 0) {
                if (unroll_m >= 4) {
                    if (!isDirect) {
                        uni_preld(0, AO1, (PREFETCHSIZEA + 2 * 8) * SIZE);
                    } else {
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 1 || i == 2) {
                if (unroll_m >= 8) {
                    if (!isDirect) {
                        uni_preld(0, AO1, (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE);
                    } else {
                        preldx(0, AO1, LDA4);
                    }
                }
            }
            if (i == 3) {
                if (!isTransB) {
                    addi_d(BO1, BO1, 4 * SIZE);
                    if (unroll_n >= 4) { addi_d(BO2, BO2, 4 * SIZE); }
                }
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr0, AO1, (unroll_m * (i + 1) + 0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                    } else {
                        uni_xvld(xr1, AO1, (unroll_m * (i + 1) + 1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
            }
        }

        if (!isDirect) { add_imm(AO1, AO1, unroll_m * 4 * SIZE, X_TMP_0); }
    }

    // Inner kernel with k=2
    void innerkernel2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        XVReg fmareg(0);

        for (int i = 0; i < 2; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    } else {
                        uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                add_d(AO1, AO1, LDA);
            }

            if (!isTransB) {
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, xr0, xr2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, xr1, xr2, fmareg);
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    add_d(X_TMP_1, BO1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        add_d(X_TMP_1, BO1, LDB);
                        add_d(X_TMP_1, X_TMP_1, LDB);
                        uni_preld(0, X_TMP_1, PREFETCHSIZEB * SIZE);
                    }
                    add_d(X_TMP_1, BO1, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    uni_xvldrepl_w(xr2, BO2, (0 - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    add_d(X_TMP_1, BO2, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    add_d(X_TMP_1, BO2, LDB);
                    add_d(X_TMP_1, X_TMP_1, LDB);
                    uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
                } else {
                    uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, xr0, xr2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, xr1, xr2, fmareg);
                }
            }

            if (isCopy) {
                uni_xvst(xr0, LDA4, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
                if (unroll_m >= 16) {
                    uni_xvst(xr1, LDA4, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
                }
                add_imm(LDA4, LDA4, unroll_m * SIZE, X_TMP_0);
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
                    xvand_v(xr0, xr0, VMASK);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                    } else {
                        uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                        xvand_v(xr1, xr1, VMASK);
                    }
                }
                add_imm(AO1, AO1, unroll_m * SIZE, X_TMP_0);
            }

            if (!isTransB) {
                addi_d(BO1, BO1, SIZE);
                if (unroll_n >= 4) { addi_d(BO2, BO2, SIZE); }
            } else {
                add_d(BO1, BO1, LDB);
            }
        }
    }

    // Inner kernel with k=1
    void innerkernel1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11) {
        if (isDirect) {
            if (isLoad1Unmasked) {
                uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
            } else {
                uni_xvld(xr0, AO1, (0 * 8 - OFFSET) * SIZE);
                xvand_v(xr0, xr0, VMASK);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr1, AO1, (1 * 8 - OFFSET) * SIZE);
                    xvand_v(xr1, xr1, VMASK);
                }
            }
            add_d(AO1, AO1, LDA);
        }

        if (!isTransB) {
            uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
        } else {
            uni_xvldrepl_w(xr2, BO1, (0 - OFFSET) * SIZE);
        }
        fma(useFma, xr0, xr2, reg00);
        if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg06); }

        if (unroll_n >= 2) {
            if (!isTransB) {
                add_d(X_TMP_1, BO1, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (1 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg01);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg07); }
        }

        if (unroll_n >= 3) {
            if (!isTransB) {
                add_d(X_TMP_1, BO1, LDB);
                add_d(X_TMP_1, X_TMP_1, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (2 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg02);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg08); }
        }

        if (unroll_n >= 4) {
            if (!isTransB) {
                uni_xvldrepl_w(xr2, BO2, (0 - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (3 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg03);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg09); }
        }

        if (unroll_n >= 5) {
            if (!isTransB) {
                add_d(X_TMP_1, BO2, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (4 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg04);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg10); }
        }

        if (unroll_n >= 6) {
            if (!isTransB) {
                add_d(X_TMP_1, BO2, LDB);
                add_d(X_TMP_1, X_TMP_1, LDB);
                uni_xvldrepl_w(xr2, X_TMP_1, (0 - OFFSET) * SIZE);
            } else {
                uni_xvldrepl_w(xr2, BO1, (5 - OFFSET) * SIZE);
            }
            fma(useFma, xr0, xr2, reg05);
            if (unroll_m >= 16) { fma(useFma, xr1, xr2, reg11); }
        }

        if (isCopy) {
            uni_xvst(xr0, LDA4, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
            if (unroll_m >= 16) {
                uni_xvst(xr1, LDA4, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
            }
            add_imm(LDA4, LDA4, unroll_m * SIZE, X_TMP_0);
        }

        if (!isDirect) {
            if (isLoad1Unmasked) {
                uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
            } else {
                uni_xvld(xr0, AO1, (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE);
                xvand_v(xr0, xr0, VMASK);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr1, AO1, (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE);
                    xvand_v(xr1, xr1, VMASK);
                }
            }
            add_imm(AO1, AO1, unroll_m * SIZE, X_TMP_0);
        }

        if (!isTransB) {
            addi_d(BO1, BO1, SIZE);
            if (unroll_n >= 4) { addi_d(BO2, BO2, SIZE); }
        } else {
            add_d(BO1, BO1, LDB);
        }
    }

    // Main kernel; does prefetching and calls innerkernel{1,2,4,8} as
    // appropriate
    // After calculating results in registers, writes back to C matrix
    void kernel(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const XVReg &reg00, const XVReg &reg01, const XVReg &reg02,
            const XVReg &reg03, const XVReg &reg04, const XVReg &reg05,
            const XVReg &reg06, const XVReg &reg07, const XVReg &reg08,
            const XVReg &reg09, const XVReg &reg10, const XVReg &reg11,
            const XVReg &reg12, const XVReg &reg13, const XVReg &reg14,
            const XVReg &reg15, const XVReg &reg16, const XVReg &reg17,
            const XVReg &reg18, const XVReg &reg19, const XVReg &reg20,
            const XVReg &reg21, const XVReg &reg22, const XVReg &reg23) {
        if (!isDirect) {
            addi_d(AO1, sp, 256 + OFFSET * SIZE);
        } else {
            add_d(AO1, A, zero);
        }

        if (isCopy) {
            addi_d(LDA4, sp, 256 + OFFSET * SIZE);
        } else {
            slli_d(X_TMP_1, LDA, 3);
            addi_d(LDA4, X_TMP_1, (8 - 1 - OFFSET) * SIZE);
        }

        if (isTransB) {
            slli_d(X_TMP_1, LDB, 2);
            addi_d(BO2, X_TMP_1, (8 - 1 - OFFSET) * SIZE);
            slli_d(X_TMP_1, LDB, 1);
            add_d(BO2, BO2, X_TMP_1);
        }

        if (!isDirect) {
            if (isLoad1Unmasked) {
                uni_xvld(xr0, AO1, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
            } else {
                uni_xvld(xr0, AO1, (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE);
                xvand_v(xr0, xr0, VMASK);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    uni_xvld(xr1, AO1, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
                } else {
                    uni_xvld(xr1, AO1, (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE);
                    xvand_v(xr1, xr1, VMASK);
                }
            }
        }

        for (int i = 4; i < 10; i++) {
            xvxor_v(XVReg(i), XVReg(i), XVReg(i));
            xvxor_v(XVReg(i + 6), XVReg(i + 6), XVReg(i + 6));
        }

        srai_d(LL, K, 3);

        std::vector<Label> labels(8);

        addi_d(LL, LL, -1 * SECOND_FETCH);
        bge(zero, LL, labels[1]);

        L(labels[0]);
        innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        blt(zero, LL, labels[0]);

        L(labels[1]);
        uni_preld(0, CO1, (unroll_m - 1) * SIZE);
        if (unroll_n >= 2) {
            add_d(X_TMP_1, CO1, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }
        if (unroll_n >= 3) {
            add_d(X_TMP_1, CO1, LDC);
            add_d(X_TMP_1, X_TMP_1, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }
        if (unroll_n >= 4) uni_preld(0, CO2, (unroll_m - 1) * SIZE);
        if (unroll_n >= 5) {
            add_d(X_TMP_1, CO2, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }
        if (unroll_n >= 6) {
            add_d(X_TMP_1, CO2, LDC);
            add_d(X_TMP_1, X_TMP_1, LDC);
            uni_preld(0, X_TMP_1, (unroll_m - 1) * SIZE);
        }

        addi_d(LL, LL, SECOND_FETCH);
        bge(zero, LL, labels[3]);

        L(labels[2]);
        innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        blt(zero, LL, labels[2]);

        L(labels[3]);
        andi(X_TMP_0, K, 4);
        bge(zero, X_TMP_0, labels[4]);
        innerkernel4(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);

        L(labels[4]);
        andi(X_TMP_0, K, 2);
        bge(zero, X_TMP_0, labels[5]);
        innerkernel2(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);

        L(labels[5]);
        if (unroll_m == 16) {
            if (unroll_n <= 3) {
                xvfadd_s(reg00, reg00, reg12);
                xvfadd_s(reg01, reg01, reg13);
                xvfadd_s(reg02, reg02, reg14);
                xvfadd_s(reg06, reg06, reg18);
                xvfadd_s(reg07, reg07, reg19);
                xvfadd_s(reg08, reg08, reg20);
            }
        }

        if (unroll_m <= 8) {
            xvfadd_s(reg00, reg00, reg12);
            xvfadd_s(reg01, reg01, reg13);
            xvfadd_s(reg02, reg02, reg14);
            xvfadd_s(reg03, reg03, reg15);
            xvfadd_s(reg04, reg04, reg16);
            xvfadd_s(reg05, reg05, reg17);
        }

        andi(X_TMP_0, K, 1);
        bge(zero, X_TMP_0, labels[6]);
        innerkernel1(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11);

        L(labels[6]);
        xvldrepl_w(VALPHA, ALPHA.getXReg(), ALPHA.getOffset());

        if (isBetaN) { xvldrepl_w(VBETA, BETA.getXReg(), BETA.getOffset()); }

        // Write back the results; all beta and bias cases need to be
        // handled
        switch (unroll_n) {
            case 1: add_d(rax, LDC, zero); break;
            case 2: slli_d(rax, LDC, 1); break;
            case 3: { mov_imm(X_TMP_0, 3); mul_d(rax, LDC, X_TMP_0); } break;
            case 4: { mov_imm(X_TMP_0, 5); mul_d(rax, LDC, X_TMP_0); } break;
            case 5:
                mov_imm(X_TMP_0, 5);
                mul_d(rax, LDC, X_TMP_0);
                break;
            case 6:
                mov_imm(X_TMP_0, 6);
                mul_d(rax, LDC, X_TMP_0);
                break;
        }

        if (hasBias) {
            ld_d(BIAS1, BIAS.getXReg(), BIAS.getOffset());
            if (isLoad1Unmasked) {
                xvld(VBIAS1, BIAS1, 0);
            } else {
                xvld(VBIAS1, BIAS1, 0);
                xvand_v(VBIAS1, VBIAS1, VMASK);
            }
        }

        for (int i = 0; i < unroll_n; i++) {
            xvfmul_s(XVReg(i + 4), XVReg(i + 4), VALPHA);
            if (!isBeta0) {
                if (isLoad1Unmasked) {
                    switch (i) {
                        case 0: xvld(xr0, CO1, 0); break;
                        case 1: xvldx(xr0, CO1, LDC); break;
                        case 2:
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO1, X_TMP_0);
                            break;
                        case 3: xvld(xr0, CO2, 0); break;
                        case 4: xvldx(xr0, CO2, LDC); break;
                        case 5:
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO2, X_TMP_0);
                            break;
                    }
                } else {
                    switch (i) {
                        case 0:
                            xvld(xr0, CO1, 0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 1:
                            xvldx(xr0, CO1, LDC);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 2:
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO1, X_TMP_0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 3:
                            xvld(xr0, CO2, 0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 4:
                            xvldx(xr0, CO2, LDC);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                        case 5:
                            slli_d(X_TMP_0, LDC, 1);
                            xvldx(xr0, CO2, X_TMP_0);
                            xvand_v(xr0, xr0, VMASK);
                            break;
                    }
                }

                if (!isBetaN) {
                    xvfadd_s(XVReg(i + 4), xr0, XVReg(i + 4));
                } else {
                    fma(useFma, VBETA, xr0, XVReg(i + 4), true);
                }
            }
            if (hasBias) { xvfadd_s(XVReg(i + 4), VBIAS1, XVReg(i + 4)); }
            if (isLoad1Unmasked) {
                switch (i) {
                    case 0: xvst(XVReg(i + 4), CO1, 0); break;
                    case 1:
                        xvstx(XVReg(i + 4), CO1, LDC);
                        break;
                    case 2:
                        slli_d(X_TMP_0, LDC, 1);
                        xvstx(XVReg(i + 4), CO1, X_TMP_0);
                        break;
                    case 3: xvst(XVReg(i + 4), CO2, 0); break;
                    case 4:
                        xvstx(XVReg(i + 4), CO2, LDC);
                        break;
                    case 5:
                        slli_d(X_TMP_0, LDC, 1);
                        xvstx(XVReg(i + 4), CO2, X_TMP_0);
                        break;
                }
            } else {
                switch (i) {
                    case 0:
                        store_mask_words(XVReg(i + 4), CO1, 0, VMASK);
                        break;
                    case 1:
                        add_d(X_TMP_1, CO1, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                    case 2:
                        add_d(X_TMP_1, CO1, LDC);
                        add_d(X_TMP_1, X_TMP_1, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                    case 3:
                        store_mask_words(XVReg(i + 4), CO2, 0, VMASK);
                        break;
                    case 4:
                        add_d(X_TMP_1, CO2, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                    case 5:
                        add_d(X_TMP_1, CO2, LDC);
                        add_d(X_TMP_1, X_TMP_1, LDC);
                        store_mask_words(XVReg(i + 4), X_TMP_1, 0, VMASK);
                        break;
                }
            }

            if (unroll_m >= 16) {
                // Re-use xr4 (VBIAS2)
                if (i == 0) {
                    if (hasBias) {
                        if (isLoad1Unmasked) {
                            uni_xvld(VBIAS2, BIAS1, 8 * SIZE);
                        } else {
                            uni_xvld(VBIAS2, BIAS1, 8 * SIZE);
                            xvand_v(VBIAS2, VBIAS2, VMASK);
                        }
                    }
                }
                xvfmul_s(XVReg(i + 10), XVReg(i + 10), VALPHA);
                if (!isBeta0) {
                    if (isLoad2Unmasked) {
                        switch (i) {
                            case 0: uni_xvld(xr0, CO1, 8 * SIZE); break;
                            case 1:
                                add_d(X_TMP_1, CO1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                            case 2:
                                add_d(X_TMP_1, CO1, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                            case 3: uni_xvld(xr0, CO2, 8 * SIZE); break;
                            case 4:
                                add_d(X_TMP_1, CO2, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                            case 5:
                                add_d(X_TMP_1, CO2, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                break;
                        }
                    } else {
                        switch (i) {
                            case 0:
                                uni_xvld(xr0, CO1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 1:
                                add_d(X_TMP_1, CO1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 2:
                                add_d(X_TMP_1, CO1, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 3:
                                uni_xvld(xr0, CO2, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 4:
                                add_d(X_TMP_1, CO2, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                            case 5:
                                add_d(X_TMP_1, CO2, LDC);
                                add_d(X_TMP_1, X_TMP_1, LDC);
                                uni_xvld(xr0, X_TMP_1, 8 * SIZE);
                                xvand_v(xr0, xr0, VMASK);
                                break;
                        }
                    }
                    if (!isBetaN) {
                        xvfadd_s(XVReg(i + 10), xr0, XVReg(i + 10));
                    } else {
                        fma(useFma, VBETA, xr0, XVReg(i + 10), true);
                    }
                }
                if (hasBias) { xvfadd_s(XVReg(i + 10), VBIAS2, XVReg(i + 10)); }
                if (isLoad2Unmasked) {
                    switch (i) {
                        case 0:
                            uni_xvst(XVReg(i + 10), CO1, 8 * SIZE);
                            break;
                        case 1:
                            add_d(X_TMP_1, CO1, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                        case 2:
                            add_d(X_TMP_1, CO1, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                        case 3:
                            uni_xvst(XVReg(i + 10), CO2, 8 * SIZE);
                            break;
                        case 4:
                            add_d(X_TMP_1, CO2, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                        case 5:
                            add_d(X_TMP_1, CO2, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            uni_xvst(XVReg(i + 10), X_TMP_1, 8 * SIZE);
                            break;
                    }
                } else {
                    switch (i) {
                        case 0:
                            store_mask_words(XVReg(i + 10), CO1, 8 * SIZE, VMASK);
                            break;
                        case 1:
                            add_d(X_TMP_1, CO1, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                        case 2:
                            add_d(X_TMP_1, CO1, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                        case 3:
                            store_mask_words(XVReg(i + 10), CO2, 8 * SIZE, VMASK);
                            break;
                        case 4:
                            add_d(X_TMP_1, CO2, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                        case 5:
                            add_d(X_TMP_1, CO2, LDC);
                            add_d(X_TMP_1, X_TMP_1, LDC);
                            store_mask_words(XVReg(i + 10), X_TMP_1, 8 * SIZE, VMASK);
                            break;
                    }
                }
            }
            if (i == 2) add_d(CO1, CO1, rax);
        }
        if (unroll_n >= 4) { add_d(CO2, CO2, rax); }

        // Compute next address of B
        if (!isTransB) {
            slli_d(rax, K, 2);
            switch (unroll_n) {
                case 1:
                    add_d(BO1, BO1, LDB);
                    add_d(BO2, BO2, LDB);
                    break;
                case 2:
                    slli_d(X_TMP_0, LDB, 1);
                    add_d(BO1, BO1, X_TMP_0);
                    add_d(BO2, BO2, X_TMP_0);
                    break;
                case 3:
                    add_d(BO1, BO1, LDB3);
                    add_d(BO2, BO2, LDB3);
                    break;
                case 4:
                    slli_d(X_TMP_0, LDB, 2);
                    add_d(BO1, BO1, X_TMP_0);
                    add_d(BO2, BO2, X_TMP_0);
                    break;
                case 5:
                    addi_d(X_TMP_0, zero, 5);
                    mul_d(X_TMP_0, LDB, X_TMP_0);
                    add_d(BO1, BO1, X_TMP_0);
                    add_d(BO2, BO2, X_TMP_0);
                    break;
                case 6:
                    slli_d(X_TMP_0, LDB3, 1);
                    add_d(BO1, BO1, X_TMP_0);
                    add_d(BO2, BO2, X_TMP_0);
                    break;
            }
            sub_d(BO1, BO1, rax);
            sub_d(BO2, BO2, rax);
        } else {
            mul_d(rax, LDB, K);
            sub_d(BO1, BO1, rax);
            add_imm(BO1, BO1, unroll_n * SIZE, X_TMP_0);
        }
    }

    void kernel_16x6(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, xr4, xr5, xr6, xr7, xr8, xr9, xr10, xr11,
                xr12, xr13, xr14, xr15, xr4, xr5, xr6, xr7, xr8, xr9,
                xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_16x5(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, xr4, xr5, xr6, xr7, xr8, xr9, xr10, xr11,
                xr12, xr13, xr14, xr15, xr4, xr5, xr6, xr7, xr8, xr9,
                xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_16x4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, xr4, xr5, xr6, xr7, xr8, xr9, xr10, xr11,
                xr12, xr13, xr14, xr15, xr4, xr5, xr6, xr7, xr8, xr9,
                xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_16x3(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, xr4, xr5, xr6, xr7, xr8, xr9, xr10,
                xr11, xr12, xr13, xr14, xr15, xr7, xr8, xr9, xr7, xr8,
                xr9, xr13, xr14, xr15, xr13, xr14, xr15);
    }

    void kernel_16x2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_16x1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_8x6(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, xr4, xr5, xr6, xr7, xr8, xr9, xr10,
                xr11, xr12, xr13, xr14, xr15, xr10, xr11, xr12, xr13,
                xr14, xr15, xr10, xr11, xr12, xr13, xr14, xr15);
    }

    void kernel_8x5(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy);
    }

    void kernel_8x4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy);
    }

    void kernel_8x3(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, xr4, xr5, xr6, xr7, xr8, xr9, xr10,
                xr11, xr12, xr13, xr14, xr15, xr7, xr8, xr9, xr7, xr8,
                xr9, xr13, xr14, xr15, xr13, xr14, xr15);
    }

    void kernel_8x2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_8x1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    // Function for packing if needed
    void do_pack(int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
        std::vector<Label> labels(6);

        XReg reg(0);

        add_d(BO1, A, zero);
        addi_d(AO1, sp, 256 + OFFSET * SIZE);

        // do_pack exec only when isTransA is true
        if (isTransA) {
            // make t0 to LDA * 2
            slli_d(t0, LDA, 1);
            // make rax to LDA * 4
            slli_d(rax, LDA, 2);
            add_d(BO2, BO1, rax);
            add_d(CO1, t0, LDA);
            xvinsgr2vr_d(xr7, zero, 0);
            xvinsgr2vr_d(xr7, LDA, 1);
            xvinsgr2vr_d(xr7, t0, 2);
            xvinsgr2vr_d(xr7, CO1, 3);
        }

        srai_d(LL, K, 2);
        bge(zero, LL, labels[1]);

        L(labels[0]);
        // do_pack exec only when isTransA is true
        if (!isTransA) {
          assert(0);
        } else {
            if (isLoad1Unmasked) {
                for (int i = 0; i < 2; i++) {
                    reg = (i % 2 == 0) ? BO1 : BO2;
                    vld(vr0, reg, (0 * 8 - OFFSET) * SIZE);
                    add_d(X_TMP_1, reg, LDA);
                    vld(vr1, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                    add_d(BO2, reg, t0);
                    vilvl_w(vr4, vr1, vr0);
                    vilvh_w(vr5, vr1, vr0);
                    vld(vr0, BO2, (0 * 8 - OFFSET) * SIZE);
                    add_d(X_TMP_1, BO2, LDA);
                    vld(vr1, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                    add_d(BO2, BO2, t0);
                    vilvl_w(vr6, vr1, vr0);
                    vilvh_w(vr2, vr1, vr0);

                    vilvl_d(vr0, vr6, vr4);
                    vilvh_d(vr1, vr6, vr4);
                    vst(vr0, AO1, (unroll_m * 0 + i * 4 - OFFSET) * SIZE);
                    vst(vr1, AO1, (unroll_m * 1 + i * 4 - OFFSET) * SIZE);
                    vilvl_d(vr0, vr2, vr5);
                    vilvh_d(vr1, vr2, vr5);
                    vst(vr0, AO1, (unroll_m * 2 + i * 4 - OFFSET) * SIZE);
                    vst(vr1, AO1, (unroll_m * 3 + i * 4 - OFFSET) * SIZE);
                }
            } /* else if (is_lasx) {
                //
            }*/ else {
                vxor_v(vr4, vr4, vr4);
                add_d(BO2, BO1, rax);

                auto el_cp = [&](int section, int ld_step) {
                    XReg src_addr = section == 0 ? BO1 : BO2;
                    if (ld_step > 0) {
                        add_d(X_TMP_1, src_addr, ld_step == 1 ? LDA : (ld_step == 2 ? t0 : CO1));
                        vld(VReg(ld_step % 2), X_TMP_1, -OFFSET * SIZE);
                    } else
                        vld(VReg(ld_step % 2), src_addr, -OFFSET * SIZE);

                    vstelm_w(VReg(ld_step % 2), AO1, (ld_step + section * 4 - OFFSET) * SIZE, 0);
                    vstelm_w(VReg(ld_step % 2), AO1, unroll_m * SIZE + (ld_step + section * 4 - OFFSET) * SIZE, 1);
                    vstelm_w(VReg(ld_step % 2), AO1, unroll_m * 2 * SIZE + (ld_step + section * 4 - OFFSET) * SIZE, 2);
                    vstelm_w(VReg(ld_step % 2), AO1, unroll_m * 3 * SIZE + (ld_step + section * 4 - OFFSET) * SIZE, 3);
                };

                el_cp(0, 0);
                ld_d(X_TMP_0, M.getXReg(), M.getOffset());
                mov_imm(X_TMP_1, 1);
                beq(X_TMP_0, X_TMP_1, labels[4]);
                el_cp(0, 1);
                mov_imm(X_TMP_1, 2);
                beq(X_TMP_0, X_TMP_1, labels[4]);
                el_cp(0, 2);
                mov_imm(X_TMP_1, 3);
                beq(X_TMP_0, X_TMP_1, labels[4]);
                el_cp(0, 3);
                mov_imm(X_TMP_1, 4);
                beq(X_TMP_0, X_TMP_1, labels[4]);
                el_cp(1, 0);
                mov_imm(X_TMP_1, 5);
                beq(X_TMP_0, X_TMP_1, labels[4]);
                el_cp(1, 1);
                mov_imm(X_TMP_1, 6);
                beq(X_TMP_0, X_TMP_1, labels[4]);
                el_cp(1, 2);
                L(labels[4]);

                add_d(BO2, BO2, rax);
            }

            if (unroll_m >= 16) {
                assert(is_lasx);
                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        vld(vr0, BO2, (0 * 8 - OFFSET) * SIZE);
                        add_d(X_TMP_1, BO2, LDA);
                        vld(vr1, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                        add_d(BO2, BO2, t0);
                        vilvl_w(vr4, vr1, vr0);
                        vilvh_w(vr5, vr1, vr0);
                        vld(vr0, BO2, (0 * 8 - OFFSET) * SIZE);
                        add_d(X_TMP_1, BO2, LDA);
                        vld(vr1, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                        if (i == 0) add_d(BO2, BO2, t0);
                        vilvl_w(vr6, vr1, vr0);
                        vilvh_w(vr2, vr1, vr0);

                        vilvl_d(vr0, vr6, vr4);
                        vilvh_d(vr1, vr6, vr4);
                        vst(vr0, AO1, (unroll_m * 0 + (i + 2) * 4 - OFFSET) * SIZE);
                        vst(vr1, AO1, (unroll_m * 1 + (i + 2) * 4 - OFFSET) * SIZE);
                        vilvl_d(vr0, vr2, vr5);
                        vilvh_d(vr1, vr2, vr5);
                        vst(vr0, AO1, (unroll_m * 2 + (i + 2) * 4 - OFFSET) * SIZE);
                        vst(vr1, AO1, (unroll_m * 3 + (i + 2) * 4 - OFFSET) * SIZE);
                    }
                } else {
                    for (int i = 0; i < 2; i++) {
                        vbsll_v(vr4, vr3, 0);
                        vgatherqps(vr0, BO2, xr7, ((2 * i) - OFFSET) * SIZE, vr4);
                        vbsll_v(vr4, vr3, 0);
                        vgatherqps(vr1, BO2, xr7, ((2 * i + 1) - OFFSET) * SIZE, vr4);

                        vst(vr0, AO1, (unroll_m * (2 * i) + 2 * 4 - OFFSET) * SIZE);
                        vst(vr1, AO1, (unroll_m * (2 * i + 1) + 2 * 4 - OFFSET) * SIZE);
                    }

                    add_d(BO2, BO2, rax);

                    for (int i = 0; i < 2; i++) {
                        xvpermi_q(xr4, xr3, 0x31);
                        vgatherqps(vr0, BO2, xr7, ((2 * i) - OFFSET) * SIZE, vr4);
                        xvpermi_q(xr4, xr3, 0x31);
                        vgatherqps(vr1, BO2, xr7, ((2 * i + 1) - OFFSET) * SIZE, vr4);

                        vst(vr0, AO1, (unroll_m * (2 * i) + 3 * 4 - OFFSET) * SIZE);
                        vst(vr1, AO1, (unroll_m * (2 * i + 1) + 3 * 4 - OFFSET) * SIZE);
                    }

                    add_d(BO2, BO2, rax);
                }
            }
            addi_d(BO1, BO1, 4 * SIZE);
        }

        addi_d(AO1, AO1, unroll_m * 4 * SIZE);
        addi_d(LL, LL, -1);
        blt(zero, LL, labels[0]);

        L(labels[1]);
        add_d(LL, K, zero);
        andi(LL, LL, 3);
        bge(zero, LL, labels[3]);

        L(labels[2]);
        // do_pack exec only when isTransA is true
        if (!isTransA) {
            assert(0);
        } else {
            if (isLoad1Unmasked) {
                for (int i = 0; i < 2; i++) {
                    reg = (i % 2 == 0) ? BO1 : BO2;
                    vldrepl_w(VReg(i + 1), reg, (0 * 8 - OFFSET) * SIZE);
                    add_d(X_TMP_1, reg, LDA);
                    vldrepl_w(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                    add_d(BO2, reg, t0);
                    vilvl_w(VReg(i + 1), vr0, VReg(i + 1));
                }
                vilvl_d(vr1, vr2, vr1);
                vst(vr1, AO1, (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE);

                for (int i = 0; i < 2; i++) {
                    vldrepl_w(VReg(i + 1), BO2, (0 * 8 - OFFSET) * SIZE);
                    add_d(X_TMP_1, BO2, LDA);
                    vldrepl_w(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                    add_d(BO2, BO2, t0);
                    vilvl_w(VReg(i + 1), vr0, VReg(i + 1));
                }
                vilvl_d(vr1, vr2, vr1);
                vst(vr1, AO1, (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE);
            } /*else if (is_lasx) {
                //
            }*/ else {
                vxor_v(vr4, vr4, vr4);
                add_d(BO2, BO1, rax);

                auto el_cp = [&](int section, int ld_step) {
                    XReg src_addr = section == 0 ? BO1 : BO2;
                    if (ld_step > 0) {
                        add_d(X_TMP_1, src_addr, ld_step == 1 ? LDA : (ld_step == 2 ? t0 : CO1));
                        vldrepl_w(vr1, X_TMP_1, -OFFSET * SIZE);
                    } else
                        vldrepl_w(vr1, src_addr, -OFFSET * SIZE);

                    vstelm_w(vr1, AO1, (ld_step + section * 4 - OFFSET) * SIZE, 0);
                };

                el_cp(0, 0);
                ld_d(X_TMP_0, M.getXReg(), M.getOffset());
                mov_imm(X_TMP_1, 1);
                beq(X_TMP_0, X_TMP_1, labels[5]);
                el_cp(0, 1);
                mov_imm(X_TMP_1, 2);
                beq(X_TMP_0, X_TMP_1, labels[5]);
                el_cp(0, 2);
                mov_imm(X_TMP_1, 3);
                beq(X_TMP_0, X_TMP_1, labels[5]);
                el_cp(0, 3);
                mov_imm(X_TMP_1, 4);
                beq(X_TMP_0, X_TMP_1, labels[5]);
                el_cp(1, 0);
                mov_imm(X_TMP_1, 5);
                beq(X_TMP_0, X_TMP_1, labels[5]);
                el_cp(1, 1);
                mov_imm(X_TMP_1, 6);
                beq(X_TMP_0, X_TMP_1, labels[5]);
                el_cp(1, 2);
                L(labels[5]);

                add_d(BO2, BO2, rax);
            }

            if (unroll_m >= 16) {
                assert(is_lasx);
                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        vldrepl_w(VReg(i + 1), BO2, (0 * 8 - OFFSET) * SIZE);
                        add_d(X_TMP_1, BO2, LDA);
                        vldrepl_w(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                        add_d(BO2, BO2, t0);
                        vilvl_w(VReg(i + 1), vr0, VReg(i + 1));
                    }
                    vilvl_d(vr1, vr2, vr1);
                } else {
                    vbsll_v(vr4, vr3, 0);
                    vgatherqps(vr1, BO2, xr7, (0 * 8 - OFFSET) * SIZE, vr4);
                    add_d(BO2, BO2, rax);
                }
                vst(vr1, AO1, (unroll_m * 0 + 2 * 4 - OFFSET) * SIZE);

                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        vldrepl_w(VReg(i + 1), BO2, (0 * 8 - OFFSET) * SIZE);
                        add_d(X_TMP_1, BO2, LDA);
                        vldrepl_w(vr0, X_TMP_1, (0 * 8 - OFFSET) * SIZE);
                        add_d(BO2, BO2, t0);
                        vilvl_w(VReg(i + 1), vr0, VReg(i + 1));
                    }
                    vilvl_d(vr1, vr2, vr1);
                } else {
                    xvpermi_q(xr4, xr3, 0x31);
                    vgatherqps(vr1, BO2, xr7, (0 * 8 - OFFSET) * SIZE, vr4);
                }
                vst(vr1, AO1, (unroll_m * 0 + 3 * 4 - OFFSET) * SIZE);
            }
            addi_d(BO1, BO1, SIZE);
        }

        addi_d(AO1, AO1, unroll_m * SIZE);
        addi_d(LL, LL, -1);
        blt(zero, LL, labels[2]);

        L(labels[3]);
    }

    // High-level subroutine; does packing if needed, then splits C matrix.
    // Operates on chunks of 16 rows, 6 columns at a time (handling tail
    // cases appropriately).
    // Masking is used for tail cases where M is not divisible by 8.
    void subloop(int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
        std::vector<Label> labels(15);

        if (isTransA) { do_pack(unroll_m, isLoad1Unmasked, isLoad2Unmasked); }

        ld_d(CO1, C.getXReg(), C.getOffset());
        add_d(CO2, CO1, LDC);
        add_d(CO2, CO2, LDC);
        add_d(CO2, CO2, LDC);
        add_imm(X_TMP_0, CO1, unroll_m * SIZE, X_TMP_0);
        st_d(X_TMP_0, C.getXReg(), C.getOffset());
        add_d(BO1, B, zero);
        if (!isTransB) { add_d(BO2, B, LDB3); }

        if (!isTransA) {
            add_imm(AA, A, (unroll_m * 2 - 1 - OFFSET) * SIZE, X_TMP_0);
            mov_imm(X_TMP_0, UNROLL_M);
            ld_d(X_TMP_1, M.getXReg(), M.getOffset());
            blt(X_TMP_0, X_TMP_1, labels[13]);

            ld_d(AA, ORIG_A.getXReg(), ORIG_A.getOffset());
            add_imm(AA, AA, (unroll_m - 1 - OFFSET) * SIZE, X_TMP_0);
            L(labels[13]);
        }

        ld_d(LL, N.getXReg(), N.getOffset());
        st_d(LL, I.getXReg(), I.getOffset());
        if (!isTransA) {
            // If N is too small, skip copy operation
            mov_imm(X_TMP_0, UNROLL_N * 3);
            bge(X_TMP_0, LL, labels[7]);

            // If A is not aligned to cache line
            ld_d(X_TMP_0, FLAG.getXReg(), FLAG.getOffset());
            beqz(X_TMP_0, labels[7]);
        } else {
            mov_imm(X_TMP_0, UNROLL_N);
            blt(LL, X_TMP_0, labels[1]);
        }

        if (!isTransA) {
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, true, true);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        true, true);
            }
        } else {
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, false, false);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            }
        }

        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        addi_d(X_TMP_1, X_TMP_1, -1 * UNROLL_N);
        st_d(X_TMP_1, I.getXReg(), I.getOffset());
        mov_imm(X_TMP_0, UNROLL_N);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        blt(X_TMP_1, X_TMP_0, labels[1]);

        L(labels[0]);
        if (unroll_m == 16) {
            kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                    false, false);
        } else {
            kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                    false, false);
        }
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        addi_d(X_TMP_1, X_TMP_1, -1 * UNROLL_N);
        st_d(X_TMP_1, I.getXReg(), I.getOffset());
        mov_imm(X_TMP_0, UNROLL_N);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bge(X_TMP_1, X_TMP_0, labels[0]);

        L(labels[1]);
        mov_imm(X_TMP_0, 1);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[2]);
        if (unroll_m == 16) {
            kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        b(labels[14]);

        L(labels[2]);
        mov_imm(X_TMP_0, 2);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[3]);
        if (unroll_m == 16) {
            kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        b(labels[14]);

        L(labels[3]);
        mov_imm(X_TMP_0, 3);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[4]);
        if (unroll_m == 16) {
            kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        b(labels[14]);

        L(labels[4]);
        mov_imm(X_TMP_0, 4);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[5]);
        if (unroll_m == 16) {
            kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        b(labels[14]);

        L(labels[5]);
        mov_imm(X_TMP_0, 5);
        ld_d(X_TMP_1, I.getXReg(), I.getOffset());
        bne(X_TMP_1, X_TMP_0, labels[14]);
        if (unroll_m == 16) {
            kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        b(labels[14]);

        if (!isTransA) {
            L(labels[7]);
            mov_imm(X_TMP_0, UNROLL_N);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            blt(X_TMP_1, X_TMP_0, labels[6]);

            L(labels[8]);
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, true, false);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        true, false);
            }
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            addi_d(X_TMP_1, X_TMP_1, -1 * UNROLL_N);
            st_d(X_TMP_1, I.getXReg(), I.getOffset());
            mov_imm(X_TMP_0, UNROLL_N);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bge(X_TMP_1, X_TMP_0, labels[8]);

            L(labels[6]);
            mov_imm(X_TMP_0, 1);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[9]);
            if (unroll_m == 16) {
                kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            b(labels[14]);

            L(labels[9]);
            mov_imm(X_TMP_0, 2);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[10]);
            if (unroll_m == 16) {
                kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            b(labels[14]);

            L(labels[10]);
            mov_imm(X_TMP_0, 3);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[11]);
            if (unroll_m == 16) {
                kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            b(labels[14]);

            L(labels[11]);
            mov_imm(X_TMP_0, 4);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[12]);
            if (unroll_m == 16) {
                kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            b(labels[14]);

            L(labels[12]);
            mov_imm(X_TMP_0, 5);
            ld_d(X_TMP_1, I.getXReg(), I.getOffset());
            bne(X_TMP_1, X_TMP_0, labels[14]);
            if (unroll_m == 16) {
                kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
        }

        L(labels[14]);
        // Compute address for A
        if (!isTransA) {
            add_imm(A, A, unroll_m * SIZE, X_TMP_0);
        } else {
            mov_imm(X_TMP_0, unroll_m);
            mul_d(rax, LDA, X_TMP_0);
            add_d(A, A, rax);
        }

        // Compute next address of BIAS
        if (hasBias) {
            ld_d(X_TMP_0, BIAS.getXReg(), BIAS.getOffset());
            add_imm(X_TMP_0, X_TMP_0, unroll_m * SIZE, X_TMP_1);
            st_d(X_TMP_0, BIAS.getXReg(), BIAS.getOffset());
        }
    }

    void generate() override ATTRIBUTE_OPTIMIZE {

        preamble();

        Label buffer_in_ws, buffer_allocated;

        // Get the registers

        ld_d(s3, ARG_BETA.getXReg(), ARG_BETA.getOffset());
        ld_d(s4, ARG_C.getXReg(), ARG_C.getOffset());
        if (hasBias) ld_d(s5, ARG_BIAS.getXReg(), ARG_BIAS.getOffset());
        ld_d(LDC, ARG_LDC.getXReg(), ARG_LDC.getOffset());

        vldrepl_w(vr0, ARG_ALPHA, 0);
        vldrepl_w(vr1, s3, 0);

        mov_imm(X_TMP_0, STACK_K_CAPACITY);
        blt(X_TMP_0, K, buffer_in_ws);

        // Create buffer and align to 4kB page
        slli_d(rax, K, 2);
        slli_d(rax, rax, 4);
        addi_d(rax, rax, 256);
        sub_d(sp, sp, rax);
        mov_imm(X_TMP_1, -PAGE_4K);
        and_(sp, sp, X_TMP_1);
        b(buffer_allocated);

        L(buffer_in_ws);
        ld_d(sp, ARG_WS.getXReg(), ARG_WS.getOffset());

        L(buffer_allocated);

        st_d(s6, ORIG_SP.getXReg(), ORIG_SP.getOffset());
        st_d(ARG_M, M.getXReg(), M.getOffset());
        st_d(ARG_N, N.getXReg(), N.getOffset());
        st_d(s4, C.getXReg(), C.getOffset());
        if (hasBias) st_d(s5, BIAS.getXReg(), BIAS.getOffset());
        vstelm_w(vr0, ALPHA.getXReg(), ALPHA.getOffset(), 0);
        vstelm_w(vr1, BETA.getXReg(), BETA.getOffset(), 0);
        addi_d(A, A, OFFSET * SIZE);
        addi_d(B, B, OFFSET * SIZE);
        st_d(A, ORIG_A.getXReg(), ORIG_A.getOffset());
        slli_d(LDA, LDA, BASE_SHIFT);
        slli_d(LDB, LDB, BASE_SHIFT);
        slli_d(LDC, LDC, BASE_SHIFT);
        add_d(LDB3, LDB, LDB);
        add_d(LDB3, LDB3, LDB);

        for (int i = 0; i < 8; i++) {
            addi_d(X_TMP_0, zero, i);
            st_w(X_TMP_0, sp, 88 + i * 4);
        }

        /* this code make xr7={0,LDA,LDA2,LDA3} for vgatherqps
         * but in loongarch donot have vgatherqps, so this will
         * make instruction numbers increase. */
        if (isTransA && is_lasx) {
            assert(0);
        }

        // Check A alignment and leading dimension; take copy-based path as
        // needed
        add_d(rax, LDA, zero);
        or_(rax, rax, A);
        andi(rax, rax, 0x1f);
        st_d(rax, FLAG.getXReg(), FLAG.getOffset());
        std::vector<Label> labels(5);

        mov_imm(X_TMP_1, UNROLL_M);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());
        blt(X_TMP_0, X_TMP_1, labels[0]);

        L(labels[1]);
        subloop(UNROLL_M, true, true);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());
        addi_d(X_TMP_0, X_TMP_0, -1 * UNROLL_M);
        st_d(X_TMP_0, M.getXReg(), M.getOffset());
        mov_imm(X_TMP_1, UNROLL_M);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());
        bge(X_TMP_0, X_TMP_1, labels[1]);

        L(labels[0]);
        ld_d(X_TMP_0, M.getXReg(), M.getOffset());
        bge(zero, X_TMP_0, labels[4]);

        if (UNROLL_M > 8) {
            mov_imm(X_TMP_1, 8);
            ld_d(X_TMP_0, M.getXReg(), M.getOffset());
            bge(X_TMP_1, X_TMP_0, labels[2]);

            ld_d(X_TMP_0, M.getXReg(), M.getOffset());
            addi_d(X_TMP_0, X_TMP_0, -8);
            st_d(X_TMP_0, M.getXReg(), M.getOffset());
            xvldrepl_w(VMASK, M.getXReg(), M.getOffset());
            xvld(xr31, MASK.getXReg(), MASK.getOffset());
            xvslt_w(VMASK, xr31, VMASK);

            subloop(16, true, false);
            b(labels[4]);

            L(labels[2]);
            mov_imm(X_TMP_1, 8);
            ld_d(X_TMP_0, M.getXReg(), M.getOffset());
            bne(X_TMP_0, X_TMP_1, labels[3]);
            subloop(8, true, true);
            b(labels[4]);
        }


        L(labels[3]);
        xvldrepl_w(VMASK, M.getXReg(), M.getOffset());
        if (is_lasx) {
            xvld(xr31, MASK.getXReg(), MASK.getOffset());
            xvslt_w(VMASK, xr31, VMASK);
        } //else {
        //    auto xmm_tmp = vr4;

        //}
        subloop(8, false, false);

        L(labels[4]);
        // Restore original stack
        ld_d(sp, ORIG_SP.getXReg(), ORIG_SP.getOffset());

        postamble();
    }

private:
    const char isTransA;
    const char isTransB;
    const bool hasBias;
    const bool is_lasx;
    const int UNROLL_M;
    const int UNROLL_N;
    const bool isBeta0;
    const bool isBetaN;
    const int PREFETCHSIZEA;
    const int PREFETCHSIZEB;

    const XReg ARG_M = abi_param1;
    const XReg ARG_N = abi_param2;
    const XReg K = abi_param3;
    const XReg ARG_ALPHA = abi_param4;

    const XReg ARG_A = abi_param5;
    const XReg ARG_LDA = abi_param6;
    const int stackOffset = STACKSIZE;
    const XReg A = ARG_A;
    const XReg LDA = ARG_LDA;

    const XReg ARG_B = abi_param7; // loongarch has 8 abi_params so ARG_B is abi_param7
    const XReg ARG_LDB = abi_param8; // loongarch has 8 abi_params so ARG_LDB is abi_param8
    const Address ARG_BETA = ptr_a(sp, stackOffset); // from ARG_BETA the param in sp
    const Address ARG_C = ptr_a(sp, 8 + stackOffset);
    const Address ARG_LDC = ptr_a(sp, 16 + stackOffset);
    const Address ARG_BIAS = ptr_a(sp, 24 + stackOffset);
    const Address ARG_WS = ptr_a(sp, 32 + stackOffset);

    const XReg B = ARG_B;
    const XReg LDB = ARG_LDB;
    const XReg rax = t1; // for calc in loongarch
    const XReg LDC = t2;
    const XReg LL = t3;
    const XReg AO1 = abi_param2;
    const XReg BO1 = abi_param4;
    const XReg BO2 = t4;
    const XReg CO1 = t5;
    const XReg CO2 = t6;
    const XReg LDB3 = t7;
    const XReg LDA4 = abi_param1;
    const XReg AA = t8;
    const XReg BIAS1 = abi_param1;

    const Address M = ptr_a(sp, 0);
    const Address N = ptr_a(sp, 8);
    const Address FLAG = ptr_a(sp, 16);
    const Address I = ptr_a(sp, 24);
    const Address C = ptr_a(sp, 32);
    const Address BIAS = ptr_a(sp, 40);
    const Address ALPHA = ptr_a(sp, 48);
    const Address BETA = ptr_a(sp, 64);
    const Address ORIG_A = ptr_a(sp, 80);
    const Address MASK = ptr_a(sp, 88);
    const Address ORIG_SP = ptr_a(sp, 120);

    const XVReg VALPHA = xr1;
    const XVReg VBETA = xr2;
    const XVReg VMASK = xr3;
    const XVReg VBIAS1 = xr2;
    const XVReg VBIAS2 = xr4;
};

xbyak_gemm_t *get_xbyak_gemm(
        bool isTransA, bool isTransB, float beta, bool hasBias) {
    auto beta_idx = [](float beta) {
        return (beta == 0.0) ? 0 : (beta == 1.0 ? 1 : 2);
    };

    // Kernel table [isTransA][isTransB][hasBias][beta (0, 1, other)]
    static xbyak_gemm_t *kernel_table[2][2][2][3];
    static std::once_flag initialized;
    dnnl_status_t st = dnnl_success;
    std::call_once(initialized, [&] {
        for (bool isTransA : {false, true})
            for (bool isTransB : {false, true})
                for (bool hasBias : {false, true})
                    for (float beta : {0.0f, 1.0f, 2.0f}) {
                        // nocopy sgemm with bias for beta != 0.0 is not supported
                        if (hasBias && beta != 0.0) continue;
                        auto &kern = kernel_table[isTransA][isTransB][hasBias]
                                                 [beta_idx(beta)];

                        kern = new xbyak_gemm_t(
                                isTransA, isTransB, beta, hasBias);
                        if (kern->create_kernel() != dnnl_success) {
                            st = dnnl_runtime_error;
                            return;
                        }
                    }
    });

    return (st == dnnl_success)
            ? kernel_table[isTransA][isTransB][hasBias][beta_idx(beta)]
            : nullptr;
}

dnnl_status_t sgemm_nocopy_driver(const char *transa, const char *transb,
        dim_t m, dim_t n, dim_t k, const float *alpha, const float *a,
        dim_t lda, const float *b, dim_t ldb, const float *beta, float *c,
        dim_t ldc, const float *bias, float *ws) {

    bool isTransA = (*transa == 'T' || *transa == 't');
    bool isTransB = (*transb == 'T' || *transb == 't');

    dim_t Bm, sizeM, Bn, sizeN, Bk, sizeK;

    dim_t i, j;

    if ((m <= 0) || (n <= 0)) return dnnl_success;

    if ((k <= 0) || (alpha[0] == 0.)) {

        if (beta[0] == 0.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] = 0.0;
        } else if (beta[0] != 1.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] *= beta[0];
        }

        return dnnl_success;
    }

    assert(IMPLICATION(bias != nullptr, *beta == 0.0));

    // XXX: this happens on every thread...
    bool hasBias = (bias != nullptr);
    auto ker_bn = get_xbyak_gemm(isTransA, isTransB, *beta, hasBias);
    auto ker_b1 = get_xbyak_gemm(isTransA, isTransB, 1.0, false);
    auto ker_b0 = get_xbyak_gemm(isTransA, isTransB, 0.0, false);
    if (utils::any_null(ker_bn, ker_b1, ker_b0)) return dnnl_runtime_error;

    dim_t BM = 4032;
    dim_t BN = isTransA ? 96 : 48;
    dim_t BK = isTransB ? 96 : 256;
    const float *curA, *curB, *curBias = nullptr;
    float *curC;

    for (Bk = 0; Bk < k; Bk += sizeK) {
        sizeK = k - Bk;
        if (sizeK >= BK * 2)
            sizeK = BK;
        else {
            if (sizeK > BK) sizeK = (sizeK + 1) / 2;
        }

        for (Bm = 0; Bm < m; Bm += sizeM) {
            sizeM = m - Bm;
            if (sizeM >= BM * 2)
                sizeM = BM;
            else {
                if (sizeM > BM + BM / 2) sizeM = (sizeM + 1) / 2;
            }

            for (Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN >= BN * 2)
                    sizeN = BN;
                else {
                    if (sizeN > BN + BN / 2) sizeN = (sizeN + 1) / 2;
                }

                if (!isTransA) {
                    curA = a + Bm + Bk * lda;
                } else {
                    curA = a + Bk + Bm * lda;
                }
                if (!isTransB) {
                    curB = b + Bk + Bn * ldb;
                } else {
                    curB = b + Bn + Bk * ldb;
                }
                curC = c + Bm + (size_t)Bn * ldc;
                if (bias != nullptr) {
                    if (Bk == 0) {
                        curBias = bias + Bm;
                    } else {
                        curBias = nullptr;
                    }
                }
                if (Bk == 0) {
                    if (*beta == 0.0 && bias == nullptr)
                        (*ker_b0)(sizeM, sizeN, sizeK, alpha, curA, lda, curB,
                                ldb, beta, curC, ldc, curBias, ws);
                    else
                        (*ker_bn)(sizeM, sizeN, sizeK, alpha, curA, lda, curB,
                                ldb, beta, curC, ldc, curBias, ws);
                } else {
                    (*ker_b1)(sizeM, sizeN, sizeK, alpha, curA, lda, curB, ldb,
                            beta, curC, ldc, curBias, ws);
                }
            }
        }
    }
    msan_unpoison_matrix(c, m, n, ldc, sizeof(*c));

    return dnnl_success;
}

} // namespace lasx_gemm_f32

dnnl_status_t jit_lasx_gemm_f32(int nthrs, const char *transa,
        const char *transb, const dim_t *p_m, const dim_t *p_n,
        const dim_t *p_k, const float *p_alpha, const float *A,
        const dim_t *p_lda, const float *B, const dim_t *p_ldb,
        const float *p_beta, float *C, const dim_t *p_ldc, const float *bias) {

    using namespace dnnl::impl::utils;
    using namespace lasx_gemm_f32;
    using namespace gemm_utils;

    if (*p_beta != 0 && bias)
        return ref_gemm(transa, transb, p_m, p_n, p_k, p_alpha, A, p_lda, B,
                p_lda, p_beta, C, p_ldc, bias);

    int nthr_max = dnnl_get_current_num_threads();
    int nthr_to_use = nstl::min(nthrs, nthr_max);

    dim_t m = *p_m;
    dim_t n = *p_n;
    dim_t k = *p_k;
    dim_t lda = *p_lda;
    dim_t ldb = *p_ldb;
    dim_t ldc = *p_ldc;
    float beta = *p_beta;
    dim_t MB, NB, KB;

    int nthr_m = 1, nthr_n = 1, nthr_k = 1, nthr_mn = 1;

    // Determine threading partitioning
    calc_nthr_nocopy_avx(
            m, n, k, nthr_to_use, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!dnnl_thr_syncable(), nthr_k == 1));

    nthr_to_use = nthr_m * nthr_n * nthr_k;

    nthr_mn = nthr_m * nthr_n;

    unsigned char *ompstatus_ = nullptr;
    unsigned char volatile *ompstatus = nullptr;

    float *c_buffers = nullptr;
    float *ws_buffers = nullptr;

    if (nthr_k > 1) {
        ompstatus_ = (unsigned char *)malloc(
                nthr_to_use * CACHE_LINE_SIZE, CACHE_LINE_SIZE);
        if (!ompstatus_) return dnnl_out_of_memory;

        ompstatus = (unsigned char volatile *)ompstatus_;
        assert(ompstatus);

        for (int i = 0; i < nthr_to_use; i++)
            ompstatus[i * CACHE_LINE_SIZE] = 0;

        c_buffers = (float *)malloc(
                sizeof(*c_buffers) * nthr_m * nthr_n * (nthr_k - 1) * MB * NB,
                PAGE_4K);
        if (!c_buffers) {
            free(ompstatus_);
            return dnnl_out_of_memory;
        }
    }

    const size_t ws_elems_per_thr
            = (size_t)rnd_up(div_up(k, nthr_k), KB) * 16 + 64;
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(float), PAGE_4K);
    if (k > STACK_K_CAPACITY) {
        ws_buffers = (float *)malloc(nthr_to_use * ws_size_per_thr, PAGE_4K);
        if (!ws_buffers) {
            free(ompstatus_);
            free(c_buffers);
            return dnnl_out_of_memory;
        }
    }

    if (nthr_to_use == 1) {
        auto status = sgemm_nocopy_driver(transa, transb, m, n, k, p_alpha, A,
                lda, B, ldb, p_beta, C, ldc, bias, ws_buffers);
        if (ws_buffers) free(ws_buffers);
        return status;
    }

    // Always use the maximum number of threads to avoid OMP overhead that can
    // occur due to change thread counts.
    int nthr_spawn = dnnl_thr_syncable() ? nthr_max : nthr_to_use;

    std::atomic<dnnl_status_t> st(dnnl_success);
    parallel(nthr_spawn, [&](int ithr, int nthr) {
        assert(nthr_spawn == nthr);
        MAYBE_UNUSED(nthr);

        int ithr_m, ithr_n, ithr_k, ithr_mn;
        dim_t m_from, m_to, myM;
        dim_t n_from, n_to, myN;
        dim_t k_from, k_to, myK;
        int cbase, ibase;
        const float *myA, *myB, *myBias = nullptr;
        float *myC = C, myBeta;
        float *ws = ws_buffers
                ? ws_buffers + ithr * ws_size_per_thr / sizeof(float)
                : nullptr;
        dim_t ld = ldc;

        int sum_later = (nthr < nthr_m * nthr_n * nthr_k);

        if (ithr < nthr_m * nthr_n * nthr_k) {

            ithr_mn = ithr % nthr_mn;
            ithr_m = ithr_mn % nthr_m;
            ithr_n = ithr_mn / nthr_m;
            ithr_k = ithr / nthr_mn;

            /* swap ithr_k for performance improvement */
            if (ithr_k == 0)
                ithr_k = nthr_k - 1;
            else if (ithr_k == nthr_k - 1)
                ithr_k = 0;

            m_from = MB * (ithr_m);
            m_to = MB * (ithr_m + 1);
            if (m_to > m) m_to = m;
            myM = m_to - m_from;

            n_from = NB * (ithr_n);
            n_to = NB * (ithr_n + 1);
            if (n_to > n) n_to = n;
            myN = n_to - n_from;

            k_from = KB * (ithr_k);
            k_to = KB * (ithr_k + 1);
            if (k_to > k) k_to = k;
            myK = k_to - k_from;

            cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);
            ibase = (ithr_m + nthr_m * ithr_n) * nthr_k;

            if ((myM > 0) && (myN > 0)) {

                if (*transa == 'N' || *transa == 'n') {
                    myA = &(A[m_from + k_from * lda]);
                } else {
                    myA = &(A[k_from + m_from * lda]);
                }
                if (*transb == 'N' || *transb == 'n') {
                    myB = &(B[k_from + n_from * ldb]);
                } else {
                    myB = &(B[n_from + k_from * ldb]);
                }
                if (ithr_k == 0) {
                    myC = &(C[m_from + n_from * ldc]);
                    myBeta = beta;
                    ld = ldc;
                    if (bias) myBias = &(bias[m_from]);
                } else {
                    myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
                    myBeta = 0.0;
                    ld = MB;
                    myBias = nullptr;
                }

                dnnl_status_t st_thr = sgemm_nocopy_driver(transa, transb, myM,
                        myN, myK, p_alpha, myA, lda, myB, ldb, &myBeta, myC, ld,
                        myBias, ws);
                if (st_thr != dnnl_success) {
                    st = st_thr;
                    return;
                }

                if (nthr_k > 1 && !sum_later)
                    ompstatus[(ibase + ithr_k) * CACHE_LINE_SIZE] = 1;
            }

            if (nthr_k > 1 && !sum_later) {

                // sum matrices partitioned along K dimension
                dim_t n1, n2;

                partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                if (ithr_k > 0) {

                    myC = c_buffers + MB * NB * (cbase + ithr_k - 1) + n1 * MB;
                    /* need to wait until main thread finishes */
                    while (ompstatus[ibase * CACHE_LINE_SIZE] != 1) {};

                    /* my cache is hot */
                    sum_two_matrices(myM, n2, myC, MB,
                            &C[m_from + (n_from + n1) * ldc], ldc);
                }

                for (int ik = 1; ik < nthr_k; ++ik) {
                    if (ik != ithr_k) {

                        myC = c_buffers + MB * NB * (cbase + ik - 1) + n1 * MB;

                        while (ompstatus[(ibase + ik) * CACHE_LINE_SIZE] != 1) {
                        };

                        sum_two_matrices(myM, n2, myC, MB,
                                &C[m_from + (n_from + n1) * ldc], ldc);
                    }
                }
            }
        }
    });
    CHECK(st);

    // handle C summation later
    if (nthr_k > 1 && ompstatus[0] == 0) {

        parallel(nthr_spawn, [&](int ithr, int nthr) {
            assert(nthr_spawn == nthr);
            MAYBE_UNUSED(nthr);

            int ithr_m, ithr_n, ithr_k, ithr_mn;
            dim_t m_from, m_to, myM;
            dim_t n_from, n_to, myN;
            int cbase;
            float *myC = C;

            if (ithr < nthr_m * nthr_n * nthr_k) {

                ithr_mn = ithr % nthr_mn;
                ithr_m = ithr_mn % nthr_m;
                ithr_n = ithr_mn / nthr_m;
                ithr_k = ithr / nthr_mn;

                /* swap ithr_k for performance improvement */
                if (ithr_k == 0)
                    ithr_k = nthr_k - 1;
                else if (ithr_k == nthr_k - 1)
                    ithr_k = 0;

                m_from = MB * (ithr_m);
                m_to = MB * (ithr_m + 1);
                if (m_to > m) m_to = m;
                myM = m_to - m_from;

                n_from = NB * (ithr_n);
                n_to = NB * (ithr_n + 1);
                if (n_to > n) n_to = n;
                myN = n_to - n_from;

                cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

                if (nthr_k > 1) {
                    // sum matrices partitioned along K dimension
                    dim_t n1, n2;

                    partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                    if (ithr_k > 0) {

                        myC = c_buffers + MB * NB * (cbase + ithr_k - 1)
                                + n1 * MB;

                        /* my cache is hot */
                        sum_two_matrices(myM, n2, myC, MB,
                                &C[m_from + (n_from + n1) * ldc], ldc);
                    }

                    for (int ik = 1; ik < nthr_k; ++ik) {
                        if (ik != ithr_k) {

                            myC = c_buffers + MB * NB * (cbase + ik - 1)
                                    + n1 * MB;

                            sum_two_matrices(myM, n2, myC, MB,
                                    &C[m_from + (n_from + n1) * ldc], ldc);
                        }
                    }
                }
            }
        });
    }

    free(c_buffers);
    free(ompstatus_);
    free(ws_buffers);

    return dnnl_success;
}

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
