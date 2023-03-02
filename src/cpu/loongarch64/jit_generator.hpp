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

#ifndef CPU_LOONGARCH64_JIT_GENERATOR_HPP
#define CPU_LOONGARCH64_JIT_GENERATOR_HPP
#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/loongarch64/cpu_isa_traits.hpp"

#include "cpu/jit_utils/jit_utils.hpp"

#if defined(_WIN32) && !defined(__GNUC__)
#define STRUCT_ALIGN(al, ...) __declspec(align(al)) __VA_ARGS__
#else
#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))
#endif

#if GCC_WA_NO_TREE_DOMINATOR_OPTS
#define ATTRIBUTE_OPTIMIZE __attribute__((optimize("no-tree-dominator-opts")))
#else
#define ATTRIBUTE_OPTIMIZE
#endif

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return STRINGIFY(jit_name); } \
    const char *source_file() const override { return __FILE__; }

#define IMM8_MIN_VALUE     -128
#define IMM8_MAX_VALUE      127
#define IMM9_MIN_VALUE     -256
#define IMM9_MAX_VALUE      255
#define IMM10_MIN_VALUE    -512
#define IMM10_MAX_VALUE     511
#define IMM11_MIN_VALUE    -1024
#define IMM11_MAX_VALUE     1023
#define IMM12_MIN_VALUE    -2048
#define IMM12_MAX_VALUE     2047
#define UIMM12_MAX_VALUE    4095
#define IMM14_MIN_VALUE    -8192
#define IMM14_MAX_VALUE     8191
#define UIMM14_MAX_VALUE    16383

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

// TODO: move this to jit_generator class?
namespace {

typedef enum {
    MAX_CODE_SIZE = 256 * 1024,
} max_code_size_t;

// TODO: move this somewhere else? Although this is only used by jit kernels
// (Roma)
static inline int float2int(float x) {
    return utils::bit_cast<int>(x);
}

constexpr Xbyak_loongarch64::Operand::Code abi_save_gpr_regs[] = {
        Xbyak_loongarch64::Operand::s0,
        Xbyak_loongarch64::Operand::s1,
        Xbyak_loongarch64::Operand::s2,
        Xbyak_loongarch64::Operand::s3,
        Xbyak_loongarch64::Operand::s4,
        Xbyak_loongarch64::Operand::s5,
        Xbyak_loongarch64::Operand::s6,
        Xbyak_loongarch64::Operand::s7,
        Xbyak_loongarch64::Operand::s8,
};

constexpr Xbyak_loongarch64::Operand::Code abi_save_fpr_regs[] = {
        Xbyak_loongarch64::Operand::f24,
        Xbyak_loongarch64::Operand::f25,
        Xbyak_loongarch64::Operand::f26,
        Xbyak_loongarch64::Operand::f27,
        Xbyak_loongarch64::Operand::f28,
        Xbyak_loongarch64::Operand::f29,
        Xbyak_loongarch64::Operand::f30,
        Xbyak_loongarch64::Operand::f31,
};

static const Xbyak_loongarch64::XReg abi_param1(Xbyak_loongarch64::Operand::a0),
        abi_param2(Xbyak_loongarch64::Operand::a1), abi_param3(Xbyak_loongarch64::Operand::a2),
        abi_param4(Xbyak_loongarch64::Operand::a3), abi_param5(Xbyak_loongarch64::Operand::a4),
        abi_param6(Xbyak_loongarch64::Operand::a5), abi_param7(Xbyak_loongarch64::Operand::a6),
        abi_param8(Xbyak_loongarch64::Operand::a7), abi_not_param1(Xbyak_loongarch64::Operand::t8);
} // namespace

class jit_generator : public Xbyak_loongarch64::CodeGenerator, public c_compatible {
public:
    using c_compatible::operator new;
    using c_compatible::operator new[];
    using c_compatible::operator delete;
    using c_compatible::operator delete[];

private:
    const size_t xreg_len = 8;          // 8 Byte
    const size_t vreg_len_preserve = 8; // Only bottom 8byte must be preserved.
    const size_t vreg_to_preserve = 8;  // VREG24 - VREG31

    const size_t num_abi_save_gpr_regs
            = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t num_abi_save_fpr_regs
            = sizeof(abi_save_fpr_regs) / sizeof(abi_save_fpr_regs[0]);

    const size_t preserved_stack_size = xreg_len * (2 + num_abi_save_gpr_regs) //fp+ra
            + vreg_len_preserve * vreg_to_preserve;

    const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * xreg_len
            + (2 + num_abi_save_fpr_regs) * vreg_len_preserve; // fp+ra+f24~f31

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    const Xbyak_loongarch64::XReg W_TMP_0 = s0;
    const Xbyak_loongarch64::XReg W_TMP_1 = s1;
    const Xbyak_loongarch64::XReg W_TMP_2 = s2;
    const Xbyak_loongarch64::XReg W_TMP_3 = s3;
    const Xbyak_loongarch64::XReg W_TMP_4 = s4;

    const Xbyak_loongarch64::XReg X_TMP_0 = s0;
    const Xbyak_loongarch64::XReg X_TMP_1 = s1;
    const Xbyak_loongarch64::XReg X_TMP_2 = s2;
    const Xbyak_loongarch64::XReg X_TMP_3 = s3;
    const Xbyak_loongarch64::XReg X_TMP_4 = s4;
    const Xbyak_loongarch64::XReg X_DEFAULT_ADDR = s5;
    const Xbyak_loongarch64::XReg X_SP = s6;
    const Xbyak_loongarch64::XReg X_TRANSLATOR_STACK = s7;

    const std::vector<Xbyak_loongarch64::XReg> x_tmp_vec
            = {X_TMP_0, X_TMP_1, X_TMP_2, X_TMP_3, X_TMP_4};
    const int x_tmp_vec_size = x_tmp_vec.size();

    Xbyak_loongarch64::XReg param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;
    constexpr static size_t translator_stack_offset = 1024 * 128;
    const Xbyak_loongarch64::XReg reg_EVEX_max_8b_offt = t0;
    constexpr static uint32_t DUMMY_IDX = 99;

    inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

    void preamble() {
        int i = 1;
        addi_d(sp, sp, -1*preserved_stack_size);
        st_d(ra, sp, preserved_stack_size-8*(i++));
        st_d(fp, sp, preserved_stack_size-8*(i++));
        add_d(fp, sp, zero);

        for (size_t j = 0; j < num_abi_save_fpr_regs; ++j) {
            fst_d(Xbyak_loongarch64::XReg(abi_save_fpr_regs[j]), sp, preserved_stack_size-8*(i++));
        }
        for (size_t k = 0; k < num_abi_save_gpr_regs; ++k) {
            st_d(Xbyak_loongarch64::XReg(abi_save_gpr_regs[k]), sp, preserved_stack_size-8*(i++));
        }

        add_d(X_SP, sp, zero);
        sub_imm(X_TRANSLATOR_STACK, X_SP, translator_stack_offset, X_TMP_0);
    }

    void postamble() {
        int i = 3;
        add_d(sp, fp, zero);

        for (size_t j = 0; j < num_abi_save_fpr_regs; ++j) {
            fld_d(Xbyak_loongarch64::XReg(abi_save_fpr_regs[j]), sp, preserved_stack_size-8*(i++));
        }
        for (size_t k = 0; k < num_abi_save_gpr_regs; ++k) {
            ld_d(Xbyak_loongarch64::XReg(abi_save_gpr_regs[k]), sp, preserved_stack_size-8*(i++));
        }

        ld_d(ra, sp, preserved_stack_size-8);
        ld_d(fp, sp, preserved_stack_size-8*2);
        addi_d(sp, sp, preserved_stack_size);

        jirl(zero, ra, 0);
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak_loongarch64::Label &label) { Xbyak_loongarch64::CodeGenerator::L(label); }

    void L_aligned(Xbyak_loongarch64::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    void uni_ld_b(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
	        ldx_b(rd, rj, X_TMP_2);
	        return;
        }
        ld_b(rd, rj, simm);
    }

    void uni_ld_bu(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_bu(rd, rj, X_TMP_2);
            return;
        }
        ld_bu(rd, rj, simm);
    }

    void uni_ld_h(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_h(rd, rj, X_TMP_2);
            return;
        }
        ld_h(rd, rj, simm);
    }

    void uni_ld_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_w(rd, rj, X_TMP_2);
            return;
        }
        ld_w(rd, rj, simm);
    }

    void uni_ld_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            ldx_d(rd, rj, X_TMP_2);
            return;
        }
        ld_d(rd, rj, simm);
    }

    void uni_preld(const uint32_t hint, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            preldx(hint, rj, X_TMP_2);
            return;
        }
        preld(hint, rj, simm);
    }

    void uni_fld_s(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fldx_s(rd, rj, X_TMP_2);
            return;
        }
        fld_s(rd, rj, simm);
    }

    void uni_fld_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fldx_d(rd, rj, X_TMP_2);
            return;
        }
        fld_d(rd, rj, simm);
    }

    void uni_ll_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ll_w(rd, X_TMP_2, 0);
            return;
        }
        ll_w(rd, rj, simm);
    }

    void uni_ll_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ll_d(rd, X_TMP_2, 0);
            return;
        }
        ll_d(rd, rj, simm);
    }

    void uni_ldptr_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ldptr_w(rd, X_TMP_2, 0);
            return;
        }
        ldptr_w(rd, rj, simm);
    }

    void uni_ldptr_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            ldptr_d(rd, X_TMP_2, 0);
            return;
        }
        ldptr_d(rd, rj, simm);
    }

    void uni_st_b(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_b(rd, rj, X_TMP_2);
            return;
        }
        st_b(rd, rj, simm);
    }

    void uni_st_h(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_h(rd, rj, X_TMP_2);
            return;
        }
        st_h(rd, rj, simm);
    }

    void uni_st_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_w(rd, rj, X_TMP_2);
            return;
        }
        st_w(rd, rj, simm);
    }

    void uni_st_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            stx_d(rd, rj, X_TMP_2);
            return;
        }
        st_d(rd, rj, simm);
    }

    void uni_fst_s(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fstx_s(rd, rj, X_TMP_2);
            return;
        }
        fst_s(rd, rj, simm);
    }

    void uni_fst_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            fstx_d(rd, rj, X_TMP_2);
            return;
        }
        fst_d(rd, rj, simm);
    }

    void uni_sc_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            sc_w(rd, X_TMP_2, 0);
            return;
        }
        sc_w(rd, rj, simm);
    }

    void uni_sc_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            sc_d(rd, X_TMP_2, 0);
            return;
        }
        sc_d(rd, rj, simm);
    }

    void uni_stptr_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            stptr_w(rd, X_TMP_2, 0);
            return;
        }
        stptr_w(rd, rj, simm);
    }

    void uni_stptr_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM14_MAX_VALUE || simm < IMM14_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            stptr_d(rd, X_TMP_2, 0);
            return;
        }
        stptr_d(rd, rj, simm);
    }

    void uni_xvld(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            vldx(vd, rj, X_TMP_2);
            return;
        }
        vld(vd, rj, simm);
    }

    void uni_xvld(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            xvldx(xd, rj, X_TMP_2);
            return;
        }
        xvld(xd, rj, simm);
    }

    void uni_xvld(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &r1,
            const Xbyak_loongarch64::XReg &r2, const int32_t simm) {
        if (0 != simm) {
            add_imm(X_TMP_2, r2, simm, X_TMP_0);
            xvldx(xd, r1, X_TMP_2);
            return;
        }
        xvldx(xd, r1, r2);
    }

    void uni_xvst(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            vstx(vd, rj, X_TMP_2);
            return;
        }
        vst(vd, rj, simm);
    }

    void uni_xvst(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            mov_imm(X_TMP_2, simm);
            xvstx(xd, rj, X_TMP_2);
            return;
        }
        xvst(xd, rj, simm);
    }

    void uni_xvst(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &r1,
            const Xbyak_loongarch64::XReg &r2, const int32_t simm) {
        if (0 != simm) {
            add_imm(X_TMP_2, r2, simm, X_TMP_0);
            xvstx(xd, r1, X_TMP_2);
            return;
        }
        xvstx(xd, r1, r2);
    }

    void uni_vpxor(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
            const Xbyak_loongarch64::VReg &vk) {
        vxor_v(vd, vj, vk);
    }

    void uni_vpxor(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
            const Xbyak_loongarch64::XVReg &xk) {
        xvxor_v(xd, xj, xk);
    }

    void uni_xvldrepl_b(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vldrepl_b(vd, X_TMP_2, 0);
            return;
        }
        vldrepl_b(vd, rj, simm);
    }

    void uni_xvldrepl_h(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vldrepl_h(vd, X_TMP_2, 0);
            return;
        }
        vldrepl_h(vd, rj, simm);
    }

    void uni_xvldrepl_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vldrepl_w(vd, X_TMP_2, 0);
            return;
        }
        vldrepl_w(vd, rj, simm);
    }

    void uni_xvldrepl_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vldrepl_d(vd, X_TMP_2, 0);
            return;
        }
        vldrepl_d(vd, rj, simm);
    }

    void uni_xvldrepl_b(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvldrepl_b(xd, X_TMP_2, 0);
            return;
        }
        xvldrepl_b(xd, rj, simm);
    }

    void uni_xvldrepl_h(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvldrepl_h(xd, X_TMP_2, 0);
            return;
        }
        xvldrepl_h(xd, rj, simm);
    }

    void uni_xvldrepl_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvldrepl_w(xd, X_TMP_2, 0);
            return;
        }
        xvldrepl_w(xd, rj, simm);
    }

    void uni_xvldrepl_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM12_MAX_VALUE || simm < IMM12_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvldrepl_d(xd, X_TMP_2, 0);
            return;
        }
        xvldrepl_d(xd, rj, simm);
    }

    // we use the real offset
    void uni_xvstelm_b(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM8_MAX_VALUE || simm < IMM8_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vstelm_b(vd, X_TMP_2, 0, idx);
            return;
        }
        vstelm_b(vd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.h use si8 << 1)
    void uni_xvstelm_h(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM9_MAX_VALUE || simm < IMM9_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vstelm_h(vd, X_TMP_2, 0, idx);
            return;
        }
        vstelm_h(vd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.w use si8 << 2)
    void uni_xvstelm_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM10_MAX_VALUE || simm < IMM10_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vstelm_w(vd, X_TMP_2, 0, idx);
            return;
        }
        vstelm_w(vd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.d use si8 << 3)
    void uni_xvstelm_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM11_MAX_VALUE || simm < IMM11_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vstelm_d(vd, X_TMP_2, 0, idx);
            return;
        }
        vstelm_d(vd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.w use si8 << 2)
    void uni_xvstelm_w0(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM10_MAX_VALUE || simm < IMM10_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vstelm_w(vd, X_TMP_2, 0, 0);
            return;
        }
        vstelm_w(vd, rj, simm, 0);
    }

    // we use the real offset(but the xvstelm.d use si8 << 3)
    void uni_xvstelm_d0(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM11_MAX_VALUE || simm < IMM11_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            vstelm_d(vd, X_TMP_2, 0, 0);
            return;
        }
        vstelm_d(vd, rj, simm, 0);
    }

    // we use the real offset
    void uni_xvstelm_b(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM8_MAX_VALUE || simm < IMM8_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvstelm_b(xd, X_TMP_2, 0, idx);
            return;
        }
        xvstelm_b(xd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.h use si8 << 1)
    void uni_xvstelm_h(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM9_MAX_VALUE || simm < IMM9_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvstelm_h(xd, X_TMP_2, 0, idx);
            return;
        }
        xvstelm_h(xd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.w use si8 << 2)
    void uni_xvstelm_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM10_MAX_VALUE || simm < IMM10_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvstelm_w(xd, X_TMP_2, 0, idx);
            return;
        }
        xvstelm_w(xd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.d use si8 << 3)
    void uni_xvstelm_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm, const uint32_t idx) {
        if (simm > IMM11_MAX_VALUE || simm < IMM11_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvstelm_d(xd, X_TMP_2, 0, idx);
            return;
        }
        xvstelm_d(xd, rj, simm, idx);
    }

    // we use the real offset(but the xvstelm.w use si8 << 2)
    void uni_xvstelm_w0(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM10_MAX_VALUE || simm < IMM10_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvstelm_w(xd, X_TMP_2, 0, 0);
            return;
        }
        xvstelm_w(xd, rj, simm, 0);
    }

    // we use the real offset(but the xvstelm.d use si8 << 3)
    void uni_xvstelm_d0(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const int32_t simm) {
        if (simm > IMM11_MAX_VALUE || simm < IMM11_MIN_VALUE) {
            add_imm(X_TMP_2, rj, simm, X_TMP_2);
            xvstelm_d(xd, X_TMP_2, 0, 0);
            return;
        }
        xvstelm_d(xd, rj, simm, 0);
    }

    void uni_xvreplve0_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj) {
        xvreplve0_w(xd, xj);
    }

    void uni_xvreplve0_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj) {
        vreplvei_w(vd, vj, 0);
    }

    void uni_xvand_v(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvand_v(xd, xj, xk);
    }

    void uni_xvand_v(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vand_v(vd, vj, vk);
    }

    void uni_xvnor_v(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvnor_v(xd, xj, xk);
    }

    void uni_xvnor_v(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vnor_v(vd, vj, vk);
    }

    void uni_xvpickev_h(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvpickev_h(xd, xj, xk);
    }

    void uni_xvpickev_h(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vpickev_h(vd, vj, vk);
    }

    void uni_xvpickev_b(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvpickev_b(xd, xj, xk);
    }

    void uni_xvpickev_b(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vpickev_b(vd, vj, vk);
    }

    void uni_xvffint_s_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj) {
        xvffint_s_w(xd, xj);
    }

    void uni_xvffint_s_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj) {
        vffint_s_w(vd, vj);
    }

    void uni_xvftint_w_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj) {
        xvftint_w_s(xd, xj);
    }

    void uni_xvftint_w_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj) {
        vftint_w_s(vd, vj);
    }

    void uni_xvinsgr2vr_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj,
            const uint32_t ui) {
        xvinsgr2vr_w(xd, rj, ui);
    }

    void uni_xvinsgr2vr_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj,
            const uint32_t ui) {
        vinsgr2vr_w(vd, rj, ui);
    }

    void uni_replgr2vr_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj) {
        xvreplgr2vr_w(xd, rj);
    }

    void uni_replgr2vr_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj) {
        vreplgr2vr_w(vd, rj);
    }

    void uni_replgr2vr_b(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XReg &rj) {
        xvreplgr2vr_b(xd, rj);
    }

    void uni_replgr2vr_b(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::XReg &rj) {
        vreplgr2vr_b(vd, rj);
    }

    void uni_xvseq_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvseq_d(xd, xj, xk);
    }

    void uni_xvseq_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vseq_d(vd, vj, vk);
    }

    void uni_fmax_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfmax_s(xd, xj, xk);
    }

    void uni_fmax_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfmax_s(vd, vj, vk);
    }

    void uni_fmax_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfmax_d(xd, xj, xk);
    }

    void uni_fmax_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfmax_d(vd, vj, vk);
    }

    void uni_fmin_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfmin_s(xd, xj, xk);
    }

    void uni_fmin_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfmin_s(vd, vj, vk);
    }

    void uni_fmin_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfmin_d(xd, xj, xk);
    }

    void uni_fmin_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfmin_d(vd, vj, vk);
    }

    void uni_fadd_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfadd_s(xd, xj, xk);
    }

    void uni_fadd_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfadd_s(vd, vj, vk);
    }

    void uni_fadd_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfadd_d(xd, xj, xk);
    }

    void uni_fadd_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfadd_d(vd, vj, vk);
    }

    void uni_fsub_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfsub_s(xd, xj, xk);
    }

    void uni_fsub_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfsub_s(vd, vj, vk);
    }

    void uni_fmul_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfmul_s(xd, xj, xk);
    }

    void uni_fmul_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfmul_s(vd, vj, vk);
    }

    void uni_fmul_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfmul_d(xd, xj, xk);
    }

    void uni_fmul_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfmul_d(vd, vj, vk);
    }

    void uni_fdiv_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfdiv_s(xd, xj, xk);
    }

    void uni_fdiv_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfdiv_s(vd, vj, vk);
    }

    void uni_bsll_v(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const uint32_t ui5) {
        xvbsll_v(xd, xj, ui5);
    }

    void uni_bsll_v(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const uint32_t ui5) {
        vbsll_v(vd, vj, ui5);
    }

    void uni_fcmp_caf_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_caf_s(xd, xj, xk);
    }

    void uni_fcmp_caf_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_caf_s(vd, vj, vk);
    }

    void uni_fcmp_cun_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cun_s(xd, xj, xk);
    }

    void uni_fcmp_cun_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cun_s(vd, vj, vk);
    }

    void uni_fcmp_ceq_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_ceq_s(xd, xj, xk);
    }

    void uni_fcmp_ceq_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_ceq_s(vd, vj, vk);
    }

    void uni_fcmp_cueq_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cueq_s(xd, xj, xk);
    }

    void uni_fcmp_cueq_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cueq_s(vd, vj, vk);
    }

    void uni_fcmp_clt_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_clt_s(xd, xj, xk);
    }

    void uni_fcmp_clt_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_clt_s(vd, vj, vk);
    }

    void uni_fcmp_cle_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cle_s(xd, xj, xk);
    }

    void uni_fcmp_cle_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cle_s(vd, vj, vk);
    }

    void uni_fcmp_cule_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cule_s(xd, xj, xk);
    }

    void uni_fcmp_cule_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cule_s(vd, vj, vk);
    }

    void uni_fcmp_cult_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cult_s(xd, xj, xk);
    }

    void uni_fcmp_cult_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cult_s(vd, vj, vk);
    }

    void uni_fcmp_cne_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cne_s(xd, xj, xk);
    }

    void uni_fcmp_cne_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cne_s(vd, vj, vk);
    }

    void uni_fcmp_cor_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cor_s(xd, xj, xk);
    }

    void uni_fcmp_cor_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cor_s(vd, vj, vk);
    }

    void uni_fcmp_cune_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvfcmp_cune_s(xd, xj, xk);
    }

    void uni_fcmp_cune_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vfcmp_cune_s(vd, vj, vk);
    }

    void uni_vpor(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
            const Xbyak_loongarch64::VReg &vk) {
        vor_v(vd, vj, vk);
    }

    void uni_vpor(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
            const Xbyak_loongarch64::XVReg &xk) {
        xvor_v(xd, xj, xk);
    }

    void uni_fmadd_s(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk, const Xbyak_loongarch64::XVReg &xa) {
        xvfmadd_s(xd, xj, xk, xa);
    }

    void uni_fmadd_s(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk, const Xbyak_loongarch64::VReg &va) {
        vfmadd_s(vd, vj, vk, va);
    }

    void uni_fmadd_d(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk, const Xbyak_loongarch64::XVReg &xa) {
        xvfmadd_d(xd, xj, xk, xa);
    }

    void uni_fmadd_d(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk, const Xbyak_loongarch64::VReg &va) {
        vfmadd_d(vd, vj, vk, va);
    }

    void uni_add_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvadd_w(xd, xj, xk);
    }

    void uni_add_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vadd_w(vd, vj, vk);
    }

    void uni_max_w(const Xbyak_loongarch64::XVReg &xd, const Xbyak_loongarch64::XVReg &xj,
             const Xbyak_loongarch64::XVReg &xk) {
        xvmax_w(xd, xj, xk);
    }

    void uni_max_w(const Xbyak_loongarch64::VReg &vd, const Xbyak_loongarch64::VReg &vj,
             const Xbyak_loongarch64::VReg &vk) {
        vmax_w(vd, vj, vk);
    }

    void push_xreg(const Xbyak_loongarch64::XReg &xreg) {
        addi_d(sp, sp, -1 * xreg_len);
        st_d(xreg, sp, 0);
    }

    void pop_xreg(const Xbyak_loongarch64::XReg &xreg) {
        ld_d(xreg, sp, 0);
        addi_d(sp, sp, xreg_len);
    }
    /*
      Saturation facility functions. enable to prepare the register
      holding the saturation upperbound and apply the saturation on
      the floating point register
     */
    void init_saturate_f32(Xbyak_loongarch64::XVReg vmm_lbound, Xbyak_loongarch64::XVReg vmm_ubound, const Xbyak_loongarch64::XReg &reg_tmp,
            data_type_t idt, data_type_t odt) {
        using namespace data_type;
        if (!((idt == f32) && utils::one_of(odt, u8, s8, s32))) return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data
        if (odt == u8) xvxor_v(vmm_lbound, vmm_lbound, vmm_lbound);

        float saturation_ubound = types::max_value<float>(odt);
        mov_imm(reg_tmp, float2int(saturation_ubound));
        xvreplgr2vr_w(vmm_ubound, reg_tmp);
    }

    void saturate_f32(const Xbyak_loongarch64::XVReg &vmm, const Xbyak_loongarch64::XVReg &vmm_lbound,
            const Xbyak_loongarch64::XVReg &vmm_ubound, data_type_t odt) {
        // This function is used to saturate to odt in f32 before converting
        // to s32 in order to avoid bad saturation due to cvtps2dq
        // behavior (it returns INT_MIN if the f32 is out of the
        // s32 range)
        using namespace data_type;
        if (!utils::one_of(odt, u8, s8, s32)) return;

        // no need to apply lower saturation bound when odt is
        // signed, as cvtps2dq will return MIN_INT if the value
        // does not fit
        if (odt == u8) xvfmax_s(vmm, vmm, vmm_lbound);
        xvfmin_s(vmm, vmm, vmm_ubound);
    }

    void init_saturate_f32(Xbyak_loongarch64::VReg vmm_lbound, Xbyak_loongarch64::VReg vmm_ubound, const Xbyak_loongarch64::XReg &reg_tmp,
            data_type_t idt, data_type_t odt) {
        using namespace data_type;
        if (!((idt == f32) && utils::one_of(odt, u8, s8, s32))) return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data
        if (odt == u8) vxor_v(vmm_lbound, vmm_lbound, vmm_lbound);

        float saturation_ubound = types::max_value<float>(odt);
        mov_imm(reg_tmp, float2int(saturation_ubound));
        vreplgr2vr_w(vmm_ubound, reg_tmp);
    }

    void saturate_f32(const Xbyak_loongarch64::VReg &vmm, const Xbyak_loongarch64::VReg &vmm_lbound,
            const Xbyak_loongarch64::VReg &vmm_ubound, data_type_t odt) {
        // This function is used to saturate to odt in f32 before converting
        // to s32 in order to avoid bad saturation due to cvtps2dq
        // behavior (it returns INT_MIN if the f32 is out of the
        // s32 range)
        using namespace data_type;
        if (!utils::one_of(odt, u8, s8, s32)) return;

        // no need to apply lower saturation bound when odt is
        // signed, as cvtps2dq will return MIN_INT if the value
        // does not fit
        if (odt == u8) vfmax_s(vmm, vmm, vmm_lbound);
        vfmin_s(vmm, vmm, vmm_ubound);
    }


    /**
    * load_bytes is the utility function to facilitate loading of
    * load_size (0 <= load_size <= 32) many contiguous bytes into the Xmm/Ymm
    * register from the memory referenced by ptr[reg + offset] address.
    *
    * Functionally, invocation of load_bytes is equivalent to
    * the following loop:
    *
    * for (int idx = 0; idx < load_size; ++idx)
    *     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
    *
    * TODO: Add an option to zero-out unloaded bytes in the Xmm register.
    * TODO: Add an option for unsafe_load wherein one could read outside the
    * provided memory buffer so as to minimize the total number of read
    * memory instructions.
    */
    template <typename Vmm>
    void load_bytes(const Vmm &vmm, const Xbyak_loongarch64::XReg &reg, int64_t offset,
            int load_size) {

        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // Ensure data fits completely inside the Xmm/Ymm register
        assert(load_size >= 0 && load_size <= 32);

        assert(is_valid_isa(lasx)
                && "routine is not supported for the current isa");

        auto xvreg = Xbyak_loongarch64::XVReg(vmm.getIdx());
        auto vreg = Xbyak_loongarch64::VReg(vmm.getIdx());
        auto regvalue = X_TMP_0;
        auto regaddr = reg;

        switch (load_size) {
            case 0: break;
            case 1:
                uni_ld_b(regvalue, regaddr, offset);
                vinsgr2vr_b(vreg, regvalue, 0);
                break;
            case 2:
                uni_ld_h(regvalue, regaddr, offset);
                vinsgr2vr_h(vreg, regvalue, 0);
                break;
            case 3:
                uni_ld_h(regvalue, regaddr, offset);
                vinsgr2vr_h(vreg, regvalue, 0);
                uni_ld_b(regvalue, regaddr, offset + 2);
                vinsgr2vr_b(vreg, regvalue, 2);
                break;
            case 4:
                uni_ld_w(regvalue, regaddr, offset);
                vinsgr2vr_w(vreg, regvalue, 0);
                break;
            case 5:
                uni_ld_w(regvalue, regaddr, offset);
                vinsgr2vr_w(vreg, regvalue, 0);
                uni_ld_b(regvalue, regaddr, offset + 4);
                vinsgr2vr_b(vreg, regvalue, 4);
                break;
            case 6:
                uni_ld_w(regvalue, regaddr, offset);
                vinsgr2vr_w(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 4);
                vinsgr2vr_h(vreg, regvalue, 2);
                break;
            case 7:
                uni_ld_w(regvalue, regaddr, offset);
                vinsgr2vr_w(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 4);
                vinsgr2vr_h(vreg, regvalue, 2);
                uni_ld_b(regvalue, regaddr, offset + 6);
                vinsgr2vr_b(vreg, regvalue, 6);
                break;
            case 8:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                break;
            case 9:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_b(regvalue, regaddr, offset + 8);
                vinsgr2vr_b(vreg, regvalue, 8);
                break;
            case 10:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 8);
                vinsgr2vr_h(vreg, regvalue, 4);
                break;
            case 11:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 8);
                vinsgr2vr_h(vreg, regvalue, 4);
                uni_ld_b(regvalue, regaddr, offset + 10);
                vinsgr2vr_b(vreg, regvalue, 10);
                break;
            case 12:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_w(regvalue, regaddr, offset + 8);
                vinsgr2vr_w(vreg, regvalue, 2);
                break;
            case 13:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_w(regvalue, regaddr, offset + 8);
                vinsgr2vr_w(vreg, regvalue, 2);
                uni_ld_b(regvalue, regaddr, offset + 12);
                vinsgr2vr_b(vreg, regvalue, 12);
                break;
            case 14:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_w(regvalue, regaddr, offset + 8);
                vinsgr2vr_w(vreg, regvalue, 2);
                uni_ld_h(regvalue, regaddr, offset + 12);
                vinsgr2vr_h(vreg, regvalue, 6);
                break;
            case 15:
                uni_ld_d(regvalue, regaddr, offset);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_w(regvalue, regaddr, offset + 8);
                vinsgr2vr_w(vreg, regvalue, 2);
                uni_ld_h(regvalue, regaddr, offset + 12);
                vinsgr2vr_h(vreg, regvalue, 6);
                uni_ld_b(regvalue, regaddr, offset + 14);
                vinsgr2vr_b(vreg, regvalue, 14);
                break;
            case 16:
                uni_xvld(vreg, regaddr, offset);
                break;
            case 17:
                uni_xvld(vreg, regaddr, offset);
                uni_ld_b(regvalue, regaddr, offset + 16);
                xvinsgr2vr_w(xvreg, regvalue, 4);
                break;
            case 18:
                uni_xvld(vreg, regaddr, offset);
                uni_ld_h(regvalue, regaddr, offset + 16);
                xvinsgr2vr_w(xvreg, regvalue, 4);
                break;
            case 19:
                uni_ld_h(regvalue, regaddr, offset + 16);
                vinsgr2vr_h(vreg, regvalue, 0);
                uni_ld_b(regvalue, regaddr, offset + 18);
                vinsgr2vr_b(vreg, regvalue, 2);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 20:
                uni_xvld(vreg, regaddr, offset);
                uni_ld_w(regvalue, regaddr, offset + 16);
                xvinsgr2vr_w(xvreg, regvalue, 4);
                break;
            case 21:
                uni_ld_w(regvalue, regaddr, offset + 16);
                vinsgr2vr_w(vreg, regvalue, 0);
                uni_ld_b(regvalue, regaddr, offset + 20);
                vinsgr2vr_b(vreg, regvalue, 4);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 22:
                uni_ld_w(regvalue, regaddr, offset + 16);
                vinsgr2vr_w(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 20);
                vinsgr2vr_h(vreg, regvalue, 2);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 23:
                uni_ld_w(regvalue, regaddr, offset + 16);
                vinsgr2vr_w(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 20);
                vinsgr2vr_h(vreg, regvalue, 2);
                uni_ld_b(regvalue, regaddr, offset + 22);
                vinsgr2vr_b(vreg, regvalue, 6);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 24:
                uni_xvld(vreg, regaddr, offset);
                uni_ld_d(regvalue, regaddr, offset + 16);
                xvinsgr2vr_d(xvreg, regvalue, 2);
                break;
            case 25:
                uni_ld_d(regvalue, regaddr, offset + 16);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_b(regvalue, regaddr, offset + 24);
                vinsgr2vr_b(vreg, regvalue, 8);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 26:
                uni_ld_d(regvalue, regaddr, offset + 16);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 24);
                vinsgr2vr_h(vreg, regvalue, 4);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 27:
                uni_ld_d(regvalue, regaddr, offset + 16);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_h(regvalue, regaddr, offset + 24);
                vinsgr2vr_h(vreg, regvalue, 4);
                uni_ld_b(regvalue, regaddr, offset + 26);
                vinsgr2vr_b(vreg, regvalue, 10);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 28:
                uni_xvld(vreg, regaddr, offset);
                uni_ld_d(regvalue, regaddr, offset + 16);
                xvinsgr2vr_d(xvreg, regvalue, 2);
                uni_ld_w(regvalue, regaddr, offset + 24);
                xvinsgr2vr_w(xvreg, regvalue, 6);
                break;
            case 29:
                uni_ld_d(regvalue, regaddr, offset + 16);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_w(regvalue, regaddr, offset + 24);
                vinsgr2vr_w(vreg, regvalue, 2);
                uni_ld_b(regvalue, regaddr, offset + 28);
                vinsgr2vr_b(vreg, regvalue, 12);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 30:
                uni_ld_d(regvalue, regaddr, offset + 16);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_w(regvalue, regaddr, offset + 24);
                vinsgr2vr_w(vreg, regvalue, 2);
                uni_ld_h(regvalue, regaddr, offset + 28);
                vinsgr2vr_h(vreg, regvalue, 6);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 31:
                uni_ld_d(regvalue, regaddr, offset + 16);
                vinsgr2vr_d(vreg, regvalue, 0);
                uni_ld_w(regvalue, regaddr, offset + 24);
                vinsgr2vr_w(vreg, regvalue, 2);
                uni_ld_h(regvalue, regaddr, offset + 28);
                vinsgr2vr_h(vreg, regvalue, 6);
                uni_ld_b(regvalue, regaddr, offset + 30);
                vinsgr2vr_b(vreg, regvalue, 14);
                xvpermi_q(xvreg, xvreg, 0);
                uni_xvld(vreg, regaddr, offset);
                break;
            case 32:
                uni_xvld(xvreg, regaddr, offset);
                break;
            default:
                break;
        }
    }

    /**
    * store_bytes is the utility function to facilitate storing of
    * store_size (0 <= store_size <= 32) many contiguous bytes from the Xmm/Ymm
    * register into the memory referenced by ptr[reg + offset] address.
    *
    * Additionally, when store_size > 16, the input Ymm register will not be
    * preserved due to the usage of vextracti128 instruction.
    *
    * Functionally, invocation of store_bytes is equivalent
    * to the following loop:
    *
    * for (int idx = 0; idx < store_size; ++idx)
    *     vpextrb(ptr[reg + offset + idx], xmm, idx);
    *
    * TODO: Add an option for unsafe_store wherein one could store extra dwords
    * past the provided memory buffer so as to minimize the total number of
    * write memory instructions.
    */
public:
    template <typename Vmm>
    void store_bytes(const Vmm &vmm, const Xbyak_loongarch64::XReg &reg, int64_t offset,
            int store_size) {

        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // Ensure data fits completely inside the Xmm/Ymm register
        assert(store_size >= 0 && store_size <= 32);

        assert(is_valid_isa(lasx)
                && "routine is not supported for the current isa");

        auto xvreg = Xbyak_loongarch64::XVReg(vmm.getIdx());
        auto vreg = Xbyak_loongarch64::VReg(vmm.getIdx());
        auto regaddr = reg;
        switch (store_size) {
            case 0: break;
            case 1:
                uni_xvstelm_b(vreg, regaddr, offset, 0);
                break;
            case 2:
                uni_xvstelm_h(vreg, regaddr, offset, 0);
                break;
            case 3:
                uni_xvstelm_h(vreg, regaddr, offset, 0);
                uni_xvstelm_b(vreg, regaddr, offset + 2, 2);
                break;
            case 4:
                uni_xvstelm_w(vreg, regaddr, offset, 0);
                break;
            case 5:
                uni_xvstelm_w(vreg, regaddr, offset, 0);
                uni_xvstelm_b(vreg, regaddr, offset + 4, 4);
                break;
            case 6:
                uni_xvstelm_w(vreg, regaddr, offset, 0);
                uni_xvstelm_h(vreg, regaddr, offset + 4, 2);
                break;
            case 7:
                uni_xvstelm_w(vreg, regaddr, offset, 0);
                uni_xvstelm_h(vreg, regaddr, offset, 2);
                uni_xvstelm_b(vreg, regaddr, offset, 6);
                break;
            case 8:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                break;
            case 9:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                uni_xvstelm_b(vreg, regaddr, offset + 8, 8);
                break;
            case 10:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                uni_xvstelm_h(vreg, regaddr, offset + 8, 4);
                break;
            case 11:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                uni_xvstelm_h(vreg, regaddr, offset + 8, 4);
                uni_xvstelm_b(vreg, regaddr, offset + 10, 10);
                break;
            case 12:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                uni_xvstelm_w(vreg, regaddr, offset + 8, 2);
                break;
            case 13:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                uni_xvstelm_h(vreg, regaddr, offset + 8, 2);
                uni_xvstelm_b(vreg, regaddr, offset + 12, 12);
                break;
            case 14:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                uni_xvstelm_w(vreg, regaddr, offset + 8, 2);
                uni_xvstelm_h(vreg, regaddr, offset + 12, 6);
                break;
            case 15:
                uni_xvstelm_d(vreg, regaddr, offset, 0);
                uni_xvstelm_w(vreg, regaddr, offset + 8, 2);
                uni_xvstelm_h(vreg, regaddr, offset + 12, 6);
                uni_xvstelm_b(vreg, regaddr, offset + 14, 14);
                break;
            case 16:
                uni_xvst(vreg, regaddr, offset);
                break;
            case 17:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_b(xvreg, regaddr, offset + 16, 16);
                break;
            case 18:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_h(xvreg, regaddr, offset + 16, 8);
                break;
            case 19:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_h(xvreg, regaddr, offset + 16, 8);
                uni_xvstelm_b(xvreg, regaddr, offset + 18, 18);
                break;
            case 20:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_w(xvreg, regaddr, offset + 16, 4);
                break;
            case 21:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_w(xvreg, regaddr, offset + 16, 4);
                uni_xvstelm_b(xvreg, regaddr, offset + 20, 20);
                break;
            case 22:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_w(xvreg, regaddr, offset + 16, 4);
                uni_xvstelm_h(xvreg, regaddr, offset + 20, 10);
                break;
            case 23:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_w(xvreg, regaddr, offset + 16, 4);
                uni_xvstelm_h(xvreg, regaddr, offset + 20, 10);
                uni_xvstelm_b(xvreg, regaddr, offset + 22, 22);
                break;
            case 24:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                break;
            case 25:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                uni_xvstelm_b(xvreg, regaddr, offset + 24, 24);
                break;
            case 26:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                uni_xvstelm_h(xvreg, regaddr, offset + 24, 12);
                break;
            case 27:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                uni_xvstelm_h(xvreg, regaddr, offset + 24, 12);
                uni_xvstelm_b(xvreg, regaddr, offset + 26, 26);
                break;
            case 28:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                uni_xvstelm_w(xvreg, regaddr, offset + 24, 6);
                break;
            case 29:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                uni_xvstelm_w(xvreg, regaddr, offset + 24, 6);
                uni_xvstelm_b(xvreg, regaddr, offset + 28, 28);
                break;
            case 30:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                uni_xvstelm_w(xvreg, regaddr, offset + 24, 6);
                uni_xvstelm_h(xvreg, regaddr, offset + 28, 14);
                break;
            case 31:
                uni_xvst(vreg, regaddr, offset);
                uni_xvstelm_d(xvreg, regaddr, offset + 16, 2);
                uni_xvstelm_w(xvreg, regaddr, offset + 24, 6);
                uni_xvstelm_h(xvreg, regaddr, offset + 28, 14);
                uni_xvstelm_b(xvreg, regaddr, offset + 30, 30);
                break;
            case 32:
                uni_xvst(xvreg, regaddr, offset);
                break;
            default:
                break;
        }
    }

    void store_mask_words(const Xbyak_loongarch64::XVReg &xvreg, const Xbyak_loongarch64::XReg &regaddr,
            int64_t offset, const Xbyak_loongarch64::XVReg& xvmask) {
        Xbyak_loongarch64::Label cond_store[8];
        for (int mi = 0; mi < 8; ++mi) {
            xvpickve2gr_w(X_TMP_0, xvmask, mi);
            beqz(X_TMP_0, cond_store[mi]);
            uni_xvstelm_w(xvreg, regaddr, offset + mi * sizeof(float), mi);
            L(cond_store[mi]);
        }
    }

    void vgatherqps(const Xbyak_loongarch64::VReg &vrdst, const Xbyak_loongarch64::XReg &raddr,
            const Xbyak_loongarch64::XVReg &xvind, const int32_t offset, const Xbyak_loongarch64::VReg &vrmask) {
        Xbyak_loongarch64::Label cond_load[4];
        for (int j = 0; j < 4; ++j) {
            vpickve2gr_w(X_TMP_0, vrmask, j);
            beqz(X_TMP_0, cond_load[j]);
            xvpickve2gr_d(X_TMP_1, xvind, j);
            add_d(X_TMP_1, raddr, X_TMP_1);
            uni_ld_w(X_TMP_0, X_TMP_1, offset);
            vinsgr2vr_w(vrdst, X_TMP_0, j);
            L(cond_load[j]);
        }
    }

public:
    /**
    * load_bytes_to_dword_extension is the utility function to facilitate
    * loading of load_size (0 <= load_size <= 16) many contiguous bytes in
    * the Xmm register from the memory referenced by ptr[reg + offset]
    * address and then do signed/zero extension of those to double words.
    *
    * Functionally, invocation of load_bytes_to_dword_extension is equivalent
    * to the following:
    *
    * for (int idx = 0; idx < load_size; ++idx)
    *     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
    * if (is_signed) vpmovsxbd(vmm, vmm); else vpmovzxbd(vmm, vmm);
    *
    * Valid values for the load_size variable are:
    * [0..4] for XMM version of the function
    * [0..8] for YMM version of the function.
    * TODO: Implement this routine for every ISA.
    */
    template <typename Vmm>
    void load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak_loongarch64::XReg &reg,
            int64_t offset, bool is_signed, int load_size) {
        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // Ensure extended double words fit inside Ymm (32 * load_size <= 256)
        assert(load_size >= 0 && load_size <= 8);

        assert(is_valid_isa(lasx)
                && "routine is not supported for the current isa");
	    for (int32_t i = 0; i < load_size; ++i) {
            if (is_signed)
                uni_ld_b(X_TMP_1, reg, offset + i);
            else
                uni_ld_bu(X_TMP_1, reg, offset + i);
            uni_xvinsgr2vr_w(vmm, X_TMP_1, i);
        }
    }

    /* A utility function to load data of type type_in to vmm register
     * from the memory. Moreover load_size many chunks are read from the memory
     * beginning with ptr[reg + offset] address.
     *
     * TODO: Support for every possible data type.
     */
    template <typename Vmm>
    void load_data(data_type_t type_in, const Vmm &vmm, const Xbyak_loongarch64::XReg &reg,
            int64_t offset, int load_size) {
        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        assert(is_valid_isa(lsx)
                && "routine is not supported for the current isa");

        switch (type_in) {
            case data_type::f32:
            case data_type::s32:
                load_bytes(vmm, reg, offset, sizeof(int32_t) * load_size);
                break;
            case data_type::s8:
            case data_type::u8:
                load_bytes_to_dword_extension(
                        vmm, reg, offset, type_in == data_type::s8, load_size);
                break;
            default: assert(!"unsupported source data type");
        }
    }

    /* A utility function to process f32 tail (load, store or other) depending
     * on tail size, stored in Reg64. Tail size must be value from 0 to 3/7
     * (Xmm/Ymm). Tail process functions require integer as argument to specify
     * behavior for each tail size.
     *
     * Only supported for Xmm and Ymm.
     */
    template <typename Vmm>
    void runtime_tail_process(const Xbyak_loongarch64::XReg &reg_tail,
            const Xbyak_loongarch64::XReg &reg_tmp,
            const std::function<void(int)> &tail_process) {
        constexpr int simd_w_ymm = 8;
        constexpr int f32_bits = sizeof(float) * 8;
        const auto simd_w = Vmm(0).getBit() / f32_bits;

        Xbyak_loongarch64::Label label_tbl, label_tbl_end;
        Xbyak_loongarch64::Label l_case[simd_w_ymm];

        pcaddi(reg_tmp, label_tbl);
        mov_imm(X_TMP_0, sizeof(void*));
        mul_d(X_TMP_0, reg_tail, X_TMP_0);
        add_d(reg_tmp, reg_tmp, X_TMP_0);
        jirl(zero, reg_tmp, 0);

        // create jump table
        L(label_tbl);
        for (size_t i = 0; i < simd_w; i++)
            putL(l_case[i]);

        // cases for each tail size - from 0 to 3/7
        L(l_case[0]);
        b(label_tbl_end);
        for (size_t i = 1; i < simd_w; i++) {
            L(l_case[i]);
            tail_process(i);
            b(label_tbl_end);
        }
        L(label_tbl_end);
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    /* All uni_ instructions -- apart from uni_vzeroupper() -- will comply with
     * the max_cpu_isa argument */
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true, cpu_isa_t max_cpu_isa = isa_all)
        : Xbyak_loongarch64::CodeGenerator(code_size,
                (code_ptr == nullptr && use_autogrow) ? Xbyak_loongarch64::AutoGrow
                                                      : code_ptr)
        , max_cpu_isa_(max_cpu_isa) {}

    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const uint8_t *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const uint8_t *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? status::success : status::runtime_error;
    }

private:
    const cpu_isa_t max_cpu_isa_;
    const uint8_t *getCode() {
        this->ready();
        if (!is_initialized()) return nullptr;
        const uint8_t *code = CodeGenerator::getCode();
        register_jit_code(code, getSize());
        return code;
    }

    inline bool is_valid_isa(cpu_isa_t isa) {
        return mayiuse(isa);
    }

    static inline bool is_initialized() {
        //return Xbyak_loongarch64::GetError() == Xbyak_loongarch64::ERR_NONE;
	/* At the moment, Xbyak_loongarch64 does not have GetError()\
         so that return dummy result. */
	    return true;
    }

protected:
    virtual void generate() = 0;
    const uint8_t *jit_ker_ = nullptr;
};

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
