/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_LOONGARCH64_CPU_ISA_TRAITS_HPP
#define CPU_LOONGARCH64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* turn off `size_t to other-type implicit casting` warning
 * currently we have a lot of jit-generated instructions that
 * take uint32_t, but we pass size_t (e.g. due to using sizeof).
 * FIXME: replace size_t parameters with the appropriate ones */
#pragma warning(disable : 4267)
#endif
#include "cpu/loongarch64/xbyak_loongarch64/xbyak_loongarch64/xbyak_loongarch64.h"
#include "cpu/loongarch64/xbyak_loongarch64/xbyak_loongarch64/xbyak_loongarch64_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

enum {
    // LOONGARCH64 Advanced SIMD & floating-point
    dnnl_cpu_isa_lsx = 0x1,
    dnnl_cpu_isa_lasx = 0x2,
};

enum cpu_isa_bit_t : unsigned {
    lsx_bit = 1u << 0,
    lasx_bit = 1u << 1,
};

enum cpu_isa_t : unsigned {
    isa_any = 0u,
    lsx = lsx_bit,
    lasx = lasx_bit,
    isa_all = ~0u,
};

const char *get_isa_info();

cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

template <cpu_isa_t>
struct cpu_isa_traits {};

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_all;
    static constexpr const char *user_option_env = "ALL";
};

template <>
struct cpu_isa_traits<lsx> {
    typedef Xbyak_loongarch64::VReg Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_lsx);
    static constexpr const char *user_option_env = "LSX";
};

template <>
struct cpu_isa_traits<lasx> {
    typedef Xbyak_loongarch64::XVReg Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_lasx);
    static constexpr const char *user_option_env = "LASX";
};

namespace {

static Xbyak_loongarch64::util::Cpu cpu;
static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak_loongarch64::util;

    unsigned cpu_isa_mask = loongarch64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case lsx: return cpu.has(Cpu::tLSX);
        case lasx: return cpu.has(Cpu::tLASX);
        case isa_any: return true;
        case isa_all: return false;
    }
    return false;
}

static inline bool mayiuse_atomic() {
    using namespace Xbyak_loongarch64::util;
    return cpu.isAtomicSupported();
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return false;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_any ? prefix STRINGIFY(any) : \
    ((isa) == lsx ? prefix STRINGIFY(lsx) : \
    ((isa) == lasx ? prefix STRINGIFY(lasx) : \
    prefix suffix_if_any)))
/* clang-format on */

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
