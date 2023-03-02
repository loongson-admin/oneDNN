/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2021-2026 Loongson
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
#include <numeric>
#include "common/broadcast_strategy.hpp"
#include "cpu/loongarch64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {
namespace injector_utils {

static std::size_t get_vmm_size_bytes(const Xbyak_loongarch64::VReg &vmm) {
    static constexpr int byte_size_bits = 8;
    return vmm.getBit() / byte_size_bits;
}

static std::size_t calc_vmm_to_preserve_size_bytes(
        const std::initializer_list<Xbyak_loongarch64::VReg> &vmm_to_preserve) {

    return std::accumulate(vmm_to_preserve.begin(), vmm_to_preserve.end(),
            std::size_t(0u), [](std::size_t accum, const Xbyak_loongarch64::VReg &vmm) {
                return accum + get_vmm_size_bytes(vmm);
            });
}

register_preserve_guard_t::register_preserve_guard_t(jit_generator *host,
        std::initializer_list<Xbyak_loongarch64::XReg> reg64_to_preserve,
        std::initializer_list<Xbyak_loongarch64::VReg> vmm_to_preserve)
    : host_(host)
    , reg64_stack_(reg64_to_preserve)
    , vmm_stack_(vmm_to_preserve)
    , vmm_to_preserve_size_bytes_(
              calc_vmm_to_preserve_size_bytes(vmm_to_preserve)) {

    for (const auto &reg : reg64_to_preserve) {
        host_->addi_d(host_->sp, host_->sp, -8);
        host_->st_d(reg, host_->sp, 0);
    }

    if (!vmm_stack_.empty()) {
        host_->add_imm(host_->sp, host_->sp, -1 * vmm_to_preserve_size_bytes_, host_->X_TMP_0);

        auto stack_offset = vmm_to_preserve_size_bytes_;
        for (const auto &vmm : vmm_to_preserve) {
            stack_offset -= get_vmm_size_bytes(vmm);
            const auto idx = vmm.getIdx();
            if (vmm.isVReg())
                host_->uni_xvst(Xbyak_loongarch64::VReg(idx), host_->sp, stack_offset);
            else if (vmm.isXVReg())
                host_->uni_xvst(Xbyak_loongarch64::XVReg(idx), host_->sp, stack_offset);
            else
                assert("unreachable");
        }
    }
}

register_preserve_guard_t::~register_preserve_guard_t() {

    auto tmp_stack_offset = 0;

    while (!vmm_stack_.empty()) {
        const Xbyak_loongarch64::VReg &vmm = vmm_stack_.top();
        const auto idx = vmm.getIdx();
        if (vmm.isVReg())
            host_->uni_xvld(Xbyak_loongarch64::VReg(idx), host_->sp, tmp_stack_offset);
        else if (vmm.isXVReg())
            host_->uni_xvld(Xbyak_loongarch64::XVReg(idx), host_->sp, tmp_stack_offset);
        else
            assert("unreachable");

        tmp_stack_offset += get_vmm_size_bytes(vmm);
        vmm_stack_.pop();
    }

    if (vmm_to_preserve_size_bytes_)
        host_->add_imm(host_->sp, host_->sp, vmm_to_preserve_size_bytes_, host_->X_TMP_0);

    while (!reg64_stack_.empty()) {
        host_->ld_d(reg64_stack_.top(), host_->sp, 0);
        host_->addi_d(host_->sp, host_->sp, 8);
        reg64_stack_.pop();
    }
}

size_t register_preserve_guard_t::stack_space_occupied() const {
    constexpr static size_t reg64_size = 8;
    const size_t stack_space_occupied
            = vmm_to_preserve_size_bytes_ + reg64_stack_.size() * reg64_size;

    return stack_space_occupied;
};

conditional_register_preserve_guard_t::conditional_register_preserve_guard_t(
        bool condition_to_be_met, jit_generator *host,
        std::initializer_list<Xbyak_loongarch64::XReg> reg64_to_preserve,
        std::initializer_list<Xbyak_loongarch64::VReg> vmm_to_preserve)
    : register_preserve_guard_t {condition_to_be_met
                    ? register_preserve_guard_t {host, reg64_to_preserve,
                            vmm_to_preserve}
                    : register_preserve_guard_t {nullptr, {}, {}}} {};

} // namespace injector_utils
} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
