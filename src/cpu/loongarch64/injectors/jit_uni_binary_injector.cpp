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
#include <algorithm>
#include <cmath>

#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "cpu/loongarch64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {
namespace binary_injector {

static bcast_set_t get_all_strategies_supported_by_injector() {
    return bcast_set_t {broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::no_broadcast};
}

bool is_data_supported(cpu_isa_t isa, data_type_t data_type) {
    return IMPLICATION(
            data_type == data_type::bf16, false);
}

static bool src1_desc_layout_same_as_dst_d(
        const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d) {
    if (dst_d.md_ == nullptr) return false;
    const auto &lhs = src1_desc;
    const auto &rhs = *(dst_d.md_);

    using namespace dnnl::impl::utils;
    return lhs.ndims == rhs.ndims
            && (lhs.format_kind == rhs.format_kind
                    || one_of(
                            format_kind::any, lhs.format_kind, rhs.format_kind))
            && array_cmp(lhs.dims, rhs.dims, lhs.ndims)
            && array_cmp(lhs.padded_dims, rhs.padded_dims, lhs.ndims)
            && array_cmp(lhs.padded_offsets, rhs.padded_offsets, lhs.ndims)
            && lhs.offset0 == rhs.offset0;
}

bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
            src1_desc, dst_d, supported_strategy_set);

    if (bcast_type == broadcasting_strategy_t::no_broadcast) {
        // in case of no broadcast data layout of dst and src1 have to be the same
        if (!src1_desc_layout_same_as_dst_d(src1_desc, dst_d)) return false;
    }

    return bcast_type != broadcasting_strategy_t::unsupported;
}

bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return is_data_supported(isa, src1_desc.data_type)
            && is_bcast_supported(src1_desc, dst_d, supported_strategy_set);
}

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d,
                            supported_strategy_set);
                    return bcast_type == broadcasting_strategy_t::unsupported;
                }
                return false;
            });
}

bool binary_args_tail_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d, int vlen,
        const bcast_set_t &supported_strategy_set) {
    const auto channels = dst_d.dims()[1];
    const int vmm_l_len = vlen / 4;

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d,
                            supported_strategy_set);
                    return utils::one_of(bcast_type,
                                   broadcasting_strategy_t::per_oc,
                                   broadcasting_strategy_t::per_oc_spatial)
                            && (channels % vmm_l_len != 0);
                }
                return false;
            });
}

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) {
                if (entry.is_binary()) {
                    const memory_desc_wrapper rhs_arg_d(entry.binary.src1_desc);
                    return rhs_arg_d.matches_tag(tag);
                }
                return true;
            });
}

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return any_binary_postop_rhs_per_oc_broadcast(
            post_ops, dst_d, get_all_strategies_supported_by_injector());
}

bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d,
                            supported_strategy_set);
                    return bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                            == broadcasting_strategy_t::per_oc_spatial;
                }
                return false;
            });
}

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> &predicate) {
    return all_binary_postop_rhs_per_oc_broadcast(post_ops, dst_d,
            get_all_strategies_supported_by_injector(), predicate);
}

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set,
        const std::function<bool(const memory_desc_wrapper &)> &predicate) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d,
                            supported_strategy_set);
                    if (bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                                    == broadcasting_strategy_t::per_oc_spatial)
                        return predicate(
                                memory_desc_wrapper(entry.binary.src1_desc));
                }
                return true;
            });
}

static_params_t::static_params_t(const Xbyak_loongarch64::XReg &param1,
        const bcast_set_t &supported_strategy_set,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : param1(param1)
    , supported_strategy_set(supported_strategy_set)
    , rhs_arg_static_params(rhs_arg_static_params) {}

static_params_t::static_params_t(const Xbyak_loongarch64::XReg &param1,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : static_params_t(param1, get_all_strategies_supported_by_injector(),
            rhs_arg_static_params) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak_loongarch64::XReg &rhs_addr_reg,
        const Xbyak_loongarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_vmm_helper,
            abi_param_offset, dst_d, tail_size, Xbyak_loongarch64::XReg(31),
            use_exact_tail_scalar_bcast, rhs_helper_reg,
            false /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak_loongarch64::XReg &rhs_addr_reg,
        const Xbyak_loongarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_loongarch64::XReg &tail_opmask, bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_vmm_helper,
            abi_param_offset, dst_d, tail_size, tail_opmask,
            use_exact_tail_scalar_bcast, rhs_helper_reg,
            true /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak_loongarch64::XReg &rhs_addr_reg,
        const Xbyak_loongarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_loongarch64::XReg &tail_opmask, const Xbyak_loongarch64::XReg &reg_tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_vmm_helper,
            abi_param_offset, dst_d, tail_size, tail_opmask,
            use_exact_tail_scalar_bcast, reg_tail_size,
            true /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak_loongarch64::XReg &rhs_addr_reg,
        const Xbyak_loongarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_loongarch64::XReg &tail_opmask, bool use_exact_tail_scalar_bcast,
        const Xbyak_loongarch64::XReg &reg_tail_size, bool is_opmask_set)
    : rhs_dt_helper_vmm_idx(rhs_dt_helper_vmm_idx)
    , rhs_addr_reg(rhs_addr_reg)
    , rhs_helper_reg(rhs_helper_reg)
    , preserve_gpr_helpers(preserve_gpr_helpers)
    , preserve_vmm_helper(preserve_vmm_helper)
    , abi_param_offset(abi_param_offset)
    , dst_d(dst_d)
    , tail_size(tail_size)
    , tail_opmask(tail_opmask)
    , use_exact_tail_scalar_bcast(use_exact_tail_scalar_bcast)
    , reg_tail_size(reg_tail_size)
    , is_tail(tail_size)
    , is_opmask_set_(is_opmask_set) {}

template <cpu_isa_t isa, typename Vmm>
jit_uni_binary_injector_t<isa, Vmm>::jit_uni_binary_injector_t(
        jit_generator *host, const static_params_t &static_params)
    : host_(host)
    , rhs_arg_static_params_(static_params.rhs_arg_static_params)
    , param1_(static_params.param1)
    , supported_strategy_set_(static_params.supported_strategy_set) {}

template <typename ParamsMap>
static bool params_differ(ParamsMap &params,
        const typename ParamsMap::key_type key1,
        const typename ParamsMap::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    return it1->second != it2->second;
}

static bool rhs_arg_params_differ(size_t vmm_idx1, size_t vmm_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_elem_off_addr = rhs_arg_params.vmm_idx_to_out_elem_off_addr;
    const auto &out_elem_off_val = rhs_arg_params.vmm_idx_to_out_elem_off_val;
    const auto &out_off_oprnd = rhs_arg_params.vmm_idx_to_out_off_oprnd;
    const auto &oc_off_addr = rhs_arg_params.vmm_idx_to_oc_elem_off_addr;
    const auto &oc_off_val = rhs_arg_params.vmm_idx_to_oc_elem_off_val;
    const auto &oc_off_oprnd = rhs_arg_params.vmm_idx_to_oc_off_oprnd;
    const auto &sp_off_addr = rhs_arg_params.vmm_idx_to_sp_elem_off_addr;
    const auto &sp_off_val = rhs_arg_params.vmm_idx_to_sp_elem_off_val;
    const auto &sp_off_oprnd = rhs_arg_params.vmm_idx_to_sp_off_oprnd;

    if (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar) {
        return false;
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::no_broadcast) {
        return params_differ(out_elem_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(out_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc
            || rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial) {
        return params_differ(oc_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::per_mb_spatial) {
        return params_differ(sp_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_val, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_oprnd, vmm_idx1, vmm_idx2);
    }
    return true;
}

template <cpu_isa_t isa, typename Vmm>
int jit_uni_binary_injector_t<isa, Vmm>::adjust_temp_vmm_hint(
        int user_hint, int start_idx, int end_idx, int max_vmm_idx) const {
    const bool user_hint_in_vector_range
            = user_hint >= start_idx && user_hint <= end_idx;
    const bool user_hint_exceeded_limit = user_hint > max_vmm_idx;
    const bool user_hint_invalid
            = user_hint_in_vector_range || user_hint_exceeded_limit;

    if (user_hint_invalid) {
        const bool max_vmm_idx_in_vector_range
                = max_vmm_idx >= start_idx && max_vmm_idx <= end_idx;

        if (max_vmm_idx_in_vector_range || user_hint_exceeded_limit
                || user_hint == max_vmm_idx)
            return 0;
        else
            return max_vmm_idx;
    }

    return user_hint;
}

template <typename Vmm>
static void push_vmm(jit_generator *host, const Vmm &vmm) {
    host->addi_d(host->sp, host->sp, (int32_t)(-1 * injector_utils::vmm_size_t<Vmm>::bytes));
    host->uni_xvst(vmm, host->sp, 0);
}

template <typename Vmm>
static void pop_vmm(jit_generator *host, const Vmm &vmm) {
    host->uni_xvld(vmm, host->sp, 0);
    host->addi_d(host->sp, host->sp, injector_utils::vmm_size_t<Vmm>::bytes);
}

// unused function
//static void push_opmask(jit_generator *host, const Xbyak_loongarch64::XReg &k) {
//    static constexpr int k_mask_size = 8;
//    host->addi_d(host->sp, host->sp, -1 * k_mask_size);
//    host->st_d(k, host->sp, 0);
//}

// unused function
//static void pop_opmask(jit_generator *host, const Xbyak_loongarch64::XReg &k) {
//    static constexpr int k_mask_size = 8;
//    host->ld_d(k, host->sp, 0);
//    host->addi_d(host->sp, host->sp, k_mask_size);
//}

template <typename Vmm>
static void restore_stack(jit_generator *host, const Vmm &vmm) {
    host->addi_d(host->sp, host->sp, injector_utils::vmm_size_t<Vmm>::bytes);
}

template <cpu_isa_t isa, typename Vmm>
std::pair<bool, int> jit_uni_binary_injector_t<isa, Vmm>::should_preserve_vmm(
        int curr_idx, int vmm_hint, int max_vmm_idx,
        bool dt_helper_vmm_needed) const {
    if (dt_helper_vmm_needed && vmm_hint == curr_idx) {
        if (curr_idx == 0)
            return std::make_pair(true, max_vmm_idx);
        else
            return std::make_pair(true, 0);
    }
    return std::make_pair(false, vmm_hint);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::compute_vector_range(size_t start_idx,
        size_t end_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {

    if (vmm_idxs.empty()) return;
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin());

    // Phase 1 Validate temporary vmm user hint
    static constexpr int max_vmm_idx = cpu_isa_traits<isa>::n_vregs - 1;
    auto &vmm_hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    vmm_hint = adjust_temp_vmm_hint(vmm_hint, start_idx, end_idx, max_vmm_idx);

    const auto rhs_broadcasting_strategy
            = get_rhs_arg_broadcasting_strategy(post_op.binary.src1_desc,
                    rhs_arg_static_params_.dst_d, supported_strategy_set_);
    const auto rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const auto &vmm_tail_idx = rhs_arg_params.vmm_tail_idx_;
    const bool tail_exists_in_range = !vmm_tail_idx.empty();
    const bool bcast_f32_non_avx512 = !is_avx512_
            && utils::one_of(rhs_broadcasting_strategy,
                    broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc_spatial)
            && rhs_arg_data_type == data_type::f32;
    const bool should_preserve_vmm_tail = tail_exists_in_range
            && (!is_avx512_
                    || !utils::one_of(rhs_broadcasting_strategy,
                            broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc_spatial)
                    || rhs_arg_data_type != data_type::f32);
    const bool dt_helper_vmm_needed
            = !binary_op_with_unaligned_mem_operand_allowed_
            || rhs_arg_data_type != data_type::f32 || bcast_f32_non_avx512
            || should_preserve_vmm_tail;
    const auto tail_load_mode = rhs_arg_params.tail_load_mode;

    // Phase 2 Protect temporary registers content.
    const injector_utils::register_preserve_guard_t register_guard {host_,
            (rhs_arg_static_params_.preserve_gpr_helpers
                            ? std::initializer_list<Xbyak_loongarch64::XReg>(
                                    {rhs_arg_static_params_.rhs_addr_reg,
                                            rhs_arg_static_params_
                                                    .rhs_helper_reg})
                            : std::initializer_list<Xbyak_loongarch64::XReg>()),
            (rhs_arg_static_params_.preserve_vmm_helper && dt_helper_vmm_needed
                            ? std::initializer_list<Xbyak_loongarch64::VReg>({Vmm(vmm_hint).getIdx()})
                            : std::initializer_list<Xbyak_loongarch64::VReg>())};

    bool vmm0_was_preserved = false;
    static const Vmm zero_vmm(0);

    Xbyak_loongarch64::Address rhs_arg_addr(Xbyak_loongarch64::XReg(0), 0);

    // Phase 3 Apply binary post-op over all vmms.
    for (const auto vmm_idx : vmm_idxs) {
        if (vmm_idx == start_idx
                || rhs_arg_params_differ(vmm_idx, vmm_idx - 1, rhs_arg_params,
                        rhs_broadcasting_strategy)) {
            rhs_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy);
        }

        const auto local_vmm_preservation = should_preserve_vmm(
                vmm_idx, vmm_hint, max_vmm_idx, dt_helper_vmm_needed);
        const bool &vmm_preservation_needed = local_vmm_preservation.first;
        const Vmm dst_vmm(vmm_idx);
        const bool with_tail = rhs_arg_static_params_.is_tail
                && vmm_tail_idx.find(vmm_idx) != vmm_tail_idx.cend()
                && IMPLICATION(rhs_broadcasting_strategy
                                == broadcasting_strategy_t::scalar,
                        rhs_arg_static_params_.use_exact_tail_scalar_bcast);

        if (vmm_preservation_needed) {
            const Vmm vmm_to_preserve(local_vmm_preservation.second);
            push_vmm(host_, vmm_to_preserve);
            inject_binary(
                    post_op, dst_vmm, rhs_arg_addr, with_tail, tail_load_mode);
            pop_vmm(host_, vmm_to_preserve);
            // in case all Vmm are occupied, Vmm(0) is chosen for tmp by default,
            // so it's content needs to be preserved...

            push_vmm(host_, zero_vmm);
            vmm0_was_preserved = true;
        } else
            inject_binary(
                    post_op, dst_vmm, rhs_arg_addr, with_tail, tail_load_mode);
    }
    // ...and restored afterwards
    if (vmm0_was_preserved) pop_vmm(host_, zero_vmm);
}

template <cpu_isa_t isa, typename Vmm>
Xbyak_loongarch64::Address jit_uni_binary_injector_t<isa, Vmm>::prepare_rhs_arg_addr(
        std::size_t vmm_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy) const {

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto &abi_param_offset = rhs_arg_static_params_.abi_param_offset;
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;
    const auto rhs_arg_elem_size
            = types::data_type_size(post_op.binary.src1_desc.data_type);

    host_->uni_ld_d(rhs_addr_reg, param1_, abi_param_offset);
    host_->uni_ld_d(rhs_addr_reg, rhs_addr_reg, rhs_arg_idx * rhs_arg_ptr_size);

    switch (rhs_broadcasting_strategy) {
        case broadcasting_strategy_t::scalar: return ptr_b(rhs_addr_reg, 0);
        case broadcasting_strategy_t::no_broadcast: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_out_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_out_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_out_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);

            return ptr_a(rhs_addr_reg, 0);
        }
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_oc_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_oc_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_oc_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);

            return rhs_broadcasting_strategy
                            == broadcasting_strategy_t::per_oc_spatial
                    ? ptr_b(rhs_addr_reg, 0) : ptr_a(rhs_addr_reg, 0);
        }
        case broadcasting_strategy_t::per_mb_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_sp_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_sp_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_sp_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);

            return ptr_a(rhs_addr_reg, 0);
        }
        default: assert(false && "Broadcasting type not supported");
    }

    return ptr_a(rhs_addr_reg, 0);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::append_offset_from_operand(
        const std::map<int, Xbyak_loongarch64::XReg> &vmm_idx_to_elem_operand_off,
        int vmm_idx, const Xbyak_loongarch64::XReg &addr_reg, const Xbyak_loongarch64::XReg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_operand_off = vmm_idx_to_elem_operand_off.find(vmm_idx);
    if (it_operand_off != vmm_idx_to_elem_operand_off.end()) {
        if (elem_size_bytes == 1) {
            host_->add_d(addr_reg, addr_reg, it_operand_off->second);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->add_d(tmp_reg, it_operand_off->second, host_->zero);
            host_->slli_d(tmp_reg, tmp_reg, shift_val);
            host_->add_d(addr_reg, addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::append_offset_under_mem_addr(
        const std::map<int, Xbyak_loongarch64::Address> &vmm_idx_to_elem_addr_off,
        int vmm_idx, const Xbyak_loongarch64::XReg &addr_reg, const Xbyak_loongarch64::XReg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_off_addr = vmm_idx_to_elem_addr_off.find(vmm_idx);
    if (it_off_addr != vmm_idx_to_elem_addr_off.end()) {
        if (elem_size_bytes == 1) {
            host_->add_d(addr_reg, addr_reg, it_off_addr->second.getXReg());
            host_->add_imm(addr_reg, addr_reg,
                            it_off_addr->second.getOffset(), host_->X_TMP_0);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->add_imm(tmp_reg, it_off_addr->second.getXReg(),
                            it_off_addr->second.getOffset(), host_->X_TMP_0);
            host_->slli_d(tmp_reg, tmp_reg, shift_val);
            host_->add_d(addr_reg, addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::append_value_offset(
        const std::map<int, int> &vmm_idx_to_elem_val_off, int vmm_idx,
        const Xbyak_loongarch64::XReg &addr_reg, std::size_t elem_size_bytes) const {

    const auto it_off_val = vmm_idx_to_elem_val_off.find(vmm_idx);
    if (it_off_val != vmm_idx_to_elem_val_off.end())
        host_->add_imm(addr_reg, addr_reg, it_off_val->second * elem_size_bytes, host_->X_TMP_0);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const Xbyak_loongarch64::Address &rhs_addr, bool with_tail,
        const tail_lode_mode_t tail_load_mode) const {

    const auto &alg = post_op.binary.alg;
    const bool cmp_op = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    const auto &rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const bool scalar_f32
            = rhs_addr.getBroadcast() && rhs_arg_data_type == data_type::f32;
    const bool with_tail_not_fusable_to_binary_op
            = with_tail && !(scalar_f32 && is_avx512_);
    const bool process_rhs_arg_using_tmp_vmm
            = rhs_arg_data_type != data_type::f32 || (scalar_f32 && !is_avx512_)
            || with_tail_not_fusable_to_binary_op
            || !binary_op_with_unaligned_mem_operand_allowed_
            || (cmp_op && !is_avx512_);

    if (process_rhs_arg_using_tmp_vmm) {

        const Vmm tmp_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);

        if (rhs_addr.getBroadcast())
            execute_broadcast(rhs_arg_data_type, tmp_vmm,
                    remove_bcast_bit(rhs_addr), with_tail);
        else
            load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr, tail_load_mode,
                    with_tail);

        if (rhs_arg_data_type != data_type::bf16
                && rhs_arg_data_type != data_type::f32)
            cvt_to_f32(tmp_vmm);

        execute_binary(alg, dst, dst, tmp_vmm);
    } else {
        const auto lhs = dst;

        execute_binary(alg, dst, lhs, rhs_addr);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::execute_broadcast(
        const dnnl_data_type_t &data_type, const Vmm &tmp_reg,
        const Xbyak_loongarch64::Address &rhs_addr, bool with_tail) const {
    if (with_tail)
        execute_broadcast_tail(data_type, tmp_reg, rhs_addr);
    else
        execute_broadcast_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::load_rhs(
        const dnnl_data_type_t &data_type, const Vmm &tmp_reg,
        const Xbyak_loongarch64::Address &rhs_addr, const tail_lode_mode_t tail_load_mode,
        bool with_tail) const {
    if (with_tail) {
        if (tail_load_mode == tail_lode_mode_t::DYNAMIC
                || (tail_load_mode == tail_lode_mode_t::DEFAULT
                        && is_avx512_)) {
            load_rhs_tail_dynamically_with_gpr(data_type, tmp_reg);
        } else
            load_rhs_tail_statically(data_type, tmp_reg, rhs_addr);
    } else
        load_rhs_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa, typename Vmm>
Xbyak_loongarch64::Address jit_uni_binary_injector_t<isa, Vmm>::remove_bcast_bit(
        const Xbyak_loongarch64::Address &rhs_addr) const {
    return Xbyak_loongarch64::Address(rhs_addr.getXReg(), rhs_addr.getOffset());
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::cvt_to_f32(const Vmm &tmp_vmm) const {
    host_->uni_xvffint_s_w(tmp_vmm, tmp_vmm);
}

template <>
void jit_uni_binary_injector_t<lsx, Xbyak_loongarch64::VReg>::cvt_to_f32(
        const Xbyak_loongarch64::VReg &tmp_vmm) const {
    host_->vffint_s_w(tmp_vmm, tmp_vmm);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::execute_broadcast_no_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    switch (data_type) {
        case data_type::f32: host_->uni_xvldrepl_w(tmp_vmm, rhs_addr.getXReg(), rhs_addr.getOffset()); break;
        case data_type::s32: host_->uni_xvldrepl_w(tmp_vmm, rhs_addr.getXReg(), rhs_addr.getOffset()); break;
        case data_type::s8:
        case data_type::u8:
            execute_broadcast_s8u8_no_tail(data_type, tmp_vmm, rhs_addr);
            break;
        case data_type::bf16:
                assert(!"unsupported bf16 type");
                break;
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    assert(utils::one_of(data_type, data_type::s8, data_type::u8)
            && "unsupported data type");

    const Xbyak_loongarch64::VReg xmm(tmp_vmm.getIdx());
    const Xbyak_loongarch64::XVReg ymm(tmp_vmm.getIdx());

    host_->uni_ld_b(host_->X_TMP_0, rhs_addr.getXReg(), rhs_addr.getOffset());
    host_->uni_xvinsgr2vr_w(tmp_vmm, host_->X_TMP_0, 0);
    host_->uni_xvreplve0_w(tmp_vmm, tmp_vmm);
}

template <cpu_isa_t isa, typename Vmm>
struct helper_broadcast_s8u8_t {};

template <typename Vmm>
struct helper_broadcast_s8u8_t<lasx, Vmm> {
    static void execute_broadcast_s8u8_no_tail(jit_generator *host,
            const int rhs_helper_reg_idx, const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak_loongarch64::Address &rhs_addr,
            const std::function<void()> &post_process) {

        if (data_type != data_type::s8 && data_type != data_type::u8)
            assert(!"unsupported data type");

        const Xbyak_loongarch64::XReg tmp_reg32 = Xbyak_loongarch64::XReg(rhs_helper_reg_idx);
        const auto tmp_xmm = Xbyak_loongarch64::VReg(tmp_vmm.getIdx());
        const auto tmp_ymm = Xbyak_loongarch64::XVReg(tmp_vmm.getIdx());
        host->uni_ld_b(tmp_reg32, rhs_addr.getXReg(), rhs_addr.getOffset());
        host->vinsgr2vr_w(tmp_xmm, tmp_reg32, 0);
        host->vilvl_b(tmp_xmm, tmp_xmm, tmp_xmm);
        host->vshuf4i_h(Xbyak_loongarch64::VReg(31), tmp_xmm, 0); // low d use low h
        host->vextrins_d(tmp_xmm, Xbyak_loongarch64::VReg(31), 0); // high d keep unchanged
        if (data_type == data_type::s8)
            host->vext2xv_w_b(tmp_ymm, tmp_ymm);
        else
            host->vext2xv_wu_bu(tmp_ymm, tmp_ymm);

        if (post_process) post_process();
    }
};

template <>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::XVReg>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Xbyak_loongarch64::XVReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {

    const auto rhs_helper_reg_idx
            = rhs_arg_static_params_.rhs_helper_reg.getIdx();
    const auto expand_xmm_to_ymm = [&] {
        const auto tmp_ymm = Xbyak_loongarch64::XVReg(tmp_vmm.getIdx());
        host_->xvpermi_q(tmp_vmm, tmp_ymm, 0x02);
    };

    helper_broadcast_s8u8_t<lasx, Xbyak_loongarch64::XVReg>::execute_broadcast_s8u8_no_tail(
            host_, rhs_helper_reg_idx, data_type, tmp_vmm, rhs_addr,
            expand_xmm_to_ymm);
}

template <>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::VReg>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Xbyak_loongarch64::VReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {

    const auto rhs_helper_reg_idx
            = rhs_arg_static_params_.rhs_helper_reg.getIdx();
    helper_broadcast_s8u8_t<lasx, Xbyak_loongarch64::VReg>::execute_broadcast_s8u8_no_tail(
            host_, rhs_helper_reg_idx, data_type, tmp_vmm, rhs_addr, nullptr);
}

template <>
void jit_uni_binary_injector_t<lsx,
        Xbyak_loongarch64::VReg>::execute_broadcast_s8u8_no_tail(const data_type_t
                                                            &data_type,
        const Xbyak_loongarch64::VReg &tmp_vmm, const Xbyak_loongarch64::Address &rhs_addr) const {

    if (data_type == data_type::s8 || data_type == data_type::u8) {
        const auto tmp_reg64_idx
                = rhs_arg_static_params_.rhs_helper_reg.getIdx();
        const Xbyak_loongarch64::XReg tmp_reg32 = Xbyak_loongarch64::XReg(tmp_reg64_idx);
        const Xbyak_loongarch64::XVReg tmp_ymm = Xbyak_loongarch64::XVReg(tmp_vmm.getIdx());
        host_->uni_ld_b(tmp_reg32, rhs_addr.getXReg(), rhs_addr.getOffset());
        host_->vinsgr2vr_w(tmp_vmm, tmp_reg32, 0);
        host_->vilvl_b(tmp_vmm, tmp_vmm, tmp_vmm);
        host_->vshuf4i_h(Xbyak_loongarch64::VReg(31), tmp_vmm, 0); // low d use low h
        host_->vextrins_d(tmp_vmm, Xbyak_loongarch64::VReg(31), 0); // high d keep unchanged
        if (data_type == data_type::s8)
            host_->vext2xv_w_b(tmp_ymm, tmp_ymm);
        else
            host_->vext2xv_wu_bu(tmp_ymm, tmp_ymm);
    } else
        assert(!"unsupported data type");
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {

    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");
    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;

    switch (data_type) {
        case data_type::f32: {
            host_->uni_ld_w(host_->X_TMP_0, rhs_addr.getXReg(), rhs_addr.getOffset());
            host_->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
            for (uint32_t idx = 0; idx < rhs_arg_static_params_.tail_size; ++idx) {
                host_->uni_xvinsgr2vr_w(tmp_vmm, host_->X_TMP_0, idx);
            }
            break;
        }
        case data_type::s32: {
            host_->uni_ld_w(host_->X_TMP_0, rhs_addr.getXReg(), rhs_addr.getOffset());
            host_->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
            for (uint32_t idx = 0; idx < rhs_arg_static_params_.tail_size; ++idx) {
                host_->uni_xvinsgr2vr_w(tmp_vmm, host_->X_TMP_0, idx);
            }
            break;
        }
        case data_type::s8:
        case data_type::u8: {
            host_->uni_ld_b(host_->X_TMP_0, rhs_addr.getXReg(), rhs_addr.getOffset());
            host_->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
            for (uint32_t idx = 0; idx < rhs_arg_static_params_.tail_size; ++idx) {
                host_->uni_xvinsgr2vr_w(tmp_vmm, host_->X_TMP_0, idx);
            }
            break;
        }
        case data_type::bf16:
            assert("unsupported bf16 type!");
            break;
        default: assert(!"unsupported data type");
    }
}

static constexpr int xmm_size_elem = 4;

static void load_tail_avx(jit_generator *host, std::size_t ymm_idx,
        std::size_t tail_size, const std::function<void()> &init_op,
        const std::function<void(int, bool)> &ymm_upper_half_op,
        const std::function<void(int)> &ymm_lower_half_op) {

    if (init_op) init_op();

    const auto res = std::div(tail_size, xmm_size_elem);
    const auto &ymm_upper_half_op_data_size = res.rem;
    const bool should_load_lower_half = res.quot;

    if (ymm_upper_half_op_data_size && ymm_upper_half_op)
        ymm_upper_half_op(ymm_upper_half_op_data_size, should_load_lower_half);

    if (should_load_lower_half) {
        const auto tmp_xmm = Xbyak_loongarch64::VReg(ymm_idx);

        if (ymm_upper_half_op_data_size) push_vmm(host, tmp_xmm);

        if (ymm_lower_half_op) ymm_lower_half_op(ymm_upper_half_op_data_size);

        if (ymm_upper_half_op_data_size) {
            const auto tmp_ymm = Xbyak_loongarch64::XVReg(ymm_idx);
            host->uni_xvld(Xbyak_loongarch64::XVReg(31), host->sp, 0);
            host->xvpermi_q(tmp_ymm, Xbyak_loongarch64::XVReg(31), 0x02);
            restore_stack(host, tmp_xmm);
        }
    }
}

// unused function
//static void load_tail_avx(jit_generator *host, std::size_t ymm_idx,
//        std::size_t tail_size,
//        const std::function<void(int, bool)> &ymm_upper_half_op,
//        const std::function<void(int)> &ymm_lower_half_op) {
//    load_tail_avx(host, ymm_idx, tail_size, nullptr, ymm_upper_half_op,
//            ymm_lower_half_op);
//}

//static Xbyak::uint8 MM_SHUFFLE(
//        Xbyak::uint8 z, Xbyak::uint8 y, Xbyak::uint8 x, Xbyak::uint8 w) {
//    return (((z) << 6) | ((y) << 4) | ((x) << 2) | (w));
//}

static void execute_broadcast_f32_tail_avx(jit_generator *host,
        const Xbyak_loongarch64::XVReg &vmm, const Xbyak_loongarch64::Address &rhs_addr,
        std::size_t tail_size) {

    const auto vmm_idx = vmm.getIdx();
    const auto tmp_xmm = Xbyak_loongarch64::VReg(vmm_idx);

    const auto init_op = [&] {  host->uni_ld_w(host->X_TMP_0, rhs_addr.getXReg(), rhs_addr.getOffset());
                                host->vinsgr2vr_w(tmp_xmm, host->X_TMP_0, 0); };
    const auto upper_half_op
            = [&](int upper_half_data_size, bool should_load_lower_half) {
                  // one element is already loaded
                  if (upper_half_data_size > 1) {
                      // rem is 2 the result is 3,2,0,0;rem is 3 the result is 3,0,0,0
                      if (upper_half_data_size == 2) {
                          // use [0] insert to [1] the other unchanged
                          host->vextrins_w(tmp_xmm, tmp_xmm, 0x10);
                      } else if (upper_half_data_size == 3) {
                          // use [0] insert to [1] [2]
                          host->vextrins_w(tmp_xmm, tmp_xmm, 0x10);
                          host->vextrins_w(tmp_xmm, tmp_xmm, 0x20);
                      }
                  }
              };
    const auto lower_half_op = [&](int upper_half_data_size) {
        host->vreplvei_w(tmp_xmm, tmp_xmm, 0);
    };

    load_tail_avx(
            host, vmm_idx, tail_size, init_op, upper_half_op, lower_half_op);
}

static void execute_broadcast_f32_tail_avx(jit_generator *host,
        const Xbyak_loongarch64::VReg &vmm, const Xbyak_loongarch64::Address &rhs_addr,
        std::size_t tail_size) {

    host->uni_ld_w(host->X_TMP_0, rhs_addr.getXReg(), rhs_addr.getOffset());
    host->vinsgr2vr_w(vmm, host->X_TMP_0, 0);
    // one element is already loaded
    if (tail_size > 1) {
        if (tail_size == 2) {
            host->vextrins_w(vmm, vmm, 0x10);
        } else if (tail_size == 3) {
            host->vextrins_w(vmm, vmm, 0x10);
            host->vextrins_w(vmm, vmm, 0x20);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
struct helper_bcast_tail_t {};

template <typename Vmm>
struct helper_bcast_tail_t<lasx, Vmm> {
    static void execute_broadcast_tail(jit_generator *host,
            const size_t tail_size, const dnnl_data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak_loongarch64::Address &rhs_addr) {
        host->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);

        if (data_type == data_type::f32 || data_type == data_type::s32) {
            execute_broadcast_f32_tail_avx(host, tmp_vmm, rhs_addr, tail_size);
        } else if (data_type == data_type::u8 || data_type == data_type::s8) {
            host->load_bytes_to_dword_extension(tmp_vmm, rhs_addr.getXReg(), rhs_addr.getOffset(),
                                data_type == data_type::s8, tail_size);
        } else
            assert(!"unsupported data type");
    }
};

template <>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::XVReg>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Xbyak_loongarch64::XVReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    const auto &tail_size = rhs_arg_static_params_.tail_size;
    helper_bcast_tail_t<lasx, Xbyak_loongarch64::XVReg>::execute_broadcast_tail(
            host_, tail_size, data_type, tmp_vmm, rhs_addr);
}

template <>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::VReg>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Xbyak_loongarch64::VReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    const auto &tail_size = rhs_arg_static_params_.tail_size;
    helper_bcast_tail_t<lasx, Xbyak_loongarch64::VReg>::execute_broadcast_tail(
            host_, tail_size, data_type, tmp_vmm, rhs_addr);
}

template <>
void jit_uni_binary_injector_t<lsx, Xbyak_loongarch64::VReg>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Xbyak_loongarch64::VReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {

    host_->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
    const auto &tail_size = rhs_arg_static_params_.tail_size;
    if (data_type == data_type::f32 || data_type == data_type::s32) {
        host_->uni_ld_w(host_->X_TMP_0, rhs_addr.getXReg(), rhs_addr.getOffset());
        host_->vinsgr2vr_w(tmp_vmm, host_->X_TMP_0, 0);
        if (tail_size > 1) {
            if (tail_size == 2) {
                host_->vextrins_w(tmp_vmm, tmp_vmm, 0x10);
            } else if (tail_size == 3) {
                host_->vextrins_w(tmp_vmm, tmp_vmm, 0x10);
                host_->vextrins_w(tmp_vmm, tmp_vmm, 0x20);
            }
        }
    } else if (data_type == data_type::u8 || data_type == data_type::s8) {
        host_->load_bytes_to_dword_extension(tmp_vmm, rhs_addr.getXReg(), rhs_addr.getOffset(),
                                data_type == data_type::s8, tail_size);
    } else
        assert(!"unsupported data type");
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::load_rhs_no_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32: host_->uni_xvld(tmp_vmm, rhs_addr.getXReg(), rhs_addr.getOffset()); break;
        case data_type::s8:
        case data_type::u8:
            load_rhs_i8_no_tail(data_type, tmp_vmm, rhs_addr);
            break;
        case data_type::bf16:
            assert(!"unsupported bf16 type");
            break;
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::load_rhs_i8_no_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    host_->load_bytes_to_dword_extension(tmp_vmm, rhs_addr.getXReg(), rhs_addr.getOffset(),
                                data_type == data_type::s8, tmp_vmm.getBit() / sizeof(int32_t));
}

template <>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::XVReg>::load_rhs_i8_no_tail(
        const dnnl_data_type_t &data_type, const Xbyak_loongarch64::XVReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    static constexpr int xmm_size_elem = 4;
    static constexpr int one_load_size = xmm_size_elem * sizeof(uint8_t);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto tmp_xmm = Xbyak_loongarch64::VReg(tmp_vmm.getIdx());

    host_->load_bytes_to_dword_extension(tmp_xmm, rhs_addr_reg, one_load_size,
                                data_type == data_type::s8, tmp_xmm.getBit() / sizeof(float));
    push_vmm(host_, tmp_xmm);
    //load_i8_fn(rhs_addr);
    host_->load_bytes_to_dword_extension(tmp_xmm, rhs_addr.getXReg(), rhs_addr.getOffset(),
                                data_type == data_type::s8, tmp_xmm.getBit() / sizeof(float));
    host_->uni_xvld(Xbyak_loongarch64::XVReg(31), host_->sp, 0);
    host_->xvpermi_q(tmp_vmm, Xbyak_loongarch64::XVReg(31), 0x02);
    restore_stack(host_, tmp_xmm);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::load_rhs_tail_dynamically_with_gpr(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm) const {
    const bool is_ymm = std::is_same<Vmm, Xbyak_loongarch64::XVReg>::value;
    const Xbyak_loongarch64::XReg &reg_addr = rhs_arg_static_params_.rhs_addr_reg;
    const Xbyak_loongarch64::XReg &reg_tmp = rhs_arg_static_params_.rhs_helper_reg;
    const Xbyak_loongarch64::XReg &reg_tail_size = rhs_arg_static_params_.reg_tail_size;
    const Xbyak_loongarch64::VReg x = Xbyak_loongarch64::VReg(tmp_vmm.getIdx());
    const Xbyak_loongarch64::XVReg y = Xbyak_loongarch64::XVReg(tmp_vmm.getIdx());

    auto runtime_tail_load = [&](int load_size) {
        if (is_ymm)
            host_->load_data(data_type, y, reg_addr, 0, load_size);
        else
            host_->load_data(data_type, x, reg_addr, 0, load_size);
    };

    host_->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
    host_->runtime_tail_process<Vmm>(reg_tail_size, reg_tmp, runtime_tail_load);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::load_rhs_tail_statically(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    assert(!"unsupported tail load mode");
}
template <cpu_isa_t isa, typename Vmm>
struct helper_load_tail_t {};

template <typename Vmm>
struct helper_load_tail_t<lasx, Vmm> {
    static void load_rhs_tail_statically(jit_generator *host,
            const size_t tail_size, const Xbyak_loongarch64::XReg &rhs_addr_reg,
            const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
            const Xbyak_loongarch64::Address &rhs_addr) {
        if (!utils::one_of(data_type, data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8))
            assert(!"unsupported data type");

        host->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
        host->load_data(data_type, tmp_vmm, rhs_addr_reg, 0, tail_size);
    }
};

template <>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::XVReg>::load_rhs_tail_statically(
        const dnnl_data_type_t &data_type, const Xbyak_loongarch64::XVReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {

    const auto &tail_size = rhs_arg_static_params_.tail_size;
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    helper_load_tail_t<lasx, Xbyak_loongarch64::XVReg>::load_rhs_tail_statically(
            host_, tail_size, rhs_addr_reg, data_type, tmp_vmm, rhs_addr);
}

template <>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::VReg>::load_rhs_tail_statically(
        const dnnl_data_type_t &data_type, const Xbyak_loongarch64::VReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {

    const auto &tail_size = rhs_arg_static_params_.tail_size;
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    helper_load_tail_t<lasx, Xbyak_loongarch64::VReg>::load_rhs_tail_statically(
            host_, tail_size, rhs_addr_reg, data_type, tmp_vmm, rhs_addr);
}

template <>
void jit_uni_binary_injector_t<lsx, Xbyak_loongarch64::VReg>::load_rhs_tail_statically(
        const dnnl_data_type_t &data_type, const Xbyak_loongarch64::VReg &tmp_vmm,
        const Xbyak_loongarch64::Address &rhs_addr) const {
    if (!utils::one_of(data_type, data_type::f32, data_type::s32, data_type::s8,
                data_type::u8))
        assert(!"unsupported data type");

    const auto &tail_size = rhs_arg_static_params_.tail_size;
    host_->uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
    host_->load_data(data_type, tmp_vmm, rhs_arg_static_params_.rhs_addr_reg, 0,
            tail_size);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::execute_cmp_binary(const Vmm &dst) const {
    const uint32_t vmm_idx = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    const Vmm vreg_one = Vmm(vmm_idx);
    const Xbyak_loongarch64::XReg reg_tmp = rhs_arg_static_params_.rhs_helper_reg;

    host_->mov_imm(reg_tmp, float2int(1));
    host_->uni_replgr2vr_w(vreg_one, reg_tmp);
    host_->uni_fmin_s(dst, dst, vreg_one);
}

template <cpu_isa_t isa, typename Vmm>
template <typename T>
void jit_uni_binary_injector_t<isa, Vmm>::execute_binary(alg_kind_t binary_alg,
        const Vmm &dst, const Vmm &lhs, const T &rhs) const {
    if (std::is_same<T, Xbyak_loongarch64::Address>::value) {
        const Xbyak_loongarch64::Address &addr = (const Xbyak_loongarch64::Address&)rhs;
        host_->uni_xvld(Vmm(31), addr.getXReg(), addr.getOffset());
    } else if (std::is_same<T, Xbyak_loongarch64::XVReg>::value) {
        const Xbyak_loongarch64::XVReg &xvreg = (const Xbyak_loongarch64::XVReg&)rhs;
        host_->uni_bsll_v(Xbyak_loongarch64::XVReg(31), xvreg, 0);
    } else if (std::is_same<T, Xbyak_loongarch64::VReg>::value ) {
        const Xbyak_loongarch64::VReg &vreg = (const Xbyak_loongarch64::VReg&)rhs;
        host_->uni_bsll_v(Xbyak_loongarch64::VReg(31), vreg, 0);
    }

    switch (binary_alg) {
        case alg_kind::binary_add: host_->uni_fadd_s(dst, lhs, Vmm(31)); break;
        case alg_kind::binary_mul: host_->uni_fmul_s(dst, lhs, Vmm(31)); break;
        case alg_kind::binary_max: host_->uni_fmax_s(dst, lhs, Vmm(31)); break;
        case alg_kind::binary_min: host_->uni_fmin_s(dst, lhs, Vmm(31)); break;
        case alg_kind::binary_div: host_->uni_fdiv_s(dst, lhs, Vmm(31)); break;
        case alg_kind::binary_sub: host_->uni_fsub_s(dst, lhs, Vmm(31)); break;
        case alg_kind::binary_ge:
            host_->uni_fcmp_cle_s(dst, Vmm(31), lhs);
            execute_cmp_binary(dst);
            break;
        case alg_kind::binary_gt:
            host_->uni_fcmp_clt_s(dst, Vmm(31), lhs);
            execute_cmp_binary(dst);
            break;
        case alg_kind::binary_le:
            host_->uni_fcmp_cle_s(dst, lhs, Vmm(31));
            execute_cmp_binary(dst);
            break;
        case alg_kind::binary_lt:
            host_->uni_fcmp_clt_s(dst, lhs, Vmm(31));
            execute_cmp_binary(dst);
            break;
        case alg_kind::binary_eq:
            host_->uni_fcmp_ceq_s(dst, lhs, Vmm(31));
            execute_cmp_binary(dst);
            break;
        case alg_kind::binary_ne:
            host_->uni_fcmp_cne_s(dst, lhs, Vmm(31));
            execute_cmp_binary(dst);
            break;
        default: assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa, typename Vmm>
struct helper_binary_t {};

template <typename Vmm>
struct helper_binary_t<lasx, Vmm> {
    template <typename T, typename F>
    static void execute_binary(jit_generator *host, F execute_cmp_binary,
            alg_kind_t binary_alg, const Vmm &dst, const Vmm &lhs,
            const T &rhs) {
        if (std::is_same<T, Xbyak_loongarch64::Address>::value) {
            const Xbyak_loongarch64::Address &addr = (const Xbyak_loongarch64::Address&)rhs;
            host->uni_xvld(Vmm(31), addr.getXReg(), addr.getOffset());
        } else if (std::is_same<T, Xbyak_loongarch64::XVReg>::value) {
            const Xbyak_loongarch64::XVReg &xvreg = (const Xbyak_loongarch64::XVReg&)rhs;
            host->uni_bsll_v(Xbyak_loongarch64::XVReg(31), xvreg, 0);
        } else if (std::is_same<T, Xbyak_loongarch64::VReg>::value ) {
            const Xbyak_loongarch64::VReg &vreg = (const Xbyak_loongarch64::VReg&)rhs;
            host->uni_bsll_v(Xbyak_loongarch64::VReg(31), vreg, 0);
        }
        switch (binary_alg) {
            case alg_kind::binary_add: host->uni_fadd_s(dst, lhs, Vmm(31)); break;
            case alg_kind::binary_mul: host->uni_fmul_s(dst, lhs, Vmm(31)); break;
            case alg_kind::binary_max: host->uni_fmax_s(dst, lhs, Vmm(31)); break;
            case alg_kind::binary_min: host->uni_fmin_s(dst, lhs, Vmm(31)); break;
            case alg_kind::binary_div: host->uni_fdiv_s(dst, lhs, Vmm(31)); break;
            case alg_kind::binary_sub: host->uni_fsub_s(dst, lhs, Vmm(31)); break;
            case alg_kind::binary_ge:
                host->uni_fcmp_cle_s(dst, Vmm(31), lhs);
                execute_cmp_binary(dst);
                break;
            case alg_kind::binary_gt:
                host->uni_fcmp_clt_s(dst, Vmm(31), lhs);
                execute_cmp_binary(dst);
                break;
            case alg_kind::binary_le:
                host->uni_fcmp_cle_s(dst, lhs, Vmm(31));
                execute_cmp_binary(dst);
                break;
            case alg_kind::binary_lt:
                host->uni_fcmp_clt_s(dst, lhs, Vmm(31));
                execute_cmp_binary(dst);
                break;
            case alg_kind::binary_eq:
                host->uni_fcmp_ceq_s(dst, lhs, Vmm(31));
                execute_cmp_binary(dst);
                break;
            case alg_kind::binary_ne:
                host->uni_fcmp_cne_s(dst, lhs, Vmm(31));
                execute_cmp_binary(dst);
                break;
            default: assert(!"unsupported algorithm");
        }
    }
};

template <>
template <typename T>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::XVReg>::execute_binary(
        alg_kind_t binary_alg, const Xbyak_loongarch64::XVReg &dst, const Xbyak_loongarch64::XVReg &lhs,
        const T &rhs) const {

    const auto execute_cmp_binary_lam
            = [this](const Xbyak_loongarch64::XVReg &dst) {
                  this->execute_cmp_binary(dst);
              };
    helper_binary_t<lasx, Xbyak_loongarch64::XVReg>::execute_binary<T>(
            host_, execute_cmp_binary_lam, binary_alg, dst, lhs, rhs);
}

template <>
template <typename T>
void jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::VReg>::execute_binary(
        alg_kind_t binary_alg, const Xbyak_loongarch64::VReg &dst, const Xbyak_loongarch64::VReg &lhs,
        const T &rhs) const {

    const auto execute_cmp_binary_lam
            = [this](const Xbyak_loongarch64::VReg &dst) {
                  this->execute_cmp_binary(dst);
              };
    helper_binary_t<lasx, Xbyak_loongarch64::VReg>::execute_binary<T>(
            host_, execute_cmp_binary_lam, binary_alg, dst, lhs, rhs);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_binary_injector_t<isa, Vmm>::compute_vector(size_t idx,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    compute_vector_range({idx}, rhs_arg_idx, post_op, rhs_arg_params);
}

template class jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::XVReg>;
template class jit_uni_binary_injector_t<lasx, Xbyak_loongarch64::VReg>;
//template class jit_uni_binary_injector_t<lasx>;
template class jit_uni_binary_injector_t<lsx, Xbyak_loongarch64::VReg>;
//template class jit_uni_binary_injector_t<lsx>;

} // namespace binary_injector
} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
