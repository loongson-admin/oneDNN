/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/jit/xbyak/ir/xbyak_visitor.hpp>
#include <compiler/jit/xbyak/x86_64/registers.hpp>
#include <util/any_map.hpp>
#include <util/array_ref.hpp>

#include "x86_intrinsics_lowering.hpp"

namespace sc {
namespace sc_xbyak {

class x86_intrinsics_lowering_impl_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    x86_intrinsics_lowering_impl_t(
            const sc::runtime::target_machine_t &target_machine)
        : target_machine_(target_machine) {
        // TODO(XXX): support AVX and SSE
        COMPILE_ASSERT(target_machine_.cpu_flags_.fAVX, "Support AVX");
    }

    std::vector<stmt> transform_seq_;

    expr_c dispatch(expr_c v) override { return v; }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt> new_seq;
        for (auto &s : v->seq_) {
            auto ss = xbyak_visitor_t::dispatch(s);
            if (!transform_seq_.empty()) {
                for (auto &&ts : transform_seq_) {
                    new_seq.push_back(std::move(ts));
                }
                transform_seq_.clear();
            } else {
                new_seq.emplace_back(ss.remove_const());
            }
        }
        return copy_attr(*v, make_stmt<stmts_node_t>(std::move(new_seq)));
    }

    stmt_c visit(assign_c v) override {
        x86_intrinsics_transform(v->var_, v->value_);
        return std::move(v);
    }

    stmt_c visit(define_c v) override {
        if (v->var_.isa<var>() && v->init_.defined()) {
            add_defination(v->var_, v->linkage_);
            x86_intrinsics_transform(v->var_, v->init_);
        }
        return std::move(v);
    }

    void x86_intrinsics_transform(const expr &dst, const expr &val) {
        switch (val->node_type_) {
            case sc_expr_type::add: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::add),
                        transform_intrin(xbyak_intrin_type::add));
            } break;
            case sc_expr_type::sub: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::sub),
                        transform_intrin(xbyak_intrin_type::sub));
            } break;
            case sc_expr_type::mul: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_x86_mul(),
                        transform_intrin(xbyak_intrin_type::mul));
            } break;
            case sc_expr_type::div: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_x86_mod_div(xbyak_intrin_type::div),
                        transform_intrin(xbyak_intrin_type::div));
            } break;
            case sc_expr_type::mod: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_x86_mod_div(xbyak_intrin_type::mod),
                        transform_disabled("mod"));
            } break;
            case sc_expr_type::logic_and: {
                auto log = val.static_as<logic>();
                transform(dst, {log->l_, log->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_and),
                        transform_disabled("logic_and"));
            } break;
            case sc_expr_type::logic_or: {
                auto log = val.static_as<logic>();
                transform(dst, {log->l_, log->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_or),
                        transform_disabled("logic_or"));
            } break;
            case sc_expr_type::logic_not: {
                auto log = val.static_as<logic_not>();
                transform(dst, {log->in_, builder::make_constant(UINT64_C(1))},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_xor),
                        transform_disabled("logic_not"));
            } break;
            case sc_expr_type::select: {
                auto sel = val.static_as<select>();
                transform_select(dst, sel);
            } break;
            case sc_expr_type::intrin_call: {
                auto intrin = val.static_as<intrin_call>();
                transform_intrin_call(dst, intrin);
            } break;
            case sc_expr_type::cast: {
                auto src = val.static_as<cast>();
                transform_cast(dst, src);
            } break;
            case sc_expr_type::cmp_eq:
            case sc_expr_type::cmp_ne:
            case sc_expr_type::cmp_lt:
            case sc_expr_type::cmp_le:
            case sc_expr_type::cmp_gt:
            case sc_expr_type::cmp_ge: {
                auto src = val.static_as<cmp>();
                transform_cmp_set(dst, src, val->node_type_);
            } break;
            default: {
                add_assignment(dst, val);
            } break;
        }
    }

    void transform_intrin_call(const expr &dst, const intrin_call &intrin) {
        switch (intrin->type_) {
            case intrin_type::min: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_min_max(xbyak_intrin_type::min),
                        transform_intrin(xbyak_intrin_type::min));
            } break;
            case intrin_type::max: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_min_max(xbyak_intrin_type::max),
                        transform_intrin(xbyak_intrin_type::max));
            } break;
            case intrin_type::shl: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_shift(xbyak_intrin_type::shl),
                        transform_intrin(xbyak_intrin_type::shl));
            } break;
            case intrin_type::shr: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_shift(xbyak_intrin_type::shr),
                        transform_intrin(xbyak_intrin_type::shr));
            } break;
            case intrin_type::ceil: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("ceil"),
                        transform_intrin(xbyak_intrin_type::ceil));
            } break;
            case intrin_type::floor: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("floor"),
                        transform_intrin(xbyak_intrin_type::floor));
            } break;
            case intrin_type::sqrt: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("sqrt"),
                        transform_intrin(xbyak_intrin_type::sqrt));
            } break;
            case intrin_type::rsqrt: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("rsqrt"),
                        transform_intrin(xbyak_intrin_type::rsqrt));
            } break;
            case intrin_type::fmadd: {
                transform(dst,
                        {intrin->args_[0], intrin->args_[1], intrin->args_[2]},
                        dst->dtype_, //
                        transform_disabled("fmadd"),
                        transform_4a_to_3a(xbyak_intrin_type::fmadd));
            } break;
            case intrin_type::broadcast: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("broadcast"),
                        transform_intrin(xbyak_intrin_type::broadcast));
            } break;
            case intrin_type::reduce_add: {
                transform(dst, {intrin->args_[0]},
                        intrin->args_[0]->dtype_, //
                        transform_disabled("reduce_add"),
                        transform_simd_reduce_seq(xbyak_intrin_type::add));
            } break;
            case intrin_type::reduce_mul: {
                transform(dst, {intrin->args_[0]},
                        intrin->args_[0]->dtype_, //
                        transform_disabled("reduce_mul"),
                        transform_simd_reduce_seq(xbyak_intrin_type::mul));
            } break;
            case intrin_type::unpack_low: {
                auto imm = intrin->intrin_attrs_->get<int>("elem_bits");
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm)},
                        dst->dtype_, //
                        transform_disabled("unpack_low"),
                        transform_intrin(xbyak_intrin_type::unpack_low));
            } break;
            case intrin_type::unpack_high: {
                auto imm = intrin->intrin_attrs_->get<int>("elem_bits");
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm)},
                        dst->dtype_, //
                        transform_disabled("unpack_high"),
                        transform_intrin(xbyak_intrin_type::unpack_high));
            } break;
            case intrin_type::shuffle: {
                auto imm = intrin->intrin_attrs_->get<int>("shuffle_imm");
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm)},
                        dst->dtype_, //
                        transform_disabled("shuffle"),
                        transform_intrin(xbyak_intrin_type::shuffle));
            } break;
            case intrin_type::permute: {
                auto imm = intrin->intrin_attrs_->get<int>("permute_imm");
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm)},
                        dst->dtype_, //
                        transform_disabled("permute"),
                        transform_intrin(xbyak_intrin_type::permute));
            } break;
            case intrin_type::int_and: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_and),
                        transform_intrin(xbyak_intrin_type::bit_and));
            } break;
            case intrin_type::int_or: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_or),
                        transform_intrin(xbyak_intrin_type::bit_or));
            } break;
            case intrin_type::int_xor: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_xor),
                        transform_intrin(xbyak_intrin_type::bit_xor));
            } break;
            case intrin_type::reinterpret: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_intrin(xbyak_intrin_type::reinterpret),
                        transform_intrin(xbyak_intrin_type::reinterpret));
            } break;
            case intrin_type::permutex2var: {
                transform(dst,
                        {intrin->args_[0], intrin->args_[1], intrin->args_[2]},
                        dst->dtype_, //
                        transform_disabled("permutex2var"),
                        transform_4a_to_3a(xbyak_intrin_type::permutex2var));
            } break;
            case intrin_type::saturated_cast: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("saturated_cast"),
                        transform_intrin(xbyak_intrin_type::saturated_cast));
            } break;
            case intrin_type::round_and_cast: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("round_and_cast"),
                        transform_intrin(xbyak_intrin_type::round_and_cast));
            } break;
            default: add_assignment(dst, intrin); break;
        }
    }

    void transform_cast(const expr &dst, const cast &src) {
        const sc_data_type_t src_dtype = src->in_->dtype_;
        const sc_data_type_t dst_dtype = src->dtype_;
        if (dst_dtype == sc_data_type_t::bf16(1)
                && src_dtype == sc_data_type_t::f32(1)) {
            auto bias = builder::make_var(sc_data_type_t::u32(1), "bias");
            auto temp = builder::make_var(sc_data_type_t::u32(1), "temp");
            add_defination(bias, linkage::local);
            add_defination(temp, linkage::local);
            // temp = reinterpret_cast<u32>(src);
            transform_intrin(temp, {src->in_}, xbyak_intrin_type::reinterpret,
                    xbyak_intrin_isa::x86);
            // bias = temp;
            add_assignment(bias, temp);
            // bias = bias >> 16
            transform_intrin(bias, {builder::make_constant(UINT64_C(16))},
                    xbyak_intrin_type::shr, xbyak_intrin_isa::x86);
            // bias = bias & 1
            transform_intrin(bias, {builder::make_constant(UINT64_C(1))},
                    xbyak_intrin_type::bit_and, xbyak_intrin_isa::x86);
            // bias = bias + 0x7FFF
            transform_intrin(bias,
                    {builder::make_constant(
                            {UINT64_C(32767)}, sc_data_type_t::u32(1))},
                    xbyak_intrin_type::add, xbyak_intrin_isa::x86);
            // temp = temp + bias
            transform_intrin(temp, {bias}, xbyak_intrin_type::add,
                    xbyak_intrin_isa::x86);
            // temp = temp >> 16
            transform_intrin(temp, {builder::make_constant(UINT64_C(16))},
                    xbyak_intrin_type::shr, xbyak_intrin_isa::x86);
            // dst = u16(temp)
            add_assignment(dst, builder::make_cast(dst_dtype, temp));
        } else {
            add_assignment(dst, src);
        }
    }

    void transform_select(const expr &dst, const select &sel) {
        const sc_data_type_t dst_dtype = dst->dtype_;
        // If f32x16 blend cond is not mask16 type, cast to mask16
        auto get_cond = [&](expr cond, sc_data_type_t dtype) {
            if (dtype.lanes_ == 16
                    && cond->dtype_ != sc_data_type_t::boolean(16)) {
                auto mask16 = builder::make_var(
                        sc_data_type_t::boolean(16), "__mmask16");
                add_defination(mask16, linkage::local);
                add_assignment(mask16,
                        builder::make_cast(sc_data_type_t::boolean(16), cond));
                return mask16;
            } else {
                return cond;
            }
        };
        // If x86 select rhs is constant, load to var
        auto get_rhs = [&](expr r) {
            if (r.isa<constant>()) {
                auto rhs = builder::make_var(r->dtype_, "__sel_tmp_var");
                add_defination(rhs, linkage::local);
                add_assignment(rhs, r);
                return rhs;
            } else {
                return r;
            }
        };
        // Transform select
        auto cond = get_cond(sel->cond_, dst_dtype);
        auto zero = is_const_zero(sel->r_);
        auto isa = convert_x86_operation(dst_dtype) ? xbyak_intrin_isa::x86
                                                    : xbyak_intrin_isa::avx;
        if (cond->dtype_ == datatypes::boolean) {
            // test cond, cond
            transform_intrin(cond, {cond}, xbyak_intrin_type::test,
                    xbyak_intrin_isa::x86);
            // TODO(XXX): fix x86 select:
            // x86 conditional move only accepts cmovcc(r, r/m)
            if (isa == xbyak_intrin_isa::x86) {
                // temporary solution for cmov
                auto rax = make_physical_reg(dst_dtype, x86_64::regs::rax);
                add_defination(rax, linkage::local);
                // rax = lhs
                add_assignment(rax, sel->l_);
                // if(cond = false) rax = rhs
                add_assignment(rax,
                        make_xbyak_intrin(dst_dtype, {get_rhs(sel->r_)},
                                xbyak_intrin_type::cmov, isa,
                                xbyak_intrin_modifier(xbyak_condition::eq)));
                add_assignment(dst, rax);
            } else {
                // dst = lhs
                add_assignment(dst, sel->l_);
                // if(cond = false) dst = rhs
                add_assignment(dst,
                        make_xbyak_intrin(dst_dtype, {sel->r_},
                                xbyak_intrin_type::cmov, isa,
                                xbyak_intrin_modifier(xbyak_condition::eq)));
            }
        } else {
            if (zero) {
                // avx512 zero mask move
                add_assignment(dst,
                        make_xbyak_intrin(dst_dtype, {sel->l_},
                                xbyak_intrin_type::zero_mov, isa,
                                xbyak_intrin_modifier(cond, zero)));
            } else {
                // blend order reversed: select second operand if cond is true
                add_assignment(dst,
                        make_xbyak_intrin(dst_dtype, {sel->r_, sel->l_},
                                xbyak_intrin_type::blend, isa,
                                xbyak_intrin_modifier(cond, zero)));
            }
        }
    }

    void transform_cmp_set(const expr &dst, const cmp &src, sc_expr_type t) {
        const sc_data_type_t cmp_dtype = src->l_->dtype_;
        auto isa = convert_x86_operation(cmp_dtype) ? xbyak_intrin_isa::x86
                                                    : xbyak_intrin_isa::avx;
        add_assignment(dst,
                make_xbyak_intrin(dst->dtype_, {src->l_, src->r_},
                        xbyak_intrin_type::cmp_set, isa,
                        xbyak_intrin_modifier(get_xbyak_condition(t))));
    }

    // --------------------------------
    // Transform operations and lambdas
    // --------------------------------

    using transform_func = std::function<void(
            const expr &, array_ref<expr>, sc_data_type_t, xbyak_intrin_isa)>;

    void transform(const expr &dst, array_ref<expr> args, sc_data_type_t dtype,
            const transform_func &scalar_f, const transform_func &vec_f) {
        if (convert_x86_operation(dtype)) {
            scalar_f(dst, args, dtype, xbyak_intrin_isa::x86);
        } else {
            vec_f(dst, args, dtype, xbyak_intrin_isa::avx);
        }
    }

    transform_func transform_disabled(const std::string &str) {
        return [str](const expr &dst, array_ref<expr> src, sc_data_type_t dtype,
                       xbyak_intrin_isa isa) {
            COMPILE_ASSERT(false, "Transform disabled: " << str);
        };
    }

    transform_func transform_intrin(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_intrin(dst, src, intrin, isa);
        };
    }

    transform_func transform_3a_to_2a(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_3a_to_2a(dst, src[0], src[1], intrin, isa);
        };
    }

    transform_func transform_4a_to_3a(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_4a_to_3a(dst, src[0], src[1], src[2], intrin, isa);
        };
    }

    transform_func transform_x86_mul() {
        return [this](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_mul(dst, src[0], src[1]);
        };
    }

    transform_func transform_x86_shift(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_shift(dst, src[0], src[1], intrin);
        };
    }

    transform_func transform_x86_min_max(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_min_max(dst, src[0], src[1], intrin);
        };
    }

    transform_func transform_x86_mod_div(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_mod_div(dst, src[0], src[1], dtype, intrin);
        };
    }

    transform_func transform_simd_reduce_seq(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_simd_reduce_seq(dst, src[0], dtype, intrin, isa);
        };
    }

    // -----------------
    // Transform helpers
    // -----------------

    void transform_intrin(const expr &v, array_ref<expr> args,
            xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        add_assignment(
                v, make_xbyak_intrin(v->dtype_, args.as_vector(), intrin, isa));
    }

    void transform_3a_to_2a(const expr &dst, const expr &lhs, const expr &rhs,
            xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        // dst = lhs
        add_assignment(dst, lhs);
        // dst = 2a(rhs)
        transform_intrin(dst, {rhs}, intrin, isa);
    }

    void transform_4a_to_3a(const expr &dst, const expr &a, const expr &b,
            const expr &c, xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        // dst = a
        add_assignment(dst, a);
        // dst = 3a(b, c)
        transform_intrin(dst, {b, c}, intrin, isa);
    }

    void transform_x86_mul(const expr &dst, const expr &lhs, const expr &rhs) {
        if (rhs.isa<constant>()) {
            // mul constant always rhs
            transform_intrin(dst, {lhs, rhs}, xbyak_intrin_type::muli,
                    xbyak_intrin_isa::x86);
        } else {
            transform_3a_to_2a(dst, lhs, rhs, xbyak_intrin_type::mul,
                    xbyak_intrin_isa::x86);
        }
    }

    void transform_x86_shift(const expr &dst, const expr &lhs, const expr &rhs,
            xbyak_intrin_type intrin) {
        if (rhs.isa<constant>()) {
            transform_3a_to_2a(dst, lhs, rhs, intrin, xbyak_intrin_isa::x86);
        } else {
            // if shift rhs is not a imm, must use reg CL to store
            auto rcx = make_physical_reg(rhs->dtype_, x86_64::regs::rcx);
            add_defination(rcx, linkage::local);
            add_assignment(rcx, rhs);
            transform_3a_to_2a(dst, lhs, rcx, intrin, xbyak_intrin_isa::x86);
        }
    }

    void transform_x86_min_max(const expr &dst, const expr &lhs,
            const expr &rhs, xbyak_intrin_type intrin) {
        // dst = lhs
        add_assignment(dst, lhs);
        // dst = min/max(rhs)
        if (rhs.isa<constant>()) {
            auto tmp = builder::make_var(rhs->dtype_, "__imm_tmp_var");
            add_defination(tmp, linkage::local);
            add_assignment(tmp, rhs);
            transform_intrin(dst, {tmp}, intrin, xbyak_intrin_isa::x86);
        } else {
            transform_intrin(dst, {rhs}, intrin, xbyak_intrin_isa::x86);
        }
    }

    void transform_x86_mod_div(const expr &dst, const expr &lhs,
            const expr &rhs, sc_data_type_t dtype, xbyak_intrin_type intrin) {
        // Physical regs used by mod/div
        auto rax = make_physical_reg(dtype, x86_64::regs::rax);
        auto rdx = make_physical_reg(dtype, x86_64::regs::rdx);
        add_defination(rax, linkage::local);
        add_defination(rdx, linkage::local);
        // mov %rax, lhs
        auto mov_rax = make_stmt<assign_node_t>(rax, lhs);
        transform_seq_.emplace_back(mov_rax);
        // unsigned [xor %rdx, %rdx] or signed [CWD/CDQ/CQO]
        auto sign_ext_rdx = make_stmt<assign_node_t>(rdx,
                make_xbyak_intrin(
                        dst->dtype_, {rax}, xbyak_intrin_type::sign_ext));
        transform_seq_.emplace_back(sign_ext_rdx);
        // div rhs
        // mov dst, %rax(div)/%rdx(mod)
        // Add %rax, %rdx to xbyak_intrin args so, liveness updates
        // TODO(XXX): if transform normal div constant (not unsigned 2^n) is
        // worthy
        if (rhs.isa<constant>()) {
            auto tmp = builder::make_var(rhs->dtype_, "div_imm_tmp");
            add_defination(tmp, linkage::local);
            add_assignment(tmp, rhs);
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_, {tmp, rax, rdx}, intrin,
                            xbyak_intrin_isa::x86));
        } else {
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_, {rhs, rax, rdx}, intrin,
                            xbyak_intrin_isa::x86));
        }
    }

    void transform_simd_reduce_seq(const expr &dst, const expr &src,
            sc_data_type_t dtype, xbyak_intrin_type intrin,
            xbyak_intrin_isa isa) {
        auto lanes = dtype.lanes_;
        auto current_src = src;
        // lanes must be power of 2
        assert((lanes > 1) && ((lanes & (lanes - 1)) == 0));
        // reduce lanes down to 1
        while (lanes > 1) {
            lanes = lanes / 2;
            dtype.lanes_ = lanes;
            // hig, low
            auto extract_hig = builder::make_var(
                    dtype, "extract_hig_" + std::to_string(lanes));
            auto extract_low = builder::make_var(
                    dtype, "extract_low_" + std::to_string(lanes));
            add_defination(extract_hig, linkage::local);
            add_defination(extract_low, linkage::local);
            // hig = ext_h(src)
            add_assignment(extract_hig,
                    make_xbyak_intrin(dtype, {current_src},
                            xbyak_intrin_type::extract_high, isa));
            // low = ext_l(src)
            add_assignment(extract_low,
                    make_xbyak_intrin(dtype, {current_src},
                            xbyak_intrin_type::extract_low, isa));
            // hig = op(low, hig)
            if (lanes == 1) {
                if (convert_x86_operation(dtype)) {
                    assert(intrin == xbyak_intrin_type::add
                            || intrin == xbyak_intrin_type::mul);
                    // e.g. if reduce_add for s32x16 reach s32x1
                    // The calculation cannot use simd intrins any more
                    add_assignment(dst, extract_low);
                    add_assignment(dst,
                            make_xbyak_intrin(dst->dtype_, {extract_hig},
                                    intrin, xbyak_intrin_isa::x86));
                } else {
                    add_assignment(dst,
                            make_xbyak_intrin(dst->dtype_,
                                    {extract_low, extract_hig}, intrin, isa));
                }
            } else {
                add_assignment(extract_hig,
                        make_xbyak_intrin(extract_hig->dtype_,
                                {extract_low, extract_hig}, intrin, isa));
            }
            // src = hig
            current_src = extract_hig;
        }
    }

    // --------------
    // Intrin helpers
    // --------------

    void add_assignment(const expr &var, const expr &value) {
        transform_seq_.emplace_back(make_stmt<assign_node_t>(var, value));
    }

    void add_defination(const expr &var, linkage link) {
        transform_seq_.emplace_back(
                make_stmt<define_node_t>(var, link, expr()));
    }

    bool convert_x86_operation(const sc_data_type_t &dtype) {
        return !is_simd_data(dtype);
    }

    bool is_const_zero(const expr &v) {
        if (v.isa<constant_c>()) {
            auto c = v.static_as<constant_c>();
            return (c->value_.size() == 1) && (c->value_[0].u64 == 0);
        }
        return false;
    };

private:
    const sc::runtime::target_machine_t &target_machine_;
};

func_c x86_intrinsics_lowering_t::operator()(func_c v) {
    x86_intrinsics_lowering_impl_t x86_intrinsics_lowering(target_machine_);
    return x86_intrinsics_lowering.dispatch(std::move(v));
}

} // namespace sc_xbyak
} // namespace sc