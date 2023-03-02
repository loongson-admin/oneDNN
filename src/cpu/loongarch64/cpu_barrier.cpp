/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2022-2023 Loongson
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

#include "cpu/loongarch64/cpu_barrier.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

namespace simple_barrier {

void generate(jit_generator &code, Xbyak_loongarch64::XReg reg_ctx,
        Xbyak_loongarch64::XReg reg_nthr, bool usedAsFunc) {
#define BAR_CTR_OFF offsetof(ctx_t, ctr)
#define BAR_SENSE_OFF offsetof(ctx_t, sense)
    using namespace Xbyak_loongarch64;
    const XReg x_tmp_0 = (usedAsFunc) ? code.t0 : code.X_TMP_0;
    const XReg x_tmp_1 = (usedAsFunc) ? code.t1 : code.X_TMP_1;
    const XReg x_addr_sense = (usedAsFunc) ? code.t2 : code.X_TMP_2;
    const XReg x_addr_ctx = (usedAsFunc) ? code.t3 : code.X_TMP_3;
    const XReg x_sense = (usedAsFunc) ? code.t4 : code.X_TMP_4;
    const XReg x_tmp_addr = (usedAsFunc) ? code.t5 : code.X_DEFAULT_ADDR;

    Label barrier_exit_label, spin_label, atomic_label;

    code.mov_imm(x_tmp_0, 1);
    code.beq(reg_nthr, x_tmp_0, barrier_exit_label);

    /* take and save current sense */
    code.add_imm(x_addr_sense, reg_ctx, BAR_SENSE_OFF, x_tmp_0);
    code.ld_d(x_sense, x_addr_sense, 0);

    code.add_imm(x_addr_ctx, reg_ctx, BAR_CTR_OFF, x_tmp_addr);

    if (mayiuse_atomic()) {
        code.mov_imm(x_tmp_1, 1);
        code.amadd_d(x_tmp_0, x_tmp_1, x_addr_ctx);
        code.addi_d(x_tmp_0, x_tmp_0, 1);
    } else {
        code.L(atomic_label);
        code.ll_d(x_tmp_0, x_addr_ctx, 0);
        code.addi_d(x_tmp_0, x_tmp_0, 1);
        code.sc_d(x_tmp_0, x_addr_ctx, 0);
        code.beqz(x_tmp_0, atomic_label);
    }
    code.bne(x_tmp_0, reg_nthr, spin_label);

    /* the last thread {{{ */
    code.mov_imm(x_tmp_0, 0);
    code.st_d(x_tmp_0, x_addr_ctx, 0); // reset ctx
    /* commit CTX clear, before modify SENSE,
       otherwise other threads load old SENSE value. */
    code.dbar(0);

    // notify waiting threads
    code.nor(x_sense, x_sense, x_sense);
    code.st_d(x_sense, x_addr_sense, 0);
    code.b(barrier_exit_label);
    /* }}} the last thread */

    code.L(spin_label);
    //code.yield();
    code.ld_d(x_tmp_0, x_addr_sense, 0);
    code.beq(x_tmp_0, x_sense, spin_label);

    code.dbar(0);
    code.L(barrier_exit_label);

#undef BAR_CTR_OFF
#undef BAR_SENSE_OFF
}

/** jit barrier generator */
struct jit_t : public jit_generator {

    void generate() override {
        simple_barrier::generate(*this, abi_param1, abi_param2, true);
        jirl(zero, ra, 0);
    }

    // TODO: Need to check status
    jit_t() { create_kernel(); }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_t)
};

void barrier(ctx_t *ctx, int nthr) {
    static jit_t j;
    j(ctx, nthr);
}

} // namespace simple_barrier

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
