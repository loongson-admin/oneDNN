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

#include <cstdint>
#include <mutex>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/dnnl_traits.hpp"

#include "cpu/loongarch64/cpu_isa_traits.hpp"
#include "cpu/loongarch64/jit_generator.hpp"

#include "cpu/loongarch64/gemm/gemm_info.hpp"

#include "cpu/loongarch64/gemm/f32/common_f32.hpp"
#include "cpu/loongarch64/gemm/f32/jit_lasx_kernel_sgemm_kern.hpp"
#include "cpu/loongarch64/gemm/f32/jit_lasx_gemv_t_f32_kern.hpp"

#include "cpu/loongarch64/gemm/s8x8s32/common_u8.hpp"
#include "cpu/loongarch64/gemm/s8x8s32/jit_lasx_gemm_s8u8s32_kern.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {

static inline int decode_trans(char trans) {
    switch (trans) {
        case 'T':
        case 't': return do_trans;
        case 'P':
        case 'p': return packed;
        default: return no_trans;
    }
}

namespace {
template <typename b_t> // XXX for float and bfloat
void prepare_bo(int32_t &bo_gemm_info, const b_t *bo_orig) {
    UNUSED(bo_orig);
    bo_gemm_info = 0;
}
template <>
void prepare_bo(int32_t &bo_gemm_info, const uint8_t *bo_orig) {
    bo_gemm_info = bo_orig ? *bo_orig : 0;
}
template <>
void prepare_bo(int32_t &bo_gemm_info, const int8_t *bo_orig) {
    int bo_s32 = bo_orig ? *bo_orig : 0;
    bo_s32 += 128;
    bo_gemm_info = bo_s32;
}

} // namespace

template <typename a_t, typename b_t, typename c_t>
gemm_info_t<a_t, b_t, c_t>::gemm_info_t(const char *transA, const char *transB,
        const char *offsetC, const dim_t *m, const dim_t *n, const dim_t *k,
        const float *alpha, const a_t *a, const dim_t *lda, const a_t *oa,
        const b_t *b, const dim_t *ldb, const b_t *ob, const float *beta,
        c_t *c, const dim_t *ldc, const c_t *oc, bool force_nocopy,
        pack_type packing, gemm_pack_storage_t *pack_dst, bool measure_only) {

    this->transa = decode_trans(*transA);
    this->transb = decode_trans(*transB);

    this->m = *m;
    this->n = *n;
    this->k = *k;

    this->a = a;
    this->b = b;
    this->c = c;

    this->lda = lda ? *lda : 0;
    this->ldb = ldb ? *ldb : 0;
    this->ldc = ldc ? *ldc : 0;

    this->ao = 0;
    this->bo = 0;
    this->co = nullptr;

    this->alpha = alpha ? *alpha : 1.0f;
    this->beta = beta ? *beta : 1.0f;

    this->offsetc = offset_type::none;

    this->packing = packing;
    this->pack_dst = pack_dst;
    this->measure_only
            = measure_only && pack_dst && (packing != pack_type::none);

    if (this->transa == packed) {
        dim_t cols;

        this->a_packed.reset(new gemm_pack_storage_t(a));
        if (this->a_packed->get_nocopy(this->transa, this->lda, cols)) {
            this->a = this->a_packed->template matrix<a_t>();
            this->a_packed = nullptr;
        }
    }
    if (this->transb == packed) {
        dim_t rows;

        this->b_packed.reset(new gemm_pack_storage_t(b));
        if (this->b_packed->get_nocopy(this->transb, this->ldb, rows)) {
            this->b = this->b_packed->template matrix<b_t>();
            this->b_packed = nullptr;
        }
    }

    constexpr bool is_int8 = utils::one_of(
            data_traits<a_t>::data_type, data_type::s8, data_type::u8);
    if (is_int8) this->ao = oa ? *oa : a_t(0);
    prepare_bo<b_t>(this->bo, ob);

    if (offsetC != nullptr) {
        char offsetc = *offsetC;
        if (offsetc == 'F' || offsetc == 'f') {
            this->offsetc = offset_type::fixed;
        } else if (offsetc == 'R' || offsetc == 'r') {
            this->offsetc = offset_type::row;
        } else { // offsetc == 'C' || offsetc == 'c'
            this->offsetc = offset_type::column;
        }
        this->co = oc;
    }

    bool is_sgemm = data_traits<a_t>::data_type == data_type::f32;
    bool is_gemv = this->m == 1 || this->n == 1;

    this->force_nocopy = is_sgemm && force_nocopy && mayiuse(lasx);
    this->force_nocopy |= is_sgemm && false;

    if (!this->force_nocopy || is_gemv) { this->jit_init(); }
}

// copyA[trans][sum]
template <typename a_t, typename b_t, typename c_t>
typename gemm_info_t<a_t, b_t, c_t>::copy_a_fptr_t
        gemm_info_t<a_t, b_t, c_t>::copy_a_kern[2][2]
        = {{nullptr}};

// copyB[trans][sum]
template <typename a_t, typename b_t, typename c_t>
typename gemm_info_t<a_t, b_t, c_t>::copy_b_fptr_t
        gemm_info_t<a_t, b_t, c_t>::copy_b_kern[2][2]
        = {{nullptr}};

// kern[beta0][alpha1][col_off][row_off]
template <typename a_t, typename b_t, typename c_t>
typename gemm_info_t<a_t, b_t, c_t>::gemm_fptr_t
        gemm_info_t<a_t, b_t, c_t>::kern[2][2][2][2]
        = {{{{nullptr}}}};


// gemv do not has performance improve in loongarch
//template <typename a_t, typename b_t, typename c_t>
//typename gemm_info_t<a_t, b_t, c_t>::gemv_fptr_t
//        gemm_info_t<a_t, b_t, c_t>::gemv_kern[2]
//        = {nullptr};

/* loongarch do not support gemv_s8s8s32
template <typename a_t, typename b_t, typename c_t>
typename gemm_info_t<a_t, b_t, c_t>::gemv_s8s8s32_fptr_t
        gemm_info_t<a_t, b_t, c_t>::gemv_s8s8s32_kern
        = nullptr;
template <typename a_t, typename b_t, typename c_t>
typename gemm_info_t<a_t, b_t, c_t>::gemv_s8u8s32_fptr_t
        gemm_info_t<a_t, b_t, c_t>::gemv_s8u8s32_kern
        = nullptr;
template <typename a_t, typename b_t, typename c_t>
typename gemm_info_t<a_t, b_t, c_t>::gemv_u8s8s32_fptr_t
        gemm_info_t<a_t, b_t, c_t>::gemv_u8s8s32_kern
        = nullptr; */

template <typename a_t, typename b_t, typename c_t>
void gemm_info_t<a_t, b_t, c_t>::jit_init(void) {

    switch (data_traits<a_t>::data_type) {
        case data_type::s8:
            if (mayiuse(lasx)) {
                this->um = 16;
                this->un = 4;
                this->uk = 1;
                this->bm = 9984;
                this->bn = 384;
                this->bk = 384;

                this->bk_traditional = 256;
                this->blocking_small_k = 48;
                this->bn_small_k = 24;
            }
            break;

        /* loongarch do not support bf16 */
        case data_type::bf16:
            break;

        case data_type::f32:
            if (mayiuse(lasx)) {
                this->um = 24;
                this->un = 4;
                this->uk = 1;
                this->bm = 10000;
                this->bn = 384;
                this->bk = 192;

                this->bk_traditional = 256;
                this->blocking_small_k = 48;
                this->bn_small_k = 24;
            } else if (mayiuse(lsx)) {
                this->um = 8;
                this->un = 4;
                this->uk = 1;
                this->bm = 4096;
                this->bn = 96;
                this->bk = 256;

                this->bk_traditional = 256;
                this->blocking_small_k = 48;
                this->bn_small_k = 24;
            }
            break;
    }

    // Note: um is fixed for a given set of data types and ISA.
    const int um = this->um;

    static std::once_flag initialized;
    dnnl_status_t st = dnnl_success;
    std::call_once(initialized, [&, um] {
        bool is_int8_amx = false;
        bool is_bf16_amx = false;
        bool is_amx = is_int8_amx || is_bf16_amx;

        static jit_generator *copy_a[2][2] = {{nullptr}};
        static jit_generator *copy_b[2][2] = {{nullptr}};

        switch (data_traits<a_t>::data_type) {
            case data_type::s8:
                if (mayiuse(lasx)) {
                    copy_a[no_trans][no_sum] = new jit_lasx_u8_copy_an_kern();
                    copy_a[do_trans][no_sum] = new jit_lasx_u8_copy_at_kern();

                    copy_b[no_trans][no_sum] = new jit_lasx_u8_copy_bn_kern();
                    copy_b[do_trans][no_sum] = new jit_lasx_u8_copy_bt_kern();

                    copy_a[no_trans][do_sum]
                            = new jit_lasx_u8_copy_sum_an_kern();
                    copy_a[do_trans][do_sum]
                            = new jit_lasx_u8_copy_sum_at_kern();

                    copy_b[no_trans][do_sum]
                            = new jit_lasx_u8_copy_sum_bn_kern();
                    copy_b[do_trans][do_sum]
                            = new jit_lasx_u8_copy_sum_bt_kern();
                }
                break;

            /* loongarch do not support bf16 data_type */
            case data_type::bf16:
                break;

            case data_type::f32:
                if (mayiuse(lasx)) {
                    copy_a[no_trans][no_sum] = new jit_lasx_f32_copy_an_kern();
                    copy_a[do_trans][no_sum] = new jit_lasx_f32_copy_at_kern();

                    copy_b[no_trans][no_sum] = new jit_lasx_f32_copy_bn_kern();
                    copy_b[do_trans][no_sum] = new jit_lasx_f32_copy_bt_kern();
                }
                break;

            default: break;
        }

        static jit_generator *kernel[2][2][2][2] = {{{{nullptr}}}};
        switch (data_traits<a_t>::data_type) {
            case data_type::s8:
                if (mayiuse(lasx)) {
                    for (int isBeta0 : {no_beta0, do_beta0})
                        for (int doColSum : {no_sum, do_sum})
                            for (int doRowSum : {no_sum, do_sum}) {
                                kernel[isBeta0][do_alpha1][doColSum][doRowSum]
                                        = new jit_lasx_gemm_s8u8s32_kern(
                                                isBeta0, doColSum, doRowSum,
                                                um);
                            }
                }
                break;
            /* loongarch do not support bf16 datatype */
            case data_type::bf16:
                break;

            case data_type::f32:
                if (mayiuse(lasx)) {
                    for (int isBeta0 : {no_beta0, do_beta0}) {
                        kernel[isBeta0][do_alpha1][no_sum][no_sum]
                                = new jit_lasx_kernel_sgemm_kern(isBeta0);
                    }
                }
                break;

            default: break;
        }

        /* gemv do not has performance improve in loongarch */

        // Set copy kernels function pointer table
        for (int isTrans : {no_trans, do_trans})
            for (int isSum : {no_sum, do_sum}) {
                auto *p_copy_a = copy_a[isTrans][isSum];
                if (p_copy_a != nullptr) {
                    st = p_copy_a->create_kernel();
                    if (st != dnnl_success) return;
                    copy_a_kern[isTrans][isSum]
                            = (copy_a_fptr_t)p_copy_a->jit_ker();
                }
                auto *p_copy_b = copy_b[isTrans][isSum];
                if (p_copy_b != nullptr) {
                    st = p_copy_b->create_kernel();
                    if (st != dnnl_success) return;
                    copy_b_kern[isTrans][isSum]
                            = (copy_b_fptr_t)p_copy_b->jit_ker();
                }
            }

        /* Doesn't need in Loongarch
        // AMX copy kernels don't support row/column sum. Use wrappers for now.
        if (is_int8_amx) {
            copy_a_kern[no_trans][do_sum] = &copy_a_sum_ref<no_trans>;
            copy_a_kern[do_trans][do_sum] = &copy_a_sum_ref<do_trans>;
            copy_b_kern[no_trans][do_sum] = &copy_b_sum_ref<no_trans>;
            copy_b_kern[do_trans][do_sum] = &copy_b_sum_ref<do_trans>;
        } */

        // Set compute kernel function pointer table
        for (int isBeta0 : {no_beta0, do_beta0})
            for (int isAlpha1 : {no_alpha1, do_alpha1})
                for (int doColSum : {no_sum, do_sum})
                    for (int doRowSum : {no_sum, do_sum}) {
                        auto *p_kernel
                                = kernel[isBeta0][isAlpha1][doColSum][doRowSum];
                        if (p_kernel != nullptr) {
                            st = p_kernel->create_kernel();
                            if (st != dnnl_success) return;
                            kern[isBeta0][isAlpha1][doColSum][doRowSum]
                                    = (gemm_fptr_t)p_kernel->jit_ker();
                        }
                    }
        // Override compute kernel table with AMX kernels
        if (is_amx) {
            // AMX compute kernels don't support alpha scaling, row-offset or
            // col-offset.
            for (int isBeta0 : {no_beta0, do_beta0})
                for (int isAlpha1 : {no_alpha1, do_alpha1})
                    for (int doColSum : {no_sum, do_sum})
                        for (int doRowSum : {no_sum, do_sum}) {
                            kern[isBeta0][isAlpha1][doColSum][doRowSum]
                                    = kern[isBeta0][do_alpha1][no_sum][no_sum];
                        }
        }

        /* gemv do not has performance improve in loongarch
        // Set gemv floating point kernels
        if (utils::one_of(data_traits<a_t>::data_type, data_type::f32)) {
            for (int isTrans : {no_trans, do_trans}) {
                auto *p_gemv_kernel = gemv_kernel[isTrans];
                if (p_gemv_kernel != nullptr) {
                    st = p_gemv_kernel->create_kernel();
                    if (st != dnnl_success) return;
                    gemv_kern[isTrans] = (gemv_fptr_t)p_gemv_kernel->jit_ker();
                }
            }
        }

        // Set gemv integer gemm kernels
        if (data_traits<a_t>::data_type == data_type::s8) {
            if (gemv_s8s8s32_kernel != nullptr) {
                auto *kern = gemv_s8s8s32_kernel;
                st = kern->create_kernel();
                if (st != dnnl_success) return;
                gemv_s8s8s32_kern = (gemv_s8s8s32_fptr_t)kern->jit_ker();
            }

            if (gemv_s8u8s32_kernel != nullptr) {
                auto *kern = gemv_s8u8s32_kernel;
                st = kern->create_kernel();
                if (st != dnnl_success) return;
                gemv_s8u8s32_kern = (gemv_s8u8s32_fptr_t)kern->jit_ker();
            }

            if (gemv_u8s8s32_kernel != nullptr) {
                auto *kern = gemv_u8s8s32_kernel;
                st = kern->create_kernel();
                if (st != dnnl_success) return;
                gemv_u8s8s32_kern = (gemv_u8s8s32_fptr_t)kern->jit_ker();
            }
        } */
    });

    if (st != dnnl_success) return;

    //int doSumA = this->bo != 0 ? do_sum : no_sum;
    //int doSumB = this->ao != 0 ? do_sum : no_sum;
    int doSumA = no_sum;
    int doSumB = no_sum;
    if(this->bo != 0){
        doSumA = do_sum;
    }
    if(this->ao != 0){
        doSumB = do_sum;
    }

    //int copy_trans_a = (this->transa == do_trans) ? do_trans : no_trans;
    //int copy_trans_b = (this->transb == do_trans) ? do_trans : no_trans;
    int copy_trans_a = no_trans;
    if(this->transa == do_trans){
        copy_trans_a = do_trans;
    }
    int copy_trans_b = no_trans;
    if(this->transb == do_trans){
        copy_trans_b = do_trans;
    }

    this->copyA = copy_a_kern[copy_trans_a][doSumA];
    this->copyB = copy_b_kern[copy_trans_b][doSumB];

    constexpr bool is_bf16 = data_traits<a_t>::data_type == data_type::bf16;

    bool doAlpha1 = this->alpha != 1.0f && is_bf16 ? no_alpha1 : do_alpha1;

    for (int isBeta0 : {no_beta0, do_beta0})
        for (int doColSum : {no_sum, do_sum})
            for (int doRowSum : {no_sum, do_sum})
                this->kernel[isBeta0][doColSum][doRowSum]
                        = kern[isBeta0][doAlpha1][doColSum][doRowSum];

    /* gemv do not has performance improve in loongarch
    for (int isTrans : {no_trans, do_trans})
        this->gemv_kernel[isTrans] = gemv_kern[isTrans];

    this->gemv_s8s8s32_kernel = nullptr;
    this->gemv_s8u8s32_kernel = nullptr;
    this->gemv_u8s8s32_kernel = nullptr;
    if (data_traits<a_t>::data_type == data_type::s8) {
        this->gemv_s8s8s32_kernel = gemv_s8s8s32_kern;
        this->gemv_s8u8s32_kernel = gemv_s8u8s32_kern;
        this->gemv_u8s8s32_kernel = gemv_u8s8s32_kern;
    } */
}

// Check if copy algorithm kernels were generated on supported ISAs.
template <typename a_t, typename b_t, typename c_t>
bool gemm_info_t<a_t, b_t, c_t>::hasKernels(void) {

    switch (data_traits<a_t>::data_type) {
        case data_type::s8:
            if (mayiuse(lasx)) {
                for (int isBeta0 : {no_beta0, do_beta0})
                    for (int doColSum : {no_sum, do_sum})
                        for (int doRowSum : {no_sum, do_sum})
                            if (!this->kernel[isBeta0][doColSum][doRowSum])
                                return false;

                if (!this->copyA || !this->copyB) return false;
            }
            break;

        /* loongarch do not support bf16 */
        case data_type::bf16:
            break;

        case data_type::f32:
            if (mayiuse(lasx) && !this->force_nocopy) {
                for (int isBeta0 : {no_beta0, do_beta0})
                    if (!this->kernel[isBeta0][no_sum][no_sum]) return false;

                if (!this->copyA || !this->copyB) return false;

                // gemv do not has performance improve in loongarch
                //for (int isTrans : {no_trans, do_trans})
                //    if (!this->gemv_kernel[isTrans]) return false;
            }
            break;
    }

    // All kernels necessary have been found or ISA is not supported.
    return true;
}

// Override default blocking sizes with sizes specified in the gemm_threading_t
//  structure.
template <typename a_t, typename b_t, typename c_t>
void gemm_info_t<a_t, b_t, c_t>::update_blocking(
        const gemm_threading_t &thread_info) {

    if (thread_info.block_m > 0) this->bm = thread_info.block_m;
    if (thread_info.block_n > 0) this->bn = thread_info.block_n;
    if (thread_info.block_k > 0) this->bk = thread_info.block_k;
}

// Instantiate the gemm_info_t templates needed.
template // For gemm_s8u8s32
        struct gemm_info_t<int8_t, uint8_t, int32_t>;

template // For gemm_s8s8s32
        struct gemm_info_t<int8_t, int8_t, int32_t>;

//template // For gemm_bf16bf16f32
//        struct gemm_info_t<bfloat16_t, bfloat16_t, float>;

template // For sgemm.
        struct gemm_info_t<float, float, float>;

} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
