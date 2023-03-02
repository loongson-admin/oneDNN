/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <cassert>

#include "cpu/loongarch64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace loongarch64 {
namespace io {

io_conf_t::io_conf_t(const bool nt_stores_enabled)
    : nt_stores_enabled_(nt_stores_enabled) {}

io_tail_conf_t::io_tail_conf_t(const std::size_t simd_w,
        const std::size_t tail_size, const Xbyak_loongarch64::XReg &tail_opmask,
        const uint32_t tail_vmm_mask_idx, const Xbyak_loongarch64::XReg &reg_tmp)
    : simd_w_(simd_w)
    , tail_size_(tail_size)
    , tail_opmask_(tail_opmask)
    , tail_vmm_mask_idx_(tail_vmm_mask_idx)
    , reg_tmp_(reg_tmp) {}


io_saturation_conf_t::io_saturation_conf_t(const uint32_t vreg_zero_saturation_idx,
        const uint32_t vreg_saturation_ubound_idx, const Xbyak_loongarch64::XReg &reg_tmp)
    : vreg_zero_saturation_idx_(vreg_zero_saturation_idx)
    , vreg_saturation_ubound_idx_(vreg_saturation_ubound_idx)
    , reg_tmp_(reg_tmp) {}

io_gather_conf_t::io_gather_conf_t(const std::size_t simd_w,
        const Xbyak_loongarch64::XReg &full_opmask, const uint32_t full_vmm_mask_idx,
        const Xbyak_loongarch64::XReg &reg_tmp, const Xbyak_loongarch64::XReg &reg_tmp1,
        const utils::optional_t<uint32_t> &vmm_tmp_idx)
    : simd_w_(simd_w)
    , full_opmask_(full_opmask)
    , full_vmm_mask_idx_(full_vmm_mask_idx)
    , reg_tmp_(reg_tmp)
    , reg_tmp1_(reg_tmp1)
    , vmm_tmp_idx_(vmm_tmp_idx) {}

template <typename Vmm>
jit_io_helper_t<Vmm>::jit_io_helper_t(jit_generator *host, const cpu_isa_t &isa,
        const data_type_t &data_type, const io_conf_t &io_conf,
        const utils::optional_t<io_tail_conf_t> &tail_conf,
        const utils::optional_t<io_saturation_conf_t> &saturation_conf,
        const utils::optional_t<io_gather_conf_t> &gather_conf)
    : host_(host)
    , isa_(isa)
    , data_type_(data_type)
    , bf16_supported_(false)
    //, bf16_emu_(nullptr)
    , io_conf_(io_conf)
    , tail_conf_(tail_conf)
    //, bf16_conf_(bf16_conf)
    , saturation_conf_(saturation_conf)
    , gather_conf_(gather_conf) {

    //assert(utils::one_of(data_type_, data_type::bf16, data_type::f32,
    //               data_type::s8, data_type::u8, data_type::s32)
    //        && "Supported data types bf16, f32, s8, u8, s32");
    assert(utils::one_of(data_type_, data_type::f32,
                   data_type::s8, data_type::u8, data_type::s32)
            && "Supported data types f32, s8, u8, s32");


    static constexpr bool is_xmm = std::is_same<Vmm, Xbyak_loongarch64::VReg>::value;
    const bool is_avx_u8s8 = (isa_ == lasx
            && utils::one_of(data_type_, data_type::s8, data_type::u8));
    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_avx_u8s8);

    static constexpr bool is_zmm = false;
    MAYBE_UNUSED(is_zmm);
}

template <typename Vmm>
jit_io_helper_t<Vmm>::~jit_io_helper_t() = default;

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_bf16() {
    //if (bf16_emu_) {
    //    assert(bf16_conf_.has_value()
    //            && "Config for bf16 emulation is not set.");
    //}
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_opmask(
        const std::size_t how_many_bits_to_set, const Xbyak_loongarch64::XReg &reg_tmp,
        const Xbyak_loongarch64::XReg &mask) {
    const int mask_f32 = (1 << how_many_bits_to_set) - 1;
    host_->mov_imm(mask, mask_f32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_vmm_mask(
        const std::size_t how_many_bits_to_set, const std::size_t simd_w,
        const Xbyak_loongarch64::XReg &reg_tmp, const Vmm &mask) {
    static const uint32_t mask_f32[14]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

    if (how_many_bits_to_set < simd_w) {
        host_->mov_imm(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - how_many_bits_to_set]));
        host_->uni_xvld(mask, reg_tmp, 0);
    } else if (how_many_bits_to_set == simd_w) {
        host_->uni_xvseq_d(mask, mask, mask);
    } else {
        assert(!"Can't set so many bits.");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_i8_data_to_store(const Vmm &i8_vmm) {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    host_->uni_xvpickev_h(i8_vmm, i8_vmm, i8_vmm);
    if (isa_ == lasx) {
        // dst[63:0] = src[63:0]
        // dst[127:64] = src[191:128]
        // dst[191:128] = src[127:64]
        // dst[255:192] = src[127:64]
        const auto src_ymm = Xbyak_loongarch64::XVReg(i8_vmm.getIdx());
        host_->xvpermi_d(src_ymm, src_ymm, 0x58);
    }

    if (data_type_ == data_type::s8)
        host_->uni_xvpickev_b(i8_vmm, i8_vmm, i8_vmm);
    else
        host_->uni_xvpickev_b(i8_vmm, i8_vmm, i8_vmm);
}

template <>
void jit_io_helper_t<Xbyak_loongarch64::XVReg>::emu_gather(const Xbyak_loongarch64::XReg &src_reg,
        const Xbyak_loongarch64::XVReg &indices_vmm, const Xbyak_loongarch64::XVReg &dst_vmm,
        const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(gather_conf_->vmm_tmp_idx_.has_value()
            && "Temporary vreg is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Xbyak_loongarch64::VReg xmm_tmp = Xbyak_loongarch64::VReg(gather_conf_->full_vmm_mask_idx_);
    const Xbyak_loongarch64::VReg xmm_dst = Xbyak_loongarch64::VReg(*gather_conf_->vmm_tmp_idx_);

    host_->mov_imm(gather_conf_->reg_tmp_, 0);
    host_->add_d(gather_conf_->reg_tmp1_, src_reg, host_->zero);

    constexpr int xmm_size_elem = 4;

    const int number_of_xmms = tail
            ? utils::div_up(tail_conf_->tail_size_, xmm_size_elem)
            : utils::div_up(gather_conf_->simd_w_, xmm_size_elem);
    for (int i = 0; i < number_of_xmms; i++) {
        host_->xvpermi_q(Xbyak_loongarch64::XVReg(xmm_tmp.getIdx()), indices_vmm, 0x30 + i);

        const int number_of_values_to_load = i == number_of_xmms - 1 && tail
                        && tail_conf_->tail_size_ % xmm_size_elem != 0
                ? tail_conf_->tail_size_ % xmm_size_elem
                : xmm_size_elem;
        for (int j = 0; j < number_of_values_to_load; j++) {
            host_->vpickve2gr_w(gather_conf_->reg_tmp_, xmm_tmp, j);
            host_->add_d(src_reg, src_reg, gather_conf_->reg_tmp_);
            switch (data_type_) {
                case data_type::f32:
                case data_type::s32: {
                    host_->ld_w(host_->X_TMP_1, src_reg, 0);
                    host_->uni_xvinsgr2vr_w(xmm_dst, host_->X_TMP_1, j);
                    break;
                }
                case data_type::s8:
                case data_type::u8: {
                    host_->ld_b(host_->X_TMP_1, src_reg, 0);
                    host_->vinsgr2vr_b(xmm_dst, host_->X_TMP_1, i * xmm_size_elem + j);
                    break;
                }
                default: assert(!"Unsupported data type.");
            }
            host_->add_d(src_reg, gather_conf_->reg_tmp1_, host_->zero);
        }

        if (data_type_ == data_type::f32 || data_type_ == data_type::s32) {
            host_->xvpermi_q(dst_vmm, Xbyak_loongarch64::XVReg(xmm_dst.getIdx()), i == 0 ? 0x30 : 0x02);
        }
    }

    if (data_type_ == data_type::s32)
        convert_to_f32(dst_vmm, Xbyak_loongarch64::VReg(dst_vmm.getIdx()), data_type_);
    else if (data_type_ == data_type::s8 || data_type_ == data_type::u8)
        convert_to_f32(dst_vmm, xmm_dst, data_type_);
}

template <>
void jit_io_helper_t<Xbyak_loongarch64::VReg>::emu_gather(const Xbyak_loongarch64::XReg &src_reg,
        const Xbyak_loongarch64::VReg &indices_vmm, const Xbyak_loongarch64::VReg &dst_vmm,
        const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    host_->mov_imm(gather_conf_->reg_tmp_, 0);
    host_->add_d(gather_conf_->reg_tmp1_, src_reg, host_->zero);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_values_to_load
            = tail ? tail_conf_->tail_size_ : xmm_size_elem;
    for (unsigned j = 0; j < number_of_values_to_load; j++) {
        host_->vpickve2gr_w(gather_conf_->reg_tmp_, indices_vmm, j);
        host_->add_d(src_reg, src_reg, gather_conf_->reg_tmp_);
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: {
                host_->ld_w(host_->X_TMP_1, src_reg, 0);
                host_->uni_xvinsgr2vr_w(dst_vmm, host_->X_TMP_1, j);
                break;
            }
            case data_type::s8:
            case data_type::u8: {
                host_->ld_b(host_->X_TMP_1, src_reg, 0);
                host_->vinsgr2vr_b(dst_vmm, host_->X_TMP_1, j);
                break;
            }
            default: assert(!"Unsupported data type.");
        }
        host_->add_d(src_reg, gather_conf_->reg_tmp1_, host_->zero);
    }

    if (data_type_ != data_type::f32)
        convert_to_f32(dst_vmm, dst_vmm, data_type_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_tail_mask() {
    assert(tail_conf_.has_value() && "Config for tail processing is not set.");

    if (!tail_conf_->tail_size_) return;

    if (isa_ == lasx)
        prepare_vmm_mask(tail_conf_->tail_size_, tail_conf_->simd_w_,
                tail_conf_->reg_tmp_, Vmm(tail_conf_->tail_vmm_mask_idx_));
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (data_type_ == data_type::bf16 || data_type_ == data_type::s8
            || data_type_ == data_type::u8)
        return;

    if (isa_ == lasx)
        prepare_vmm_mask(gather_conf_->simd_w_, gather_conf_->simd_w_,
                gather_conf_->reg_tmp_, Vmm(gather_conf_->full_vmm_mask_idx_));
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (isa_ == lasx) {
        const Vmm vmm_mask = Vmm(gather_conf_->full_vmm_mask_idx_);
        host_->uni_vpxor(vmm_mask, vmm_mask, vmm_mask);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_saturate_f32() const {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    if (utils::one_of(data_type_, data_type::u8, data_type::s8, data_type::s32))
        host_->init_saturate_f32(
                Vmm(saturation_conf_->vreg_zero_saturation_idx_),
                Vmm(saturation_conf_->vreg_saturation_ubound_idx_),
                saturation_conf_->reg_tmp_, data_type::f32, data_type_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::gather(const Xbyak_loongarch64::XReg &src_reg,
        const Vmm &indices_vmm, const Vmm &dst_vmm, const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    emu_gather(src_reg, indices_vmm, dst_vmm, tail);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load(const Xbyak_loongarch64::XReg &src_addr, const int32_t offset,
        const Vmm &dst_raw_vmm, const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const bool is_avx512 = false;
    const auto dst_vmm = dst_raw_vmm;

    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);
    const bool is_tail_load_for_i8_supported = is_avx512;
    const bool can_load_byte_by_byte = tail
            && (isa_ == lasx || (!is_tail_load_for_i8_supported && is_i8));

    if (can_load_byte_by_byte) {
        load_byte_by_byte(src_addr, offset, dst_vmm,
                tail_conf_->tail_size_ * types::data_type_size(data_type_));
    } else {
        switch (data_type_) {
            case data_type::f32: load_f32(src_addr, offset, dst_vmm, tail); break;
            case data_type::s32: load_s32(src_addr, offset, dst_vmm, tail); break;
            case data_type::s8:
            case data_type::u8: load_i8(src_addr, offset, dst_vmm); break;
            default: assert(!"Unsupported data type.");
        }
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_byte_by_byte(const Xbyak_loongarch64::XReg &src_addr,
        const int32_t offset, const Vmm &dst_vmm, const int load_size) {
    host_->uni_vpxor(dst_vmm, dst_vmm, dst_vmm);
    if (data_type_ == data_type::f32 || data_type_ == data_type::s32)
        host_->load_bytes(dst_vmm, src_addr, offset, load_size);
    else if (data_type_ == data_type::s8 || data_type_ == data_type::u8)
        host_->load_bytes_to_dword_extension(dst_vmm, src_addr, offset, data_type_ == data_type::s8, load_size);
    else
        assert(!"unsupported source data type");

    if (utils::one_of(data_type_, data_type::s32, data_type::s8, data_type::u8))
        convert_to_f32(dst_vmm, Xbyak_loongarch64::VReg(dst_vmm.getIdx()), data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_f32(const Xbyak_loongarch64::XReg &src_addr,
        const int32_t offset, const Vmm &dst_vmm, const bool tail) {
    if (tail) {
        host_->uni_xvld(dst_vmm, src_addr, offset);
        host_->uni_xvand_v(dst_vmm, dst_vmm, Vmm(tail_conf_->tail_vmm_mask_idx_));
    }
    else
        host_->uni_xvld(dst_vmm, src_addr, offset);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_s32(const Xbyak_loongarch64::XReg &src_addr,
        const int32_t offset, const Vmm &dst_vmm, const bool tail) {
    load_f32(src_addr, offset, dst_vmm, tail);
    convert_to_f32(dst_vmm, Xbyak_loongarch64::VReg(dst_vmm.getIdx()), data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_i8(const Xbyak_loongarch64::XReg &src_addr,
        const int32_t offset, const Vmm &dst_vmm) {
    host_->uni_xvldrepl_d(dst_vmm, src_addr, offset);
    if (data_type_ == data_type::s8)
        host_->vext2xv_w_b(Xbyak_loongarch64::XVReg(dst_vmm.getIdx()), Xbyak_loongarch64::XVReg(dst_vmm.getIdx()));
    else
        host_->vext2xv_wu_bu(Xbyak_loongarch64::XVReg(dst_vmm.getIdx()), Xbyak_loongarch64::XVReg(dst_vmm.getIdx()));

    convert_to_f32(dst_vmm, Xbyak_loongarch64::VReg(dst_vmm.getIdx()), data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store(const Vmm &src_raw_vmm, const Xbyak_loongarch64::XReg &dst_raw_addr,
        const int32_t offset, const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");
    assert(!(tail && io_conf_.nt_stores_enabled_)
            && "Usage of non-temporal stores with tail leads to a general-protection exception.");

    const bool is_avx512 = false;
    const auto dst_addr = dst_raw_addr;
    const auto src_vmm = src_raw_vmm;

    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);
    const bool is_store_tail_for_i8_supported = is_avx512;
    const bool can_store_byte_by_byte = tail
            && (isa_ == lasx || (!is_store_tail_for_i8_supported && is_i8));

    if (data_type_ == data_type::s32 || is_i8) saturate(src_raw_vmm);

    if (can_store_byte_by_byte) {
        const size_t store_size
                = tail_conf_->tail_size_ * types::data_type_size(data_type_);
        store_byte_by_byte(src_vmm, dst_addr, offset, store_size);
    } else {
        switch (data_type_) {
            case data_type::f32:
            case data_type::s32: store_f32(src_vmm, dst_addr, offset, tail); break;
            //case data_type::bf16: store_bf16(src_vmm, dst_addr); break;
            case data_type::s8:
            case data_type::u8: store_i8(src_vmm, dst_raw_addr, offset); break;
            default: assert(!"Unsupported data type.");
        }
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::saturate(const Vmm &vmm) {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    host_->saturate_f32(vmm, Vmm(saturation_conf_->vreg_zero_saturation_idx_),
            Vmm(saturation_conf_->vreg_saturation_ubound_idx_), data_type_);
    host_->uni_xvftint_w_s(vmm, vmm);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_byte_by_byte(const Vmm &src_vmm,
        const Xbyak_loongarch64::XReg &dst_addr, const int32_t offset, const int store_size) {

    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);
    if (is_i8) prepare_i8_data_to_store(src_vmm);

    host_->store_bytes(src_vmm, dst_addr, offset, store_size);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_f32(const Vmm &src_vmm,
        const Xbyak_loongarch64::XReg &dst_addr, const int32_t offset, const bool tail) {
    if (io_conf_.nt_stores_enabled_)
        host_->uni_xvst(src_vmm, dst_addr, offset);
    else if (tail)
        host_->store_bytes(src_vmm, dst_addr, offset, tail_conf_->tail_size_ * sizeof(float));
    else
        host_->uni_xvst(src_vmm, dst_addr, offset);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8(const Vmm &src_vmm,
        const Xbyak_loongarch64::XReg &dst_addr, const int32_t offset) {
    if (isa_ == lasx) {
        prepare_i8_data_to_store(src_vmm);
        host_->uni_xvstelm_d(src_vmm, dst_addr, offset, 0);
    } else if (isa_ == lsx) {
        prepare_i8_data_to_store(src_vmm);
        host_->uni_xvstelm_w(src_vmm, dst_addr, offset, 0);
    } else
        assert(!"unsupported isa type!");
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::convert_to_f32(const Vmm &dst_vmm,
        const Xbyak_loongarch64::VReg &src_vmm, const data_type_t src_data_type) {
    switch (src_data_type) {
        case data_type::s32: {
            assert(dst_vmm.getIdx() == src_vmm.getIdx());
            host_->uni_xvffint_s_w(dst_vmm, dst_vmm);
            break;
        }
        case data_type::s8: {
            host_->vext2xv_w_b(Xbyak_loongarch64::XVReg(dst_vmm.getIdx()), Xbyak_loongarch64::XVReg(src_vmm.getIdx()));
            host_->uni_xvffint_s_w(dst_vmm, dst_vmm);
            break;
        }
        case data_type::u8: {
            host_->vext2xv_wu_bu(Xbyak_loongarch64::XVReg(dst_vmm.getIdx()), Xbyak_loongarch64::XVReg(src_vmm.getIdx()));
            host_->uni_xvffint_s_w(dst_vmm, dst_vmm);
            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::broadcast(const Xbyak_loongarch64::XReg &src_addr,
        const int32_t offset, const Vmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32: host_->uni_xvldrepl_w(dst_vmm, src_addr, offset); break;
        case data_type::s32: {
            host_->uni_xvldrepl_w(dst_vmm, src_addr, offset);
            convert_to_f32(dst_vmm, Xbyak_loongarch64::VReg(dst_vmm.getIdx()), data_type_);
            break;
        }
        case data_type::s8:
        case data_type::u8: {
            host_->uni_ld_b(host_->X_TMP_0, src_addr, offset);
            host_->vinsgr2vr_b(Xbyak_loongarch64::VReg(dst_vmm.getIdx()), host_->X_TMP_0, 0);
            convert_to_f32(dst_vmm, Xbyak_loongarch64::VReg(dst_vmm.getIdx()), data_type_);
            host_->uni_xvreplve0_w(dst_vmm, dst_vmm);
            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::jit_io_multi_dt_helper_t(jit_generator *host,
        const cpu_isa_t &isa, const data_types_t &data_types,
        const io_conf_t &io_conf,
        const utils::optional_t<io_tail_conf_t> &tail_conf,
        //const utils::optional_t<io_emu_bf16_conf_t> &bf16_conf,
        const std::map<data_type_t, io_saturation_conf_t> &saturation_confs,
        const utils::optional_t<io_gather_conf_t> &gather_conf) {
    assert(!data_types.empty());
    for (const auto &dt : data_types) {
        // can be replaced by try_emplace from C++17
        if (storage_.find(dt) == storage_.cend()) {

            const auto saturation_conf = saturation_confs.find(dt);
            const bool store_saturation_needed
                    = saturation_conf != saturation_confs.cend();

            storage_.emplace(dt,
                    std::make_shared<jit_io_helper_t<Vmm>>(host, isa, dt,
                            io_conf, tail_conf,
                            //dt == data_type::bf16 ? bf16_conf : utils::nullopt,
                            store_saturation_needed ? utils::optional_t<
                                    io_saturation_conf_t> {saturation_conf
                                                                   ->second}
                                                    : utils::nullopt,
                            gather_conf));
        }
    }
}

template <typename Vmm>
std::shared_ptr<jit_io_helper_t<Vmm>> jit_io_multi_dt_helper_t<Vmm>::at(
        const data_type_t dt) const {
    const auto it = storage_.find(dt);
    if (it != storage_.cend()) return it->second;

    return nullptr;
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::prepare_tail_mask() {
    return storage_.cbegin()->second->prepare_tail_mask();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::prepare_full_mask() {
    return storage_.cbegin()->second->prepare_full_mask();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::init_saturate_f32(
        const data_types_t &store_data_types) {
    for (const auto &dt : store_data_types) {
        const auto it = storage_.find(dt);
        if (it != storage_.cend()) {
            if (it->second->saturation_conf_.has_value())
                it->second->init_saturate_f32();
        }
    }
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::init_full_mask() {
    return storage_.cbegin()->second->init_full_mask();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::init_bf16() {
    //const auto bf16_io_helper = at(data_type::bf16);
    //if (bf16_io_helper) bf16_io_helper->init_bf16();
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::~jit_io_multi_dt_helper_t() = default;


template class jit_io_helper_t<Xbyak_loongarch64::XVReg>;
template class jit_io_helper_t<Xbyak_loongarch64::VReg>;

template class jit_io_multi_dt_helper_t<Xbyak_loongarch64::XVReg>;
template class jit_io_multi_dt_helper_t<Xbyak_loongarch64::VReg>;

} // namespace io
} // namespace loongarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
