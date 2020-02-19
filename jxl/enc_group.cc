// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "jxl/enc_group.h"

#include <hwy/static_targets.h>
#include <utility>

#include "jxl/ac_strategy.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/common.h"
#include "jxl/dct_util.h"
#include "jxl/enc_params.h"
#include "jxl/image.h"
#include "jxl/quantizer.h"

namespace jxl {

namespace {

#include "jxl/quantizer-inl.h"

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
HWY_ATTR void QuantizeBlockAC(const Quantizer& quantizer,
                              const bool error_diffusion, size_t c,
                              int32_t quant, float qm_multiplier,
                              size_t quant_kind, size_t xsize, size_t ysize,
                              const float* JXL_RESTRICT block_in,
                              ac_qcoeff_t* JXL_RESTRICT block_out) {
  PROFILER_FUNC;
  const float* JXL_RESTRICT qm = quantizer.InvDequantMatrix(quant_kind, c);
  const float qac = quantizer.Scale() * quant;
  // Not SIMD-fied for now.
  float thres[4] = {0.5f, 0.6f, 0.6f, 0.65f};
  if (c != 1) {
    for (int i = 1; i < 4; ++i) {
      thres[i] = 0.75f;
    }
  }

  CoefficientLayout(&ysize, &xsize);

  if (!error_diffusion) {
    HWY_CAPPED(float, kBlockDim) df;
    HWY_CAPPED(uint32_t, kBlockDim) du;
    const auto quant = Set(df, qac * qm_multiplier);
    const auto abs_mask = BitCast(df, Set(du, 0x7FFFFFFFu));

    for (size_t y = 0; y < ysize * kBlockDim; y++) {
      size_t yfix = static_cast<size_t>(y >= ysize * kBlockDim / 2) * 2;
      const size_t off = y * kBlockDim * xsize;
      for (size_t x = 0; x < xsize * kBlockDim; x += df.N) {
        auto thr = Zero(df);
        if (xsize == 1) {
          HWY_ALIGN uint32_t kMask[kBlockDim] = {0,   0,   0,   0,
                                                 ~0u, ~0u, ~0u, ~0u};
          const auto mask = MaskFromVec(BitCast(df, Load(du, kMask + x)));
          thr =
              IfThenElse(mask, Set(df, thres[yfix + 1]), Set(df, thres[yfix]));
        } else {
          // Same for all lanes in the vector.
          thr = Set(
              df,
              thres[yfix + static_cast<size_t>(x >= xsize * kBlockDim / 2)]);
        }

        const auto q = Load(df, qm + off + x) * quant;
        const auto in = Load(df, block_in + off + x);
        const auto val = q * in;
        const auto nzero_mask = (abs_mask & val) >= thr;
        const auto v = IfThenElseZero(nzero_mask, Round(val));
        Store(v, df, block_out + off + x);
      }
    }
    return;
  }

retry:
  int hfNonZeros[4] = {};
  float hfError[4] = {};
  float hfMaxError[4] = {};
  size_t hfMaxErrorIx[4] = {};
  for (size_t y = 0; y < ysize * kBlockDim; y++) {
    for (size_t x = 0; x < xsize * kBlockDim; x++) {
      const size_t pos = y * kBlockDim * xsize + x;
      if (x < xsize && y < ysize) {
        // Ensure block is initialized
        block_out[pos] = 0;
        continue;
      }
      const size_t hfix = (static_cast<size_t>(y >= ysize * kBlockDim / 2) * 2 +
                           static_cast<size_t>(x >= xsize * kBlockDim / 2));
      const float val = block_in[pos] * (qm[pos] * qac * qm_multiplier);
      float v = (std::abs(val) < thres[hfix]) ? 0 : std::rint(val);
      const float error = std::abs(val) - std::abs(v);
      hfError[hfix] += error;
      if (hfMaxError[hfix] < error) {
        hfMaxError[hfix] = error;
        hfMaxErrorIx[hfix] = pos;
      }
      if (v != 0.0f) {
        hfNonZeros[hfix] += std::abs(v);
      }
      block_out[pos] = static_cast<ac_qcoeff_t>(std::rint(v));
    }
  }
  if (c != 1) return;
  // TODO(veluca): include AFV?
  const size_t kPartialBlockKinds =
      (1 << AcStrategy::Type::IDENTITY) | (1 << AcStrategy::Type::DCT2X2) |
      (1 << AcStrategy::Type::DCT4X4) | (1 << AcStrategy::Type::DCT4X8) |
      (1 << AcStrategy::Type::DCT8X4);
  if ((1 << quant_kind) & kPartialBlockKinds) return;
  float hfErrorLimit = 0.1f * (xsize * ysize) * kDCTBlockSize * 0.25f;
  bool goretry = false;
  for (int i = 1; i < 4; ++i) {
    if (hfError[i] >= hfErrorLimit &&
        hfNonZeros[i] <= (xsize + ysize) * 0.25f) {
      if (thres[i] >= 0.4f) {
        thres[i] -= 0.01f;
        goretry = true;
      }
    }
  }
  if (goretry) goto retry;
  for (int i = 1; i < 4; ++i) {
    if (hfError[i] >= hfErrorLimit && hfNonZeros[i] == 0) {
      const size_t pos = hfMaxErrorIx[i];
      if (hfMaxError[i] >= 0.4f) {
        block_out[pos] = block_in[pos] > 0.0f ? 1.0f : -1.0f;
      }
    }
  }
}

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
HWY_ATTR void QuantizeRoundtripYBlockAC(
    const Quantizer& quantizer, const bool error_diffusion, int32_t quant,
    size_t quant_kind, size_t xsize, size_t ysize,
    const float* JXL_RESTRICT biases, const float* JXL_RESTRICT in,
    ac_qcoeff_t* JXL_RESTRICT quantized, float* JXL_RESTRICT out) {
  QuantizeBlockAC(quantizer, error_diffusion, 1, quant, 1.0f, quant_kind, xsize,
                  ysize, in, quantized);

  PROFILER_ZONE("enc quant adjust bias");
  const float* JXL_RESTRICT dequant_matrix =
      quantizer.DequantMatrix(quant_kind, 1);

  HWY_CAPPED(float, kDCTBlockSize) df;
  const auto inv_qac = Set(df, quantizer.inv_quant_ac(quant));
  for (size_t k = 0; k < kDCTBlockSize * xsize * ysize; k += df.N) {
    const auto quant = Load(df, quantized + k);
    const auto adj_quant = AdjustQuantBias(df, 1, quant, biases);
    const auto dequantm = Load(df, dequant_matrix + k);
    Store(adj_quant * dequantm * inv_qac, df, out + k);
  }
}

}  // namespace

HWY_ATTR JXL_NOINLINE void ComputeCoefficients(size_t group_idx,
                                               PassesEncoderState* enc_state,
                                               AuxOut* aux_out) {
  PROFILER_FUNC;
  const Rect block_group_rect = enc_state->shared.BlockGroupRect(group_idx);
  const Rect cmap_rect(
      block_group_rect.x0() / kColorTileDimInBlocks,
      block_group_rect.y0() / kColorTileDimInBlocks,
      DivCeil(block_group_rect.xsize(), kColorTileDimInBlocks),
      DivCeil(block_group_rect.ysize(), kColorTileDimInBlocks));

  const size_t xsize_blocks = block_group_rect.xsize();
  const size_t ysize_blocks = block_group_rect.ysize();

  const ImageI& full_quant_field = enc_state->shared.raw_quant_field;
  const CompressParams& cparams = enc_state->cparams;

  // TODO(user): it would be better to find & apply correlation here, when
  // quantization is chosen.

  {
    // Only use error diffusion in Wombat mode or slower.
    const bool error_diffusion = cparams.speed_tier <= SpeedTier::kWombat;
    constexpr HWY_CAPPED(float, kDCTBlockSize) d;

    ac_qcoeff_t* JXL_RESTRICT coeffs[kMaxNumPasses][3];
    size_t num_passes = enc_state->shared.multiframe->GetNumPasses();
    JXL_DASSERT(num_passes > 0);
    for (size_t i = 0; i < num_passes; i++) {
      for (size_t c = 0; c < 3; c++) {
        coeffs[i][c] = enc_state->coeffs[i].PlaneRow(c, group_idx);
      }
    }

    HWY_ALIGN float roundtrip_y[AcStrategy::kMaxCoeffArea];
    HWY_ALIGN ac_qcoeff_t quantized[3 * AcStrategy::kMaxCoeffArea];

    size_t offset = 0;

    for (size_t by = 0; by < ysize_blocks; ++by) {
      const int32_t* JXL_RESTRICT row_quant_ac =
          block_group_rect.ConstRow(full_quant_field, by);
      size_t ty = by / kColorTileDimInBlocks;
      const uint8_t* JXL_RESTRICT row_cmap[3] = {
          cmap_rect.ConstRow(enc_state->shared.cmap.ytox_map, ty),
          nullptr,
          cmap_rect.ConstRow(enc_state->shared.cmap.ytob_map, ty),
      };
      AcStrategyRow ac_strategy_row =
          enc_state->shared.ac_strategy.ConstRow(block_group_rect, by);
      for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
           tx++) {
        const auto x_factor =
            Set(d, enc_state->shared.cmap.YtoXRatio(row_cmap[0][tx]));
        const auto b_factor =
            Set(d, enc_state->shared.cmap.YtoBRatio(row_cmap[2][tx]));
        for (size_t bx = tx * kColorTileDimInBlocks;
             bx < xsize_blocks && bx < (tx + 1) * kColorTileDimInBlocks; ++bx) {
          const AcStrategy acs = ac_strategy_row[bx];
          if (!acs.IsFirstBlock()) continue;

          size_t xblocks = acs.covered_blocks_x();
          size_t yblocks = acs.covered_blocks_y();
          CoefficientLayout(&yblocks, &xblocks);
          size_t size = kDCTBlockSize * xblocks * yblocks;

          const int32_t quant_ac = row_quant_ac[bx];
          QuantizeRoundtripYBlockAC(
              enc_state->shared.quantizer, error_diffusion, quant_ac,
              acs.RawStrategy(), xblocks, yblocks, kDefaultQuantBias,
              coeffs[0][1] + offset, quantized + size, roundtrip_y);

          // Unapply color correlation
          for (size_t k = 0; k < size; k += d.N) {
            const auto in_x = Load(d, coeffs[0][0] + offset + k);
            const auto in_y = Load(d, roundtrip_y + k);
            const auto in_b = Load(d, coeffs[0][2] + offset + k);
            const auto out_x = in_x - x_factor * in_y;
            const auto out_b = in_b - b_factor * in_y;
            Store(out_x, d, coeffs[0][0] + offset + k);
            Store(out_b, d, coeffs[0][2] + offset + k);
          }

          for (size_t c : {0, 2}) {
            // Quantize
            QuantizeBlockAC(
                enc_state->shared.quantizer, error_diffusion, c, quant_ac,
                c == 0 ? enc_state->x_qm_multiplier : 1.0f, acs.RawStrategy(),
                xblocks, yblocks, coeffs[0][c] + offset, quantized + c * size);
          }
          enc_state->shared.multiframe->SplitACCoefficients(
              quantized, size, acs, bx, by, offset, coeffs);
          offset += size;
        }
      }
    }
  }
}

Status EncodeGroupTokenizedCoefficients(size_t group_idx, size_t pass_idx,
                                        const PassesEncoderState& enc_state,
                                        BitWriter* writer, AuxOut* aux_out) {
  // Select which histogram to use among those of the current pass.
  const size_t cur_histogram = 0;
  const size_t num_histograms = enc_state.shared.num_histograms;
  // num_histograms is 0 only for lossless.
  JXL_ASSERT(num_histograms == 0 || cur_histogram < num_histograms);
  size_t histo_selector_bits =
      num_histograms == 1 ? 0 : CeilLog2Nonzero(num_histograms - 1);

  if (histo_selector_bits != 0) {
    BitWriter::Allotment allotment(writer, histo_selector_bits);
    writer->Write(histo_selector_bits, cur_histogram);
    ReclaimAndCharge(writer, &allotment, kLayerAC, aux_out);
  }
  WriteTokens(enc_state.passes[pass_idx].ac_tokens[group_idx],
              enc_state.passes[pass_idx].codes,
              enc_state.passes[pass_idx].context_map, writer, kLayerACTokens,
              aux_out);

  return true;
}

}  // namespace jxl