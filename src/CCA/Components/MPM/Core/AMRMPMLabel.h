/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_AMRMPMLABEL_H
#define UINTAH_HOMEBREW_AMRMPMLABEL_H

#include <vector>

namespace Uintah {

  class VarLabel;

    class AMRMPMLabel {
    public:

      AMRMPMLabel();
      ~AMRMPMLabel();

      const VarLabel* gZOILabel;
      const VarLabel* MPMRefineCellLabel;
      const VarLabel* pPosChargeLabel;
      const VarLabel* pPosChargeLabel_preReloc;
      const VarLabel* pNegChargeLabel;
      const VarLabel* pNegChargeLabel_preReloc;
      const VarLabel* pPermittivityLabel;
      const VarLabel* pPermittivityLabel_preReloc;
      const VarLabel* pPosChargeGradLabel;
      const VarLabel* pPosChargeGradLabel_preReloc;
      const VarLabel* pNegChargeGradLabel;
      const VarLabel* pNegChargeGradLabel_preReloc;
      const VarLabel* pPosChargeFluxLabel;
      const VarLabel* pPosChargeFluxLabel_preReloc;
      const VarLabel* pNegChargeFluxLabel;
      const VarLabel* pNegChargeFluxLabel_preReloc;
      const VarLabel* pPartitionUnityLabel;

      const VarLabel* gPosChargeLabel;
      const VarLabel* gPosChargeStarLabel;
      const VarLabel* gPosChargeNoBCLabel;
      const VarLabel* gNegChargeLabel;
      const VarLabel* gNegChargeStarLabel;
      const VarLabel* gNegChargeNoBCLabel;
      const VarLabel* gPosChargeRateLabel;
      const VarLabel* gNegChargeRateLabel;
      const VarLabel* pESPotential;
      const VarLabel* pESGradPotential;
    };
} // End namespace Uintah

#endif
