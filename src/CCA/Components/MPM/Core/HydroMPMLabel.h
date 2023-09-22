/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef UINTAH_HOMEBREW_HydroMPMLABEL_H
#define UINTAH_HOMEBREW_HydroMPMLABEL_H

#include <vector>

namespace Uintah {

  class VarLabel;

    class HydroMPMLabel {
    public:

      HydroMPMLabel();
      ~HydroMPMLabel();

     // Hydro-mechanical coupling
      const VarLabel* ccPorosity;
      const VarLabel* ccPorePressure;
      const VarLabel* ccPorePressureOld;
      const VarLabel* ccRHS_FlowEquation;
      const VarLabel* ccTransmissivityMatrix;
      const VarLabel* pFluidMassLabel;
      const VarLabel* pFluidVelocityLabel;
      const VarLabel* pFluidAccelerationLabel;
      const VarLabel* pSolidMassLabel;
      const VarLabel* pPorosityLabel;
      const VarLabel* pPorosityLabel_preReloc;
      const VarLabel* pPrescribedPorePressureLabel;
      const VarLabel* pPorePressureLabel;
      const VarLabel* pPorePressureFilterLabel;

      const VarLabel* pStressRateLabel;
      const VarLabel* pStressRateLabel_preReloc;
      const VarLabel* gFluidMassBarLabel;
      const VarLabel* gFluidMassLabel;
      const VarLabel* gFluidVelocityLabel;
      const VarLabel* FluidVelInc;
      const VarLabel* gFluidVelocityStarLabel;
      const VarLabel* gFluidAccelerationLabel;
      const VarLabel* gInternalFluidForceLabel;
      const VarLabel* gExternalFluidForceLabel;
      const VarLabel* gInternalDragForceLabel;
      const VarLabel* gFlowInertiaForceLabel;
      const VarLabel* gPorePressureLabel;
      const VarLabel* gPorePressureFilterLabel;

      const VarLabel* pFluidMassLabel_preReloc;
      const VarLabel* pFluidVelocityLabel_preReloc;
      const VarLabel* pFluidAccelerationLabel_preReloc;
      const VarLabel* pSolidMassLabel_preReloc;
      const VarLabel* pPorePressureLabel_preReloc;
      const VarLabel* pPorePressureFilterLabel_preReloc;
      const VarLabel* gFluidMassBarLabel_preReloc;
      const VarLabel* gFluidMassLabel_preReloc;
      const VarLabel* gFluidVelocityLabel_preReloc;
      const VarLabel* gFluidVelocityStarLabel_preReloc;
      const VarLabel* gFluidAccelerationLabel_preReloc;

      // MPM Hydrostatic BC label
      const VarLabel* boundaryPointsPerCellLabel;

    };
} // End namespace Uintah

#endif
