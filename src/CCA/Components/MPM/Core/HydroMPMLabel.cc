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
#include <CCA/Components/MPM/Core/HydroMPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace Uintah;
using namespace std;

HydroMPMLabel::HydroMPMLabel()
{
  // Hydro-mechanical coupling MPM
  ccPorosity =
      VarLabel::create("ccPorosity", CCVariable<double>::getTypeDescription());
  ccPorePressure = VarLabel::create("ccPorePressure",
      CCVariable<double>::getTypeDescription());
  ccPorePressureOld = VarLabel::create(
      "ccPorePressureOld", CCVariable<double>::getTypeDescription());
  ccRHS_FlowEquation = VarLabel::create(
      "ccRHS_FlowEquation", CCVariable<double>::getTypeDescription());
  ccTransmissivityMatrix = VarLabel::create(
      "ccTransmissivityMatrix", CCVariable<Stencil7>::getTypeDescription());

  pFluidMassLabel = VarLabel::create(
      "p.fluidmass", ParticleVariable<double>::getTypeDescription());
  pSolidMassLabel = VarLabel::create(
      "p.solidmass", ParticleVariable<double>::getTypeDescription());
  pPorosityLabel = VarLabel::create(
      "p.porosity", ParticleVariable<double>::getTypeDescription());
  pPorePressureLabel = VarLabel::create(
      "p.porepressure", ParticleVariable<double>::getTypeDescription());
  pPorePressureFilterLabel = VarLabel::create(
      "p.porepressureFilter", ParticleVariable<double>::getTypeDescription());
  pStressRateLabel = VarLabel::create(
      "p.stressrate", ParticleVariable<double>::getTypeDescription());
  pFluidVelocityLabel = VarLabel::create(
      "p.fluidvelocity", ParticleVariable<Vector>::getTypeDescription());
  pFluidAccelerationLabel = VarLabel::create(
      "p.fluidacceleration", ParticleVariable<Vector>::getTypeDescription());

  pPrescribedPorePressureLabel =
      VarLabel::create("p.prescribedporepressure",
          ParticleVariable<Vector>::getTypeDescription());

  gFluidMassLabel =
      VarLabel::create("g.fluidmass", NCVariable<double>::getTypeDescription());
  gFluidMassBarLabel = VarLabel::create(
      "g.fluidmassbar", NCVariable<double>::getTypeDescription());

  gFluidVelocityLabel = VarLabel::create(
      "g.fluidvelocity", NCVariable<Vector>::getTypeDescription());
  FluidVelInc =
      VarLabel::create("fluidvelinc", NCVariable<Vector>::getTypeDescription());

  gFluidVelocityStarLabel = VarLabel::create(
      "g.fluidvelocitystar", NCVariable<Vector>::getTypeDescription());
  gFluidAccelerationLabel = VarLabel::create(
      "g.fluidacceleration", NCVariable<Vector>::getTypeDescription());
  gExternalFluidForceLabel = VarLabel::create(
      "g.externalfluidforce", NCVariable<Vector>::getTypeDescription());
  gInternalFluidForceLabel = VarLabel::create(
      "g.internalfluidforce", NCVariable<Vector>::getTypeDescription());
  gInternalDragForceLabel = VarLabel::create(
      "g.internaldragforce", NCVariable<Vector>::getTypeDescription());
  gFlowInertiaForceLabel = VarLabel::create(
      "g.flowinertiaforce", NCVariable<Vector>::getTypeDescription());

  pFluidMassLabel_preReloc = VarLabel::create(
      "p.fluidmass+", ParticleVariable<double>::getTypeDescription());
  pSolidMassLabel_preReloc = VarLabel::create(
      "p.solidmass+", ParticleVariable<double>::getTypeDescription());
  pPorosityLabel_preReloc = VarLabel::create(
      "p.porosity+", ParticleVariable<double>::getTypeDescription());
  pPorePressureLabel_preReloc = VarLabel::create(
      "p.porepressure+", ParticleVariable<double>::getTypeDescription());
  pPorePressureFilterLabel_preReloc = VarLabel::create(
      "p.porepressureFilter+", ParticleVariable<double>::getTypeDescription());
  pStressRateLabel_preReloc = VarLabel::create(
      "p.stressrate+", ParticleVariable<double>::getTypeDescription());
  pFluidVelocityLabel_preReloc = VarLabel::create(
      "p.fluidvelocity+", ParticleVariable<Vector>::getTypeDescription());
  pFluidAccelerationLabel_preReloc = VarLabel::create(
      "p.fluidacceleration+", ParticleVariable<Vector>::getTypeDescription());

  gFluidMassLabel_preReloc = VarLabel::create(
      "g.fluidmass+", NCVariable<double>::getTypeDescription());
  gFluidMassBarLabel_preReloc = VarLabel::create(
      "g.fluidmassbar+", NCVariable<double>::getTypeDescription());
  gFluidVelocityLabel_preReloc = VarLabel::create(
      "g.fluidvelocity+", NCVariable<Vector>::getTypeDescription());
  gFluidVelocityStarLabel_preReloc = VarLabel::create(
      "g.fluidvelocitystar+", NCVariable<Vector>::getTypeDescription());
  gFluidAccelerationLabel_preReloc = VarLabel::create(
      "g.fluidacceleration+", NCVariable<Vector>::getTypeDescription());
  gPorePressureLabel = VarLabel::create(
      "g.PorePressure", NCVariable<double>::getTypeDescription());
  gPorePressureFilterLabel = VarLabel::create(
      "g.PorePressureFilter", NCVariable<double>::getTypeDescription());

  // MPM Hydrostatic BC label
  boundaryPointsPerCellLabel = VarLabel::create(
      "boundarypointsPerCell", CCVariable<int>::getTypeDescription());
} 

HydroMPMLabel::~HydroMPMLabel()
{
  // Hydro-mechanical coupling MPM
  VarLabel::destroy(ccPorePressure);
  VarLabel::destroy(ccPorosity);
  VarLabel::destroy(ccPorePressureOld);
  VarLabel::destroy(ccRHS_FlowEquation);
  VarLabel::destroy(ccTransmissivityMatrix);
  VarLabel::destroy(pFluidVelocityLabel);
  VarLabel::destroy(pFluidAccelerationLabel);
  VarLabel::destroy(pSolidMassLabel);
  VarLabel::destroy(pFluidMassLabel);
  VarLabel::destroy(pPorosityLabel);
  VarLabel::destroy(pStressRateLabel);
  VarLabel::destroy(pPorosityLabel_preReloc);
  VarLabel::destroy(pStressRateLabel_preReloc);

  VarLabel::destroy(gFluidMassBarLabel);
  VarLabel::destroy(gFluidMassLabel);
  VarLabel::destroy(pPrescribedPorePressureLabel);
  VarLabel::destroy(pPorePressureLabel);
  VarLabel::destroy(pPorePressureLabel_preReloc);
  VarLabel::destroy(pPorePressureFilterLabel);
  VarLabel::destroy(pPorePressureFilterLabel_preReloc);
  VarLabel::destroy(gFluidVelocityLabel);
  VarLabel::destroy(FluidVelInc);
  VarLabel::destroy(gFluidAccelerationLabel);
  VarLabel::destroy(gInternalDragForceLabel);
  VarLabel::destroy(gExternalFluidForceLabel);
  VarLabel::destroy(gInternalFluidForceLabel);
  VarLabel::destroy(gFlowInertiaForceLabel);
  VarLabel::destroy(gPorePressureLabel);
  VarLabel::destroy(gPorePressureFilterLabel);

  VarLabel::destroy(pFluidMassLabel_preReloc);
  VarLabel::destroy(pFluidVelocityLabel_preReloc);
  VarLabel::destroy(pFluidAccelerationLabel_preReloc);
  VarLabel::destroy(pSolidMassLabel_preReloc);
  VarLabel::destroy(gFluidMassBarLabel_preReloc);
  VarLabel::destroy(gFluidMassLabel_preReloc);
  VarLabel::destroy(gFluidVelocityLabel_preReloc);
  VarLabel::destroy(gFluidAccelerationLabel_preReloc);

  VarLabel::destroy(gFluidVelocityStarLabel);
  VarLabel::destroy(gFluidVelocityStarLabel_preReloc);

  // MPM Hydrostatic BC label
  VarLabel::destroy(boundaryPointsPerCellLabel);
}
