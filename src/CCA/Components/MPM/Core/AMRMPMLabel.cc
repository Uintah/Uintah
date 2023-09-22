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
#include <CCA/Components/MPM/Core/AMRMPMLabel.h>
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

AMRMPMLabel::AMRMPMLabel()
{

  pPosChargeLabel = VarLabel::create( "p.posCharge",
      ParticleVariable<double>::getTypeDescription() );

  pPosChargeLabel_preReloc = VarLabel::create( "p.posCharge+",
      ParticleVariable<double>::getTypeDescription() );

  pNegChargeLabel = VarLabel::create( "p.negCharge",
      ParticleVariable<double>::getTypeDescription() );

  pNegChargeLabel_preReloc = VarLabel::create( "p.negCharge+",
      ParticleVariable<double>::getTypeDescription() );

  pPosChargeGradLabel = VarLabel::create( "p.posChargeGrad",
      ParticleVariable<Vector>::getTypeDescription() );

  pPosChargeGradLabel_preReloc =VarLabel::create("p.posChargeGrad+",
      ParticleVariable<Vector>::getTypeDescription() );

  pNegChargeGradLabel = VarLabel::create( "p.negChargeGrad",
          ParticleVariable<Vector>::getTypeDescription() );

  pNegChargeGradLabel_preReloc =VarLabel::create("p.negChargeGrad+",
          ParticleVariable<Vector>::getTypeDescription() );

  pPermittivityLabel = VarLabel::create( "p.permittivity",
      ParticleVariable<double>::getTypeDescription() );

  pPermittivityLabel_preReloc = VarLabel::create( "p.permittivity+",
        ParticleVariable<double>::getTypeDescription() );

  pPosChargeFluxLabel = VarLabel::create("p.posChargeFlux",
        ParticleVariable<Vector>::getTypeDescription() );

  pPosChargeFluxLabel_preReloc = VarLabel::create("p.posChargeFlux+",
        ParticleVariable<Vector>::getTypeDescription() );

  pNegChargeFluxLabel = VarLabel::create("p.negChargeFlux",
        ParticleVariable<Vector>::getTypeDescription() );

  pNegChargeFluxLabel_preReloc = VarLabel::create("p.negChargeFlux+",
        ParticleVariable<Vector>::getTypeDescription() );

  pPartitionUnityLabel   = VarLabel::create( "p.partitionUnity",
                        ParticleVariable<double>::getTypeDescription() );

  gPosChargeLabel = VarLabel::create("g.PosCharge",
      NCVariable<double>::getTypeDescription());
  gPosChargeStarLabel = VarLabel::create("g.PosChargeStar",
        NCVariable<double>::getTypeDescription());
  gNegChargeLabel = VarLabel::create("g.NegCharge",
      NCVariable<double>::getTypeDescription());
  gNegChargeStarLabel = VarLabel::create("g.NegChargeStar",
          NCVariable<double>::getTypeDescription());
  gPosChargeNoBCLabel = VarLabel::create("g.PosChargeNoBC",
      NCVariable<double>::getTypeDescription());
  gNegChargeNoBCLabel = VarLabel::create("g.NegChargeNoBC",
      NCVariable<double>::getTypeDescription());
  gPosChargeRateLabel = VarLabel::create("g.PosChargeRate",
      NCVariable<double>::getTypeDescription());
  gNegChargeRateLabel = VarLabel::create("g.NegChargeRate",
        NCVariable<double>::getTypeDescription());
  pESPotential = VarLabel::create( "p.esPotential",
      ParticleVariable<double>::getTypeDescription());
  pESGradPotential = VarLabel::create( "p.esGradPotential",
                     ParticleVariable<Vector>::getTypeDescription() );


  MPMRefineCellLabel  = VarLabel::create( "MPMRefineCell",
                     CCVariable<double>::getTypeDescription() );

  gZOILabel     = VarLabel::create("g.zoi",
                        NCVariable<Stencil7>::getTypeDescription());
} 

AMRMPMLabel::~AMRMPMLabel()
{
  VarLabel::destroy(pPartitionUnityLabel);
  VarLabel::destroy(pPosChargeLabel);
  VarLabel::destroy(pPosChargeLabel_preReloc);
  VarLabel::destroy(pNegChargeLabel);
  VarLabel::destroy(pNegChargeLabel_preReloc);
  VarLabel::destroy(pPosChargeGradLabel);
  VarLabel::destroy(pPosChargeGradLabel_preReloc);
  VarLabel::destroy(pNegChargeGradLabel);
  VarLabel::destroy(pNegChargeGradLabel_preReloc);
  VarLabel::destroy(pPermittivityLabel);
  VarLabel::destroy(pPermittivityLabel_preReloc);
  VarLabel::destroy(pPosChargeFluxLabel);
  VarLabel::destroy(pPosChargeFluxLabel_preReloc);
  VarLabel::destroy(pNegChargeFluxLabel);
  VarLabel::destroy(pNegChargeFluxLabel_preReloc);
  VarLabel::destroy(gPosChargeLabel);
  VarLabel::destroy(gPosChargeStarLabel);
  VarLabel::destroy(gNegChargeLabel);
  VarLabel::destroy(gNegChargeStarLabel);
  VarLabel::destroy(gPosChargeNoBCLabel);
  VarLabel::destroy(gNegChargeNoBCLabel);
  VarLabel::destroy(gPosChargeRateLabel);
  VarLabel::destroy(gNegChargeRateLabel);
  VarLabel::destroy(pESPotential);
  VarLabel::destroy(pESGradPotential);

  VarLabel::destroy(gZOILabel);
  VarLabel::destroy(MPMRefineCellLabel);
}
