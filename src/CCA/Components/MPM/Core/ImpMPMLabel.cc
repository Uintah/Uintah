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
#include <CCA/Components/MPM/Core/ImpMPMLabel.h>
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

ImpMPMLabel::ImpMPMLabel()
{
  pXXLabel  = VarLabel::create( "p.xx",
                        ParticleVariable<Point>::getTypeDescription() );

  gVelocityOldLabel = VarLabel::create("g.VelocityOld",
                                    NCVariable<Vector>::getTypeDescription()); 

  dispNewLabel = VarLabel::create("dispNew",
                                  NCVariable<Vector>::getTypeDescription()); 

  dispIncLabel = VarLabel::create("dispInc",
                                  NCVariable<Vector>::getTypeDescription()); 

  dispIncQNorm0 = VarLabel::create("dispIncQNorm0",
                                   sum_vartype::getTypeDescription());

  dispIncNormMax = VarLabel::create("dispIncNormMax",
                                    sum_vartype::getTypeDescription());

  dispIncQNorm = VarLabel::create("dispIncQNorm",
                                  sum_vartype::getTypeDescription());

  dispIncNorm = VarLabel::create("dispIncNorm",
                                 sum_vartype::getTypeDescription());
  
  pAccelerationLabel = VarLabel::create("p.acceleration",
                               ParticleVariable<Vector>::getTypeDescription()); 

  pAccelerationLabel_preReloc = VarLabel::create("p.acceleration+",
                               ParticleVariable<Vector>::getTypeDescription()); 

  pExternalHeatFluxLabel = VarLabel::create( "p.externalHeatFlux",
                        ParticleVariable<double>::getTypeDescription() );

  pExternalHeatFluxLabel_preReloc = VarLabel::create( "p.externalHeatFlux+",
                        ParticleVariable<double>::getTypeDescription() );

  gContactLabel       = VarLabel::create( "g.contact",
                        NCVariable<int>::getTypeDescription() );
} 

ImpMPMLabel::~ImpMPMLabel()
{
  VarLabel::destroy(pXXLabel);
  VarLabel::destroy(gVelocityOldLabel);
  VarLabel::destroy(dispNewLabel);
  VarLabel::destroy(dispIncLabel);
  VarLabel::destroy(dispIncQNorm0);
  VarLabel::destroy(dispIncNormMax);
  VarLabel::destroy(dispIncQNorm);
  VarLabel::destroy(dispIncNorm);
  VarLabel::destroy(pAccelerationLabel);
  VarLabel::destroy(pAccelerationLabel_preReloc);
  VarLabel::destroy(pExternalHeatFluxLabel);
  VarLabel::destroy(pExternalHeatFluxLabel_preReloc);
  VarLabel::destroy(gContactLabel);
}
