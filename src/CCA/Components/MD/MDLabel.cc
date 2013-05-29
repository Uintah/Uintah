/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#include <CCA/Components/MD/MDLabel.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace Uintah;

MDLabel::MDLabel()
{
  ///////////////////////////////////////////////////////////////////////////
  // Particle Variables
  pXLabel = VarLabel::create("p.x", ParticleVariable<Point>::getTypeDescription());
  pXLabel_preReloc = VarLabel::create("p.x+", ParticleVariable<Point>::getTypeDescription(), IntVector(0, 0, 0),
                                      VarLabel::PositionVariable);

  pForceLabel = VarLabel::create("p.force", ParticleVariable<Vector>::getTypeDescription());
  pForceLabel_preReloc = VarLabel::create("p.force+", ParticleVariable<Vector>::getTypeDescription());

  pAccelLabel = VarLabel::create("p.accel", ParticleVariable<Vector>::getTypeDescription());
  pAccelLabel_preReloc = VarLabel::create("p.accel+", ParticleVariable<Vector>::getTypeDescription());

  pVelocityLabel = VarLabel::create("p.velocity", ParticleVariable<Vector>::getTypeDescription());
  pVelocityLabel_preReloc = VarLabel::create("p.velocity+", ParticleVariable<Vector>::getTypeDescription());

  pEnergyLabel = VarLabel::create("p.energy", ParticleVariable<double>::getTypeDescription());
  pEnergyLabel_preReloc = VarLabel::create("p.energy+", ParticleVariable<double>::getTypeDescription());

  pMassLabel = VarLabel::create("p.mass", ParticleVariable<double>::getTypeDescription());
  pMassLabel_preReloc = VarLabel::create("p.mass+", ParticleVariable<double>::getTypeDescription());

  pChargeLabel = VarLabel::create("p.charge", ParticleVariable<double>::getTypeDescription());
  pChargeLabel_preReloc = VarLabel::create("p.charge+", ParticleVariable<double>::getTypeDescription());

  pParticleIDLabel = VarLabel::create("p.particleID", ParticleVariable<long64>::getTypeDescription());
  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+", ParticleVariable<long64>::getTypeDescription());

  ///////////////////////////////////////////////////////////////////////////
  // Grid Variables
  gForceLabel = VarLabel::create("g.force", NCVariable<Vector>::getTypeDescription());

  gAccelLabel = VarLabel::create("g.accel", NCVariable<Vector>::getTypeDescription());

  gVelocityLabel = VarLabel::create("g.velocity", NCVariable<Vector>::getTypeDescription());

  gEnergyLabel = VarLabel::create("g.energy", NCVariable<std::complex<double> >::getTypeDescription());

  gMassLabel = VarLabel::create("g.mass", NCVariable<double>::getTypeDescription());

  gChargeLabel = VarLabel::create("g.charge", NCVariable<double>::getTypeDescription());

  ///////////////////////////////////////////////////////////////////////////
  // Reduction Variables
  vdwEnergyLabel = VarLabel::create("vdwEnergy", sum_vartype::getTypeDescription());

  ///////////////////////////////////////////////////////////////////////////
  // Reduction Variables - Electrostatic
  spmeFourierEnergyLabel = VarLabel::create("spmeFourierEnergy", sum_vartype::getTypeDescription());
  spmeFourierStressLabel = VarLabel::create("spmeFourierStress", matrix_sum::getTypeDescription());
}

MDLabel::~MDLabel()
{
  ///////////////////////////////////////////////////////////////////////////
  // Particle Variables
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pForceLabel);
  VarLabel::destroy(pForceLabel_preReloc);
  VarLabel::destroy(pAccelLabel);
  VarLabel::destroy(pAccelLabel_preReloc);
  VarLabel::destroy(pVelocityLabel);
  VarLabel::destroy(pVelocityLabel_preReloc);
  VarLabel::destroy(pEnergyLabel);
  VarLabel::destroy(pEnergyLabel_preReloc);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pChargeLabel);
  VarLabel::destroy(pChargeLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);

  ///////////////////////////////////////////////////////////////////////////
  // Grid Variables
  VarLabel::destroy(gForceLabel);
  VarLabel::destroy(gAccelLabel);
  VarLabel::destroy(gVelocityLabel);
  VarLabel::destroy(gEnergyLabel);
  VarLabel::destroy(gMassLabel);
  VarLabel::destroy(gChargeLabel);

  ///////////////////////////////////////////////////////////////////////////
  // Reduction Variables
  VarLabel::destroy(vdwEnergyLabel);
  VarLabel::destroy(spmeFourierEnergyLabel);
  VarLabel::destroy(spmeFourierStressLabel);
}
