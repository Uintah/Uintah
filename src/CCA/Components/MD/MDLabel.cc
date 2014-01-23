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
#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <sci_defs/fftw_defs.h>

using namespace Uintah;

MDLabel::MDLabel()
{
  ///////////////////////////////////////////////////////////////////////////
  // Particle Variables
  pXLabel = VarLabel::create("p.x", ParticleVariable<Point>::getTypeDescription(), IntVector(0, 0, 0),
                             VarLabel::PositionVariable);
  pXLabel_preReloc = VarLabel::create("p.x+", ParticleVariable<Point>::getTypeDescription(), IntVector(0, 0, 0),
                                      VarLabel::PositionVariable);

  pNonbondedForceLabel = VarLabel::create("p.nonbonded_force", ParticleVariable<Vector>::getTypeDescription());
  pNonbondedForceLabel_preReloc = VarLabel::create("p.nonbonded_force+", ParticleVariable<Vector>::getTypeDescription());

  pElectrostaticsForceLabel = VarLabel::create("p.electrostatics_force", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsForceLabel_preReloc = VarLabel::create("p.electrostatics_force+", ParticleVariable<Vector>::getTypeDescription());

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
  // Reduction Variables - Nonbonded
  vdwEnergyLabel = VarLabel::create("vdwEnergy", sum_vartype::getTypeDescription());

  ///////////////////////////////////////////////////////////////////////////
  // Sole Variables - Nonbonded
  nonbondedDependencyLabel = VarLabel::create("nonbondedDependency", SoleVariable<double>::getTypeDescription());

  ///////////////////////////////////////////////////////////////////////////
  // Reduction Variables - Electrostatic
  spmeFourierEnergyLabel = VarLabel::create("spmeFourierEnergy", sum_vartype::getTypeDescription());
  spmeFourierStressLabel = VarLabel::create("spmeFourierStress", matrix_sum::getTypeDescription());

#ifdef HAVE_FFTW

  ///////////////////////////////////////////////////////////////////////////
  // Sole Variables - SPME
  forwardTransformPlanLabel = VarLabel::create("forwardTransformPlan", SoleVariable<fftw_plan>::getTypeDescription());
  backwardTransformPlanLabel = VarLabel::create("backwardTransformPlan", SoleVariable<fftw_plan>::getTypeDescription());
  electrostaticsDependencyLabel = VarLabel::create("electrostaticsDependency", SoleVariable<double>::getTypeDescription());
  subSchedulerDependencyLabel = VarLabel::create("subschedulerDependency", CCVariable<int>::getTypeDescription());

#endif

}

MDLabel::~MDLabel()
{
  ///////////////////////////////////////////////////////////////////////////
  // Particle Variables
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pNonbondedForceLabel);
  VarLabel::destroy(pNonbondedForceLabel_preReloc);
  VarLabel::destroy(pElectrostaticsForceLabel);
  VarLabel::destroy(pElectrostaticsForceLabel_preReloc);
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
  // Reduction Variables
  VarLabel::destroy(vdwEnergyLabel);
  VarLabel::destroy(spmeFourierEnergyLabel);
  VarLabel::destroy(spmeFourierStressLabel);

  ///////////////////////////////////////////////////////////////////////////
  // Sole Variables
  VarLabel::destroy(forwardTransformPlanLabel);
  VarLabel::destroy(backwardTransformPlanLabel);
  VarLabel::destroy(electrostaticsDependencyLabel);
  VarLabel::destroy(subSchedulerDependencyLabel);
}
