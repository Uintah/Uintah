/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
//.......1.........2.........3.........4.........5.........6.........7.........8.........9.........A.........B.........C.........D.........E

  //PER-PARTICLE VARIABLES
  //**********************
  //1>Force calculation variables (main simulation body)
  //1.1-> for electrostatic calculation
  //1.1.1-> real space
//  pRealDipoles                            = VarLabel::create("p.realDipole", ParticleVariable<Vector>::getTypeDescription());
  //pRealDipoles_preReloc                   = VarLabel::create("p.realDipole+", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsRealForce                = VarLabel::create("p.electrostaticsRealForce", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsRealForce_preReloc       = VarLabel::create("p.electrostaticsRealForce+", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsRealField                = VarLabel::create("p.electrostaticsRealField", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsRealField_preReloc       = VarLabel::create("p.electrostaticsRealField+", ParticleVariable<Vector>::getTypeDescription());

  //1.1.2-> reciprocal space
//  pReciprocalDipoles                      = VarLabel::create("p.recipDipole", ParticleVariable<Vector>::getTypeDescription());
  //pReciprocalDipoles_preReloc             = VarLabel::create("p.recipDipole+", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsReciprocalForce          = VarLabel::create("p.recipElectrostaticsForce", ParticleVariable<Vector>::getTypeDescription());
  //pElectrostaticsReciprocalForce_preReloc = VarLabel::create("p.recipElectrostaticsForce+", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsReciprocalField          = VarLabel::create("p.recipElectrostaticsField", ParticleVariable<Vector>::getTypeDescription());
  pElectrostaticsReciprocalField_preReloc = VarLabel::create("p.recipElectrostaticsField+", ParticleVariable<Vector>::getTypeDescription());

  //1.1.3-> total dipoles
  pTotalDipoles                           = VarLabel::create("p.totalDipole", ParticleVariable<Vector>::getTypeDescription());
  pTotalDipoles_preReloc                  = VarLabel::create("p.totalDipole+", ParticleVariable<Vector>::getTypeDescription());

  //1.2-> for nonbonded calculation
  pNonbondedForceLabel          = VarLabel::create("p.nonbonded_force", ParticleVariable<Vector>::getTypeDescription());
  //pNonbondedForceLabel_preReloc = VarLabel::create("p.nonbonded_force+", ParticleVariable<Vector>::getTypeDescription());
  //1.3-> for valence calculation
  pValenceForceLabel          = VarLabel::create("p.valence_force", ParticleVariable<Vector>::getTypeDescription());
  //pValenceForceLabel_preReloc = VarLabel::create("p.valence_force+", ParticleVariable<Vector>::getTypeDescription());
  //2>Integrator related variables
  pXLabel                 = VarLabel::create("p.x", ParticleVariable<Point>::getTypeDescription(),
                                             IntVector(0, 0, 0), VarLabel::PositionVariable);
  pXLabel_preReloc        = VarLabel::create("p.x+", ParticleVariable<Point>::getTypeDescription(),
                                             IntVector(0, 0, 0), VarLabel::PositionVariable);
//  pAccelLabel             = VarLabel::create("p.accel", ParticleVariable<Vector>::getTypeDescription());
//  pAccelLabel_preReloc    = VarLabel::create("p.accel+", ParticleVariable<Vector>::getTypeDescription());
  pVelocityLabel          = VarLabel::create("p.velocity", ParticleVariable<Vector>::getTypeDescription());
  pVelocityLabel_preReloc = VarLabel::create("p.velocity+", ParticleVariable<Vector>::getTypeDescription());
  //3>General particle quantities
  pParticleIDLabel          = VarLabel::create("p.particleID", ParticleVariable<long64>::getTypeDescription());
  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+", ParticleVariable<long64>::getTypeDescription());

  // These should be removed
  //pEnergyLabel = VarLabel::create("p.energy", ParticleVariable<double>::getTypeDescription());
  //pEnergyLabel_preReloc = VarLabel::create("p.energy+", ParticleVariable<double>::getTypeDescription());

  //pMassLabel = VarLabel::create("p.mass", ParticleVariable<double>::getTypeDescription());
  //pMassLabel_preReloc = VarLabel::create("p.mass+", ParticleVariable<double>::getTypeDescription());

//  pChargeLabel = VarLabel::create("p.charge", ParticleVariable<double>::getTypeDescription());
//  pChargeLabel_preReloc = VarLabel::create("p.charge+", ParticleVariable<double>::getTypeDescription());

  // PER SYSTEM VARIABLES (reduction variables)
  // ********************
  //1---> for nonbonded calculation
  nonbondedEnergyLabel = VarLabel::create("nonbondedEnergy", sum_vartype::getTypeDescription());
  nonbondedStressLabel = VarLabel::create("nonbondedStress", matrix_sum::getTypeDescription());
  //2---> for electrostatic calculation
  //2.1---> real space
  electrostaticRealEnergyLabel       = VarLabel::create("electrostaticRealEnergy", sum_vartype::getTypeDescription());
  electrostaticRealStressLabel       = VarLabel::create("electrostaticRealStress", matrix_sum::getTypeDescription());
  //2.2--->  reciprocal space
  electrostaticReciprocalEnergyLabel = VarLabel::create("electrostaticReciprocalEnergy", sum_vartype::getTypeDescription());
  electrostaticReciprocalStressLabel = VarLabel::create("electrostaticReciprocalStress", matrix_sum::getTypeDescription());
  //3---> for valence calculation
  //3.1---> by component
  bondEnergyLabel    = VarLabel::create("bondEnergy", sum_vartype::getTypeDescription());
  bondStressLabel    = VarLabel::create("bondStress", matrix_sum::getTypeDescription());
  bendEnergyLabel    = VarLabel::create("bendEnergy", sum_vartype::getTypeDescription());
  bendStressLabel    = VarLabel::create("bendStress", matrix_sum::getTypeDescription());
  torsionEnergyLabel = VarLabel::create("torsionEnergy", sum_vartype::getTypeDescription());
  torsionStressLabel = VarLabel::create("torsionStress", matrix_sum::getTypeDescription());
  oopEnergyLabel     = VarLabel::create("improperTorsionEnergy", sum_vartype::getTypeDescription());
  oopStressLabel     = VarLabel::create("improperTorsionStress", matrix_sum::getTypeDescription());
  //3.2---> total
  valenceEnergyLabel = VarLabel::create("valenceEnergy", sum_vartype::getTypeDescription());
  valenceStressLabel = VarLabel::create("valenceStress", matrix_sum::getTypeDescription());

  ///////////////////////////////////////////////////////////////////////////
  // Sole Variables - Nonbonded
  nonbondedDependencyLabel = VarLabel::create("nonbondedDependency", SoleVariable<double>::getTypeDescription());


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
	// PER PARTICLE VARIABLES
	//   Force calculation Variables
//	VarLabel::destroy(pRealDipoles);
//	VarLabel::destroy(pRealDipoles_preReloc);
	VarLabel::destroy(pElectrostaticsRealForce);
	VarLabel::destroy(pElectrostaticsRealForce_preReloc);
//	VarLabel::destroy(pReciprocalDipoles);
//	VarLabel::destroy(pReciprocalDipoles_preReloc);
	VarLabel::destroy(pElectrostaticsReciprocalForce);
	VarLabel::destroy(pTotalDipoles);
	VarLabel::destroy(pTotalDipoles_preReloc);
	VarLabel::destroy(pElectrostaticsReciprocalForce_preReloc);
	VarLabel::destroy(pNonbondedForceLabel);
	VarLabel::destroy(pNonbondedForceLabel_preReloc);
	VarLabel::destroy(pValenceForceLabel);
	VarLabel::destroy(pValenceForceLabel_preReloc);
	//   Integrator variables
	VarLabel::destroy(pXLabel);
	VarLabel::destroy(pXLabel_preReloc);
//	VarLabel::destroy(pAccelLabel);
//	VarLabel::destroy(pAccelLabel_preReloc);
	VarLabel::destroy(pVelocityLabel);
	VarLabel::destroy(pVelocityLabel_preReloc);
	//   General
	VarLabel::destroy(pParticleIDLabel);
	VarLabel::destroy(pParticleIDLabel_preReloc);
	// PER SYSTEM VARIABLES
    VarLabel::destroy(nonbondedEnergyLabel);
    VarLabel::destroy(nonbondedStressLabel);
    VarLabel::destroy(electrostaticRealEnergyLabel);
    VarLabel::destroy(electrostaticRealStressLabel);
    VarLabel::destroy(electrostaticReciprocalEnergyLabel);
    VarLabel::destroy(electrostaticReciprocalStressLabel);
    VarLabel::destroy(bondEnergyLabel);
    VarLabel::destroy(bondStressLabel);
    VarLabel::destroy(bendEnergyLabel);
    VarLabel::destroy(bendStressLabel);
    VarLabel::destroy(torsionEnergyLabel);
    VarLabel::destroy(torsionStressLabel);
    VarLabel::destroy(oopEnergyLabel);
    VarLabel::destroy(oopStressLabel);
    VarLabel::destroy(valenceEnergyLabel);
    VarLabel::destroy(valenceStressLabel);


// !!!!! TO BE DELETED
//  VarLabel::destroy(pEnergyLabel);
//  VarLabel::destroy(pEnergyLabel_preReloc);
//  VarLabel::destroy(pMassLabel);
//  VarLabel::destroy(pMassLabel_preReloc);
//  VarLabel::destroy(pChargeLabel);
//  VarLabel::destroy(pChargeLabel_preReloc);
// !!!!! TO BE DELETED (end)


  ///////////////////////////////////////////////////////////////////////////
  // Sole Variables
  VarLabel::destroy(forwardTransformPlanLabel);
  VarLabel::destroy(backwardTransformPlanLabel);
  VarLabel::destroy(electrostaticsDependencyLabel);
  VarLabel::destroy(subSchedulerDependencyLabel);
}
