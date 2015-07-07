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

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <sci_defs/fftw_defs.h>

using namespace Uintah;

globalLabels::globalLabels() { // create variable labels used across all simulation aspects
  // Create for timestep n
  pX            = VarLabel::create("p.x", ParticleVariable<Point>::getTypeDescription(),
                                   IntVector(0,0,0), VarLabel::PositionVariable);
  pV            = VarLabel::create("p.v", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pID           = VarLabel::create("p.ID", ParticleVariable<long64>::getTypeDescription());

  // Create for timestep n+1
  pX_preReloc   = VarLabel::create("p.x+", ParticleVariable<Point>::getTypeDescription(),
                                   IntVector(0,0,0), VarLabel::PositionVariable);
  pV_preReloc   = VarLabel::create("p.v+", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pID_preReloc  = VarLabel::create("p.ID+", ParticleVariable<long64>::getTypeDescription());

  rKineticEnergy= VarLabel::create("e_kin", sum_vartype::getTypeDescription());
  rKineticStress= VarLabel::create("S_kin", matrix_sum::getTypeDescription());
  rTotalMomentum= VarLabel::create("momentum", sumvec_vartype::getTypeDescription());
  rTotalMass    = VarLabel::create("mass", sum_vartype::getTypeDescription());
}

globalLabels::~globalLabels() { // destroy variable labels used across all simulation aspects
  VarLabel::destroy(pX);
  VarLabel::destroy(pV);
  VarLabel::destroy(pID);
  VarLabel::destroy(pX_preReloc);
  VarLabel::destroy(pV_preReloc);
  VarLabel::destroy(pID_preReloc);
  VarLabel::destroy(rKineticEnergy);
  VarLabel::destroy(rKineticStress);
  VarLabel::destroy(rTotalMomentum);
  VarLabel::destroy(rTotalMass);
}

nonbondedLabels::nonbondedLabels() { // create variable labels used in nonbonded calculation contexts
  // Reduction variables
  rNonbondedEnergy      = VarLabel::create("e_nb", sum_vartype::getTypeDescription());
  rNonbondedStress      = VarLabel::create("S_nb", matrix_sum::getTypeDescription());

  // Dependency variables
  dNonbondedDependency  = VarLabel::create("dep_nb", SoleVariable<double>::getTypeDescription());

  // Particle variables --> should not be necessary in the long run?
  pF_nonbonded          = VarLabel::create("pF_nb", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pF_nonbonded_preReloc = VarLabel::create("pF_nb+", ParticleVariable<SCIRun::Vector>::getTypeDescription());

  pNumPairsInCalc           = VarLabel::create("p.nn", ParticleVariable<long64>::getTypeDescription());
  pNumPairsInCalc_preReloc  = VarLabel::create("p.nn+", ParticleVariable<long64>::getTypeDescription());



}

nonbondedLabels::~nonbondedLabels() { // destroy variable labels used in nonbonded calculation context
  VarLabel::destroy(rNonbondedEnergy);
  VarLabel::destroy(rNonbondedStress);
  VarLabel::destroy(dNonbondedDependency);
  VarLabel::destroy(pF_nonbonded);
  VarLabel::destroy(pF_nonbonded_preReloc);
  VarLabel::destroy(pNumPairsInCalc);
  VarLabel::destroy(pNumPairsInCalc_preReloc);
}

electrostaticLabels::electrostaticLabels() { // create variable labels used in electrostatic calculation contexts
  // Reduction variables
  rElectrostaticRealEnergy          = VarLabel::create("e_elec_real", sum_vartype::getTypeDescription());
  rElectrostaticRealStress          = VarLabel::create("S_elec_real", matrix_sum::getTypeDescription());
  rElectrostaticInverseEnergy       = VarLabel::create("e_elec_inv", sum_vartype::getTypeDescription());
  rElectrostaticInverseStress       = VarLabel::create("S_elec_inv", matrix_sum::getTypeDescription());
  rElectrostaticInverseStressDipole = VarLabel::create("S_elec_inv_dip", matrix_sum::getTypeDescription());
  rPolarizationDeviation            = VarLabel::create("pol_squaredVar", sum_vartype::getTypeDescription());
  // Sole variables for interfacing with FFTW
#ifdef HAVE_FFTW
  sForwardTransformPlan         = VarLabel::create("fftw_plan_forward", SoleVariable<fftw_plan>::getTypeDescription());
  sBackwardTransformPlan        = VarLabel::create("fftw_plan_backward", SoleVariable<fftw_plan>::getTypeDescription());
#endif

  // Dependency variables
  dElectrostaticDependency      = VarLabel::create("dep_elec", SoleVariable<double>::getTypeDescription());
  dSubschedulerDependency       = VarLabel::create("dep_elec_subscheduler", CCVariable<int>::getTypeDescription());

  // Particle variables
  // --> Timestep n
  pMu                           = VarLabel::create("pMu", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pMuSub                        = VarLabel::create("pMuSub", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pQ                            = VarLabel::create("pQ", ParticleVariable<double>::getTypeDescription());
  // --> Timestep n+1
  pMu_preReloc                  = VarLabel::create("pMu+", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pMuSub_preReloc               = VarLabel::create("pMuSub+", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pQ_preReloc                   = VarLabel::create("pQ+", ParticleVariable<double>::getTypeDescription());

  //  The remainder of the particle variables should not be necessary in the long run?
  //  --> Timestep n
  pF_electroReal                = VarLabel::create("pF_elec_real", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pF_electroInverse             = VarLabel::create("pF_elec_inv", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pE_electroReal                = VarLabel::create("pE_elec_real", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pE_electroInverse             = VarLabel::create("pE_elec_inv", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  //  --> Timestep n+1
  pF_electroReal_preReloc       = VarLabel::create("pF_elec_real+", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pF_electroInverse_preReloc    = VarLabel::create("pF_elec_inv+", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pE_electroReal_preReloc       = VarLabel::create("pE_elec_real+", ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pE_electroInverse_preReloc    = VarLabel::create("pE_elec_inv+", ParticleVariable<SCIRun::Vector>::getTypeDescription());

}

electrostaticLabels::~electrostaticLabels() { // destroy variable labels used in electrostatic calculation contexts
  VarLabel::destroy(rElectrostaticRealEnergy);
  VarLabel::destroy(rElectrostaticRealStress);
  VarLabel::destroy(rElectrostaticInverseEnergy);
  VarLabel::destroy(rElectrostaticInverseStress);
  VarLabel::destroy(rElectrostaticInverseStressDipole);
  VarLabel::destroy(rPolarizationDeviation);

#ifdef HAVE_FFTW
  VarLabel::destroy(sForwardTransformPlan);
  VarLabel::destroy(sBackwardTransformPlan);
#endif

  VarLabel::destroy(dElectrostaticDependency);
  VarLabel::destroy(dSubschedulerDependency);

  VarLabel::destroy(pMu);
  VarLabel::destroy(pMuSub);
  VarLabel::destroy(pQ);
  VarLabel::destroy(pMu_preReloc);
  VarLabel::destroy(pMuSub_preReloc);
  VarLabel::destroy(pQ_preReloc);

  VarLabel::destroy(pF_electroReal);
  VarLabel::destroy(pF_electroInverse);
  VarLabel::destroy(pE_electroReal);
  VarLabel::destroy(pE_electroInverse);

  VarLabel::destroy(pF_electroReal_preReloc);
  VarLabel::destroy(pF_electroInverse_preReloc);
  VarLabel::destroy(pE_electroReal_preReloc);
  VarLabel::destroy(pE_electroInverse_preReloc);

}

valenceLabels::valenceLabels() { // create variable labels used in valence calculations
  // Reduction variables
  rValenceEnergy        = VarLabel::create("e_valence", sum_vartype::getTypeDescription());
  rValenceStress        = VarLabel::create("S_valence", matrix_sum::getTypeDescription());
  rBondEnergy           = VarLabel::create("e_bond", sum_vartype::getTypeDescription());
  rBondStress           = VarLabel::create("S_bond", matrix_sum::getTypeDescription());
  rBendEnergy           = VarLabel::create("e_bend", sum_vartype::getTypeDescription());
  rBendStress           = VarLabel::create("S_bend", matrix_sum::getTypeDescription());
  rDihedralEnergy       = VarLabel::create("e_dihedral", sum_vartype::getTypeDescription());
  rDihedralStress       = VarLabel::create("S_dihedral", matrix_sum::getTypeDescription());
  rOOPEnergy            = VarLabel::create("e_OOP", sum_vartype::getTypeDescription());
  rOOPStress            = VarLabel::create("S_OOP", matrix_sum::getTypeDescription());

}

valenceLabels::~valenceLabels() { // destroy variable labels used in valence calculations
  VarLabel::destroy(rValenceEnergy);
  VarLabel::destroy(rValenceStress);
  VarLabel::destroy(rBondEnergy);
  VarLabel::destroy(rBondStress);
  VarLabel::destroy(rBendEnergy);
  VarLabel::destroy(rBendStress);
  VarLabel::destroy(rDihedralEnergy);
  VarLabel::destroy(rDihedralStress);
  VarLabel::destroy(rOOPEnergy);
  VarLabel::destroy(rOOPStress);

}

SPME_dependencies::SPME_dependencies() {
  // Create the dependency labels for the SPME subscheduler
  // Note variable types:

  // We have material types here, but don't need them so this will slightly
  // decrease memory access
  dPreTransform             =   VarLabel::create("dep_SPME_preXForm", PerPatch<int>::getTypeDescription());

  // These routines have no material, so must be a PerPatch variable
  dReduceNodeLocalQ         =   VarLabel::create("dep_SPME_redQ", PerPatch<int>::getTypeDescription());
  dCalculateInFourierSpace  =   VarLabel::create("dep_SPME_calcInv", PerPatch<int>::getTypeDescription());
  dDistributeNodeLocalQ     =   VarLabel::create("dep_SPME_distQ", PerPatch<int>::getTypeDescription());

  // Transforms do not even have a patch basis to bind to, and occur once per
  // PROCESSOR, so use SoleVariables
  dInitializeQ              =   VarLabel::create("dep_SPME_initQ", SoleVariable<int>::getTypeDescription());
  dTransformRealToFourier   =   VarLabel::create("dep_SPME_XFormRtoF", SoleVariable<int>::getTypeDescription());
  dTransformFourierToReal   =   VarLabel::create("dep_SPME_XFormFtoR", SoleVariable<int>::getTypeDescription());
}

SPME_dependencies::~SPME_dependencies() {
    VarLabel::destroy(dPreTransform);
    VarLabel::destroy(dReduceNodeLocalQ);
    VarLabel::destroy(dTransformRealToFourier);
    VarLabel::destroy(dCalculateInFourierSpace);
    VarLabel::destroy(dTransformFourierToReal);
    VarLabel::destroy(dDistributeNodeLocalQ);
}

integratorLabels::integratorLabels() {
  fPatchFirstIntegration  =   VarLabel::create("patchFirstIntegration", PerPatch<bool>::getTypeDescription());
}

integratorLabels::~integratorLabels() {
  VarLabel::destroy(fPatchFirstIntegration);
}

MDLabel::MDLabel() {

  global        = scinew globalLabels();
  nonbonded     = scinew nonbondedLabels();
  electrostatic = scinew electrostaticLabels();
  valence       = scinew valenceLabels();
  SPME_dep      = scinew SPME_dependencies();
  integrator    = scinew integratorLabels();
}

MDLabel::~MDLabel() {

  if (global) {
    delete global;
  }

  if (nonbonded) {
    delete nonbonded;
  }

  if (electrostatic) {
    delete electrostatic;
  }

  if (valence) {
    delete valence;
  }
}

//MDLabel::MDLabel()
//{
////.......1.........2.........3.........4.........5.........6.........7.........8.........9.........A.........B.........C.........D.........E
//
//  //PER-PARTICLE VARIABLES
//  //**********************
//  //1>Force calculation variables (main simulation body)
//  //1.1-> for electrostatic calculation
//  //1.1.1-> real space
////  pRealDipoles                            = VarLabel::create("p.realDipole", ParticleVariable<Vector>::getTypeDescription());
//  //pRealDipoles_preReloc                   = VarLabel::create("p.realDipole+", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsRealForce                = VarLabel::create("p.electrostaticsRealForce", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsRealForce_preReloc       = VarLabel::create("p.electrostaticsRealForce+", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsRealField                = VarLabel::create("p.electrostaticsRealField", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsRealField_preReloc       = VarLabel::create("p.electrostaticsRealField+", ParticleVariable<Vector>::getTypeDescription());
//
//  //1.1.2-> reciprocal space
////  pReciprocalDipoles                      = VarLabel::create("p.recipDipole", ParticleVariable<Vector>::getTypeDescription());
//  //pReciprocalDipoles_preReloc             = VarLabel::create("p.recipDipole+", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsReciprocalForce          = VarLabel::create("p.recipElectrostaticsForce", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsReciprocalForce_preReloc = VarLabel::create("p.recipElectrostaticsForce+", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsReciprocalField          = VarLabel::create("p.recipElectrostaticsField", ParticleVariable<Vector>::getTypeDescription());
//  pElectrostaticsReciprocalField_preReloc = VarLabel::create("p.recipElectrostaticsField+", ParticleVariable<Vector>::getTypeDescription());
//
//  //1.1.3-> total dipoles
//  pTotalDipoles                           = VarLabel::create("p.totalDipole", ParticleVariable<Vector>::getTypeDescription());
//  pTotalDipoles_preReloc                  = VarLabel::create("p.totalDipole+", ParticleVariable<Vector>::getTypeDescription());
//
//  //1.2-> for nonbonded calculation
//  pNonbondedForceLabel          = VarLabel::create("p.nonbonded_force", ParticleVariable<Vector>::getTypeDescription());
//  pNonbondedForceLabel_preReloc = VarLabel::create("p.nonbonded_force+", ParticleVariable<Vector>::getTypeDescription());
//
//  //1.3-> for valence calculation
//  pValenceForceLabel          = VarLabel::create("p.valence_force", ParticleVariable<Vector>::getTypeDescription());
//  pValenceForceLabel_preReloc = VarLabel::create("p.valence_force+", ParticleVariable<Vector>::getTypeDescription());
//
//  //2>Integrator related variables
//  pXLabel                 = VarLabel::create("p.x", ParticleVariable<Point>::getTypeDescription(),
//                                             IntVector(0, 0, 0), VarLabel::PositionVariable);
//  pXLabel_preReloc        = VarLabel::create("p.x+", ParticleVariable<Point>::getTypeDescription(),
//                                             IntVector(0, 0, 0), VarLabel::PositionVariable);
////  pAccelLabel             = VarLabel::create("p.accel", ParticleVariable<Vector>::getTypeDescription());
////  pAccelLabel_preReloc    = VarLabel::create("p.accel+", ParticleVariable<Vector>::getTypeDescription());
//
//  pVelocityLabel          = VarLabel::create("p.velocity", ParticleVariable<Vector>::getTypeDescription());
//  pVelocityLabel_preReloc = VarLabel::create("p.velocity+", ParticleVariable<Vector>::getTypeDescription());
//
//  //3>General particle quantities
//  pParticleIDLabel          = VarLabel::create("p.particleID", ParticleVariable<long64>::getTypeDescription());
//  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+", ParticleVariable<long64>::getTypeDescription());
//
//  // These should be removed
//  //pEnergyLabel = VarLabel::create("p.energy", ParticleVariable<double>::getTypeDescription());
//  //pEnergyLabel_preReloc = VarLabel::create("p.energy+", ParticleVariable<double>::getTypeDescription());
//
//  //pMassLabel = VarLabel::create("p.mass", ParticleVariable<double>::getTypeDescription());
//  //pMassLabel_preReloc = VarLabel::create("p.mass+", ParticleVariable<double>::getTypeDescription());
//
////  pChargeLabel = VarLabel::create("p.charge", ParticleVariable<double>::getTypeDescription());
////  pChargeLabel_preReloc = VarLabel::create("p.charge+", ParticleVariable<double>::getTypeDescription());
//
//  // PER SYSTEM VARIABLES (reduction variables)
//  // ********************
//  //1---> for nonbonded calculation
//  nonbondedEnergyLabel = VarLabel::create("nonbondedEnergy", sum_vartype::getTypeDescription());
//  nonbondedStressLabel = VarLabel::create("nonbondedStress", matrix_sum::getTypeDescription());
//  //2---> for electrostatic calculation
//  //2.1---> real space
//  electrostaticRealEnergyLabel       = VarLabel::create("electrostaticRealEnergy", sum_vartype::getTypeDescription());
//  electrostaticRealStressLabel       = VarLabel::create("electrostaticRealStress", matrix_sum::getTypeDescription());
//  //2.2--->  reciprocal space
//  electrostaticReciprocalEnergyLabel = VarLabel::create("electrostaticReciprocalEnergy", sum_vartype::getTypeDescription());
//  electrostaticReciprocalStressLabel = VarLabel::create("electrostaticReciprocalStress", matrix_sum::getTypeDescription());
//  //3---> for valence calculation
//  //3.1---> by component
//  bondEnergyLabel    = VarLabel::create("bondEnergy", sum_vartype::getTypeDescription());
//  bondStressLabel    = VarLabel::create("bondStress", matrix_sum::getTypeDescription());
//  bendEnergyLabel    = VarLabel::create("bendEnergy", sum_vartype::getTypeDescription());
//  bendStressLabel    = VarLabel::create("bendStress", matrix_sum::getTypeDescription());
//  torsionEnergyLabel = VarLabel::create("torsionEnergy", sum_vartype::getTypeDescription());
//  torsionStressLabel = VarLabel::create("torsionStress", matrix_sum::getTypeDescription());
//  oopEnergyLabel     = VarLabel::create("improperTorsionEnergy", sum_vartype::getTypeDescription());
//  oopStressLabel     = VarLabel::create("improperTorsionStress", matrix_sum::getTypeDescription());
//  //3.2---> total
//  valenceEnergyLabel = VarLabel::create("valenceEnergy", sum_vartype::getTypeDescription());
//  valenceStressLabel = VarLabel::create("valenceStress", matrix_sum::getTypeDescription());
//
//  ///////////////////////////////////////////////////////////////////////////
//  // Sole Variables - Nonbonded
//  nonbondedDependencyLabel = VarLabel::create("nonbondedDependency", SoleVariable<double>::getTypeDescription());
//
//
//#ifdef HAVE_FFTW
//  ///////////////////////////////////////////////////////////////////////////
//  // Sole Variables - SPME
//  forwardTransformPlanLabel = VarLabel::create("forwardTransformPlan", SoleVariable<fftw_plan>::getTypeDescription());
//  backwardTransformPlanLabel = VarLabel::create("backwardTransformPlan", SoleVariable<fftw_plan>::getTypeDescription());
//  electrostaticsDependencyLabel = VarLabel::create("electrostaticsDependency", SoleVariable<double>::getTypeDescription());
//  subSchedulerDependencyLabel = VarLabel::create("subschedulerDependency", CCVariable<int>::getTypeDescription());
//#endif
//}
//
//MDLabel::~MDLabel()
//{
//	// PER PARTICLE VARIABLES
//	//   Force calculation Variables
////	VarLabel::destroy(pRealDipoles);
////	VarLabel::destroy(pRealDipoles_preReloc);
//	VarLabel::destroy(pElectrostaticsRealForce);
//	VarLabel::destroy(pElectrostaticsRealForce_preReloc);
////	VarLabel::destroy(pReciprocalDipoles);
////	VarLabel::destroy(pReciprocalDipoles_preReloc);
//	VarLabel::destroy(pElectrostaticsReciprocalForce);
//	VarLabel::destroy(pTotalDipoles);
//	VarLabel::destroy(pTotalDipoles_preReloc);
//	VarLabel::destroy(pElectrostaticsReciprocalForce_preReloc);
//	VarLabel::destroy(pNonbondedForceLabel);
//	VarLabel::destroy(pNonbondedForceLabel_preReloc);
//	VarLabel::destroy(pValenceForceLabel);
//	VarLabel::destroy(pValenceForceLabel_preReloc);
//	//   Integrator variables
//	VarLabel::destroy(pXLabel);
//	VarLabel::destroy(pXLabel_preReloc);
////	VarLabel::destroy(pAccelLabel);
////	VarLabel::destroy(pAccelLabel_preReloc);
//	VarLabel::destroy(pVelocityLabel);
//	VarLabel::destroy(pVelocityLabel_preReloc);
//	//   General
//	VarLabel::destroy(pParticleIDLabel);
//	VarLabel::destroy(pParticleIDLabel_preReloc);
//
//	// PER SYSTEM VARIABLES
//  VarLabel::destroy(nonbondedEnergyLabel);
//  VarLabel::destroy(nonbondedStressLabel);
//  VarLabel::destroy(electrostaticRealEnergyLabel);
//  VarLabel::destroy(electrostaticRealStressLabel);
//  VarLabel::destroy(electrostaticReciprocalEnergyLabel);
//  VarLabel::destroy(electrostaticReciprocalStressLabel);
//  VarLabel::destroy(bondEnergyLabel);
//  VarLabel::destroy(bondStressLabel);
//  VarLabel::destroy(bendEnergyLabel);
//  VarLabel::destroy(bendStressLabel);
//  VarLabel::destroy(torsionEnergyLabel);
//  VarLabel::destroy(torsionStressLabel);
//  VarLabel::destroy(oopEnergyLabel);
//  VarLabel::destroy(oopStressLabel);
//  VarLabel::destroy(valenceEnergyLabel);
//  VarLabel::destroy(valenceStressLabel);
//
//
//// !!!!! TO BE DELETED
////  VarLabel::destroy(pEnergyLabel);
////  VarLabel::destroy(pEnergyLabel_preReloc);
////  VarLabel::destroy(pMassLabel);
////  VarLabel::destroy(pMassLabel_preReloc);
////  VarLabel::destroy(pChargeLabel);
////  VarLabel::destroy(pChargeLabel_preReloc);
//// !!!!! TO BE DELETED (end)
//
//
//  ///////////////////////////////////////////////////////////////////////////
//  // Sole Variables
//  VarLabel::destroy(forwardTransformPlanLabel);
//  VarLabel::destroy(backwardTransformPlanLabel);
//  VarLabel::destroy(electrostaticsDependencyLabel);
//  VarLabel::destroy(subSchedulerDependencyLabel);
//}
