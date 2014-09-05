/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <CCA/Components/MPM/ConstitutiveModel/ViscoPlastic.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/YieldConditionFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/StabilityCheckFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/ViscoPlasticityModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DamageModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/MPMEquationOfStateFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PlasticityState.h>

#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h> //for Fracture
#include <Core/Math/TangentModulusTensor.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <math.h>

#include <iostream>


using namespace std;
using namespace Uintah;

static DebugStream cout_CST("HEP",false);
static DebugStream cout_CST1("HEP1",false);
static DebugStream CSTi("HEPi",false);
static DebugStream CSTir("HEPir",false);
static DebugStream cout_visco("ViscoPlastic", false);

ViscoPlastic::ViscoPlastic(ProblemSpecP& ps, MPMFlags* Mflag) :
  ConstitutiveModel(Mflag), ImplicitCM()
  
//   ps->require: mandatory input; ps->get: optional input with default given here
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  d_initialData.alpha = 1.0e-5; // default is per K
  ps->get("coeff_thermal_expansion", d_initialData.alpha);
  d_useModifiedEOS = false;
  ps->get("useModifiedEOS",d_useModifiedEOS);
  d_removeParticles = false;
  ps->get("remove_particles",d_removeParticles);
  d_setStressToZero = true;
  ps->get("zero_stress_upon_failure",d_setStressToZero);
  d_allowNoTension = false;
  ps->get("allow_no_tension",d_allowNoTension);
  d_checkFailure = false;
  ps->get("check_failure", d_checkFailure);
  d_usePolarDecompositionRMB = true;
  ps->get("use_polar_decomposition_RMB", d_usePolarDecompositionRMB);



  // Get the failure variable data
  getFailureVariableData(ps);

  d_tol = 1.0e-10;
  ps->get("tolerance",d_tol);
  d_initialMaterialTemperature = 269.15;
  ps->get("initial_material_temperature",d_initialMaterialTemperature);

  d_stable = StabilityCheckFactory::create(ps);
  if(!d_stable) cerr << "Stability check disabled\n";

  d_plastic = ViscoPlasticityModelFactory::create(ps);
  if(!d_plastic){
    ostringstream desc;
    desc << "An error occured in the ViscoPlasticityModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " ffjhl.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  
  d_eos = MPMEquationOfStateFactory::create(ps);
  d_eos->setBulkModulus(d_initialData.Bulk);
  if(!d_eos){
    ostringstream desc;
    desc << "An error occured in the EquationOfStateFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " ffjhl.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  // Initialize local VarLabels
  initializeLocalMPMLabels();

}

ViscoPlastic::ViscoPlastic(const ViscoPlastic* cm)
  : ConstitutiveModel(cm), ImplicitCM(cm)
{
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  d_initialData.alpha = cm->d_initialData.alpha;
  d_useModifiedEOS = cm->d_useModifiedEOS;
  d_removeParticles = cm->d_removeParticles;
  d_setStressToZero = cm->d_setStressToZero;
//  d_checkFailureMaxTensileStress = cm->d_checkFailureMaxTensileStress;
//   d_evolvePorosity = cm->d_evolvePorosity;
//   d_evolveDamage = cm->d_evolveDamage;
//   d_checkTeplaFailureCriterion = cm->d_checkTeplaFailureCriterion;
  d_tol = cm->d_tol ;
  d_initialMaterialTemperature = cm->d_initialMaterialTemperature ;
  
//   d_yield = YieldConditionFactory::createCopy(cm->d_yield);
  d_stable = StabilityCheckFactory::createCopy(cm->d_stable);
  d_plastic = ViscoPlasticityModelFactory::createCopy(cm->d_plastic);
//   d_damage = DamageModelFactory::createCopy(cm->d_damage);
  d_eos = MPMEquationOfStateFactory::createCopy(cm->d_eos);
  d_eos->setBulkModulus(d_initialData.Bulk);
  
  // Initialize local VarLabels
  initializeLocalMPMLabels();

  // Set the failure strain data
  setFailureVariableData(cm);
}

ViscoPlastic::~ViscoPlastic()
{
  // Destructor 
  VarLabel::destroy(pLeftStretchLabel);
  VarLabel::destroy(pRotationLabel);
  VarLabel::destroy(pStrainRateLabel);
  VarLabel::destroy(pPlasticStrainLabel);
//   VarLabel::destroy(pDamageLabel);
//   VarLabel::destroy(pPorosityLabel);
  VarLabel::destroy(pLocalizedLabel);
  VarLabel::destroy(pPlasticTempLabel);
  VarLabel::destroy(pPlasticTempIncLabel);
  VarLabel::destroy(pFailureVariableLabel);

  VarLabel::destroy(pLeftStretchLabel_preReloc);
  VarLabel::destroy(pRotationLabel_preReloc);
  VarLabel::destroy(pStrainRateLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
//   VarLabel::destroy(pDamageLabel_preReloc);
//   VarLabel::destroy(pPorosityLabel_preReloc);
  VarLabel::destroy(pLocalizedLabel_preReloc);
  VarLabel::destroy(pPlasticTempLabel_preReloc);
  VarLabel::destroy(pPlasticTempIncLabel_preReloc);

  VarLabel::destroy(pFailureVariableLabel_preReloc);

  delete d_plastic;
//   delete d_yield;
  delete d_stable;
//   delete d_damage;
  delete d_eos;
}

void ViscoPlastic::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","visco_plastic");
  }

  cm_ps->appendElement("bulk_modulus",d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",d_initialData.Shear);
  cm_ps->appendElement("coeff_thermal_expansion", d_initialData.alpha);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
  cm_ps->appendElement("remove_particles",d_removeParticles);
  cm_ps->appendElement("zero_stress_upon_failure",d_setStressToZero);
  cm_ps->appendElement("allow_no_tension",d_allowNoTension);
  cm_ps->appendElement("use_polar_decomposition_RMB", d_usePolarDecompositionRMB);
//   cm_ps->appendElement("evolve_porosity",d_evolvePorosity);
//   cm_ps->appendElement("evolve_damage",d_evolveDamage);
//   cm_ps->appendElement("check_TEPLA_failure_criterion",
//                        d_checkTeplaFailureCriterion);
  cm_ps->appendElement("tolerance",d_tol);
  cm_ps->appendElement("initial_material_temperature",
                       d_initialMaterialTemperature);

  cm_ps->appendElement("failure_variable_mean",d_varf.mean);
  cm_ps->appendElement("failure_variable_std",d_varf.std);
  cm_ps->appendElement("failure_variable_distrib",d_varf.dist);
  cm_ps->appendElement("failure_by_stress",d_varf.failureByStress);
  cm_ps->appendElement("failure_by_pressure",d_varf.failureByPressure);


//  cm_ps->appendElement("check_failure_max_tensile_stress",
//        d_checkFailureMaxTensileStress);

//   d_yield->outputProblemSpec(cm_ps);
  d_stable->outputProblemSpec(cm_ps);
  d_plastic->outputProblemSpec(cm_ps);
//   d_damage->outputProblemSpec(cm_ps);
  d_eos->outputProblemSpec(cm_ps);
}



ViscoPlastic* ViscoPlastic::clone()
{
  return scinew ViscoPlastic(*this);
}

void
ViscoPlastic::initializeLocalMPMLabels()
{
  pLeftStretchLabel = VarLabel::create("p.leftStretch",
        ParticleVariable<Matrix3>::getTypeDescription());
  pRotationLabel = VarLabel::create("p.rotation",
        ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel = VarLabel::create("p.strainRate",
        ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
        ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel = VarLabel::create("p.localized",
        ParticleVariable<int>::getTypeDescription());
  pPlasticTempLabel = VarLabel::create("p.plasticTemp",
        ParticleVariable<double>::getTypeDescription());
  pPlasticTempIncLabel = VarLabel::create("p.plasticTempInc",
        ParticleVariable<double>::getTypeDescription());
  pFailureVariableLabel = VarLabel::create("p.varf",
                        ParticleVariable<double>::getTypeDescription());


  pLeftStretchLabel_preReloc = VarLabel::create("p.leftStretch+",
        ParticleVariable<Matrix3>::getTypeDescription());
  pRotationLabel_preReloc = VarLabel::create("p.rotation+",
        ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel_preReloc = VarLabel::create("p.strainRate+",
        ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
        ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel_preReloc = VarLabel::create("p.localized+",
        ParticleVariable<int>::getTypeDescription());
  pPlasticTempLabel_preReloc = VarLabel::create("p.plasticTemp+",
        ParticleVariable<double>::getTypeDescription());
  pPlasticTempIncLabel_preReloc = VarLabel::create("p.plasticTempInc+",
        ParticleVariable<double>::getTypeDescription());
  pFailureVariableLabel_preReloc = VarLabel::create("p.varf+",
                         ParticleVariable<double>::getTypeDescription());

}

void 
ViscoPlastic::getFailureVariableData(ProblemSpecP& ps)
{
  d_varf.mean = 1.0e30; // Mean failure stress
  d_varf.std = 0.0;  // STD failure strain
  d_varf.dist = "constant";
  d_varf.failureByStress = true; // failure by stress default
  d_varf.failureByPressure = false; // failure by mean stress
  ps->get("failure_variable_mean",    d_varf.mean);
  ps->get("failure_variable_std",     d_varf.std);
  ps->get("failure_variable_distrib", d_varf.dist);
  ps->get("failure_by_stress", d_varf.failureByStress);
  ps->get("failure_by_pressure", d_varf.failureByPressure);
}

void
ViscoPlastic::setFailureVariableData(const ViscoPlastic* cm)
{
  d_varf.mean = cm->d_varf.mean;
  d_varf.std = cm->d_varf.std;
  d_varf.dist = cm->d_varf.dist;
  d_varf.failureByStress = cm->d_varf.failureByStress;
  d_varf.failureByPressure = cm->d_varf.failureByPressure;
}

void 
ViscoPlastic::addParticleState(std::vector<const VarLabel*>& from,
                                     std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pLeftStretchLabel);
  from.push_back(pRotationLabel);
  from.push_back(pStrainRateLabel);
  from.push_back(pPlasticStrainLabel);
//   from.push_back(pDamageLabel);
//   from.push_back(pPorosityLabel);
  from.push_back(pLocalizedLabel);
  from.push_back(pPlasticTempLabel);
  from.push_back(pPlasticTempIncLabel);
  from.push_back(pFailureVariableLabel);

  to.push_back(pLeftStretchLabel_preReloc);
  to.push_back(pRotationLabel_preReloc);
  to.push_back(pStrainRateLabel_preReloc);
  to.push_back(pPlasticStrainLabel_preReloc);
//   to.push_back(pDamageLabel_preReloc);
//   to.push_back(pPorosityLabel_preReloc);
  to.push_back(pLocalizedLabel_preReloc);
  to.push_back(pPlasticTempLabel_preReloc);
  to.push_back(pPlasticTempIncLabel_preReloc);
  to.push_back(pFailureVariableLabel_preReloc);

  // Add the particle state for the plasticity model
  d_plastic->addParticleState(from, to);
}

void 
ViscoPlastic::addInitialComputesAndRequires(Task* task,
                                                  const MPMMaterial* matl,
                                                  const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pLeftStretchLabel, matlset);
  task->computes(pRotationLabel, matlset);
  task->computes(pStrainRateLabel, matlset);
  task->computes(pPlasticStrainLabel, matlset);
//   task->computes(pDamageLabel, matlset);
//   task->computes(pPorosityLabel, matlset);
  task->computes(pLocalizedLabel, matlset);
  task->computes(pPlasticTempLabel, matlset);
  task->computes(pPlasticTempIncLabel, matlset);
  task->computes(pFailureVariableLabel, matlset);
  task->computes(lb->TotalLocalizedParticleLabel);

  // Add internal evolution variables computed by plasticity model
  d_plastic->addInitialComputesAndRequires(task, matl, patch);
}

void 
ViscoPlastic::initializeCMData(const Patch* patch,
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  if (flag->d_integrator == MPMFlags::Implicit) 
    initSharedDataForImplicit(patch, matl, new_dw);
  else {
    initSharedDataForExplicit(patch, matl, new_dw);
    computeStableTimestep(patch, matl, new_dw);
  }

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  //cout << "Initialize CM Data in ViscoPlastic" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pLeftStretch, pRotation;
  ParticleVariable<double> pPlasticStrain, pDamage, pPorosity, pStrainRate;
  ParticleVariable<int> pLocalized;
  ParticleVariable<double> pPlasticTemperature, pPlasticTempInc;
  ParticleVariable<double> pFailureVariable;

  new_dw->allocateAndPut(pLeftStretch, pLeftStretchLabel, pset);
  new_dw->allocateAndPut(pRotation, pRotationLabel, pset);
  new_dw->allocateAndPut(pStrainRate, pStrainRateLabel, pset);
  new_dw->allocateAndPut(pPlasticStrain, pPlasticStrainLabel, pset);
//   new_dw->allocateAndPut(pDamage, pDamageLabel, pset);
  new_dw->allocateAndPut(pLocalized, pLocalizedLabel, pset);
//   new_dw->allocateAndPut(pPorosity, pPorosityLabel, pset);
  new_dw->allocateAndPut(pPlasticTemperature, pPlasticTempLabel, pset);
  new_dw->allocateAndPut(pPlasticTempInc, pPlasticTempIncLabel, pset);
  new_dw->allocateAndPut(pFailureVariable, pFailureVariableLabel, pset);

  // Initialize a gaussian random number generator
  SCIRun::Gaussian gaussGen(d_varf.mean, d_varf.std, 0, 1, DBL_MAX);


  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    pLeftStretch[*iter] = one;
    pRotation[*iter] = one;
    pStrainRate[*iter] = 0.0;
    pPlasticStrain[*iter] = 0.0;
//     pDamage[*iter] = d_damage->initialize();
//     pPorosity[*iter] = d_porosity.f0;
    pLocalized[*iter] = 0;
    pPlasticTemperature[*iter] = d_initialMaterialTemperature;
    pPlasticTempInc[*iter] = 0.0;

    if (d_varf.dist == "constant") {
      pFailureVariable[*iter] = d_varf.mean;
    } else {
      pFailureVariable[*iter] = fabs(gaussGen.rand(1.0));
    }

  }


  // Initialize the data for the plasticity model
  d_plastic->initializeInternalVars(pset, new_dw);
}

void 
ViscoPlastic::computeStableTimestep(const Patch* patch,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);

  constParticleVariable<double> pMass, pVolume;
  constParticleVariable<Vector> pVelocity;

  new_dw->get(pMass,     lb->pMassLabel,     pset);
  new_dw->get(pVolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double shear = d_initialData.Shear;
  double bulk = d_initialData.Bulk;

  ParticleSubset::iterator iter = pset->begin(); 
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    Vector pvelocity_idx = pVelocity[idx];
    if(pMass[idx] > 0){
      c_dil = sqrt((bulk + 4.0*shear/3.0)*pVolume[idx]/pMass[idx]);
    }
    else{
      c_dil = 0.0;
      pvelocity_idx = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void 
ViscoPlastic::addComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  if (flag->d_integrator == MPMFlags::Implicit) {
//    addSharedCRForImplicit(task, matlset, patches);
    addSharedCRForImplicitHypo(task, matlset, patches);
  } else {
    addSharedCRForHypoExplicit(task, matlset, patches);
  }


  // Other constitutive model and input dependent computes and requires
  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 

  task->requires(Task::OldDW, pLeftStretchLabel,     matlset, gnone);
  task->requires(Task::OldDW, pRotationLabel,        matlset, gnone);
  task->requires(Task::OldDW, pStrainRateLabel,      matlset, gnone);
  task->requires(Task::OldDW, pPlasticStrainLabel,   matlset, gnone);
//   task->requires(Task::OldDW, pDamageLabel,          matlset, gnone);
//   task->requires(Task::OldDW, pPorosityLabel,        matlset, gnone);
  task->requires(Task::OldDW, pLocalizedLabel,       matlset, gnone);
  task->requires(Task::OldDW, pPlasticTempLabel,     matlset, gnone);
  task->requires(Task::OldDW, pPlasticTempIncLabel,  matlset, gnone);
  task->requires(Task::OldDW, pFailureVariableLabel,  matlset, gnone);
  task->requires(Task::OldDW, lb->pParticleIDLabel,  matlset, gnone);

  task->computes(pLeftStretchLabel_preReloc,    matlset);
  task->computes(pRotationLabel_preReloc,       matlset);
  task->computes(pStrainRateLabel_preReloc,     matlset);
  task->computes(pPlasticStrainLabel_preReloc,  matlset);
//   task->computes(pDamageLabel_preReloc,         matlset);
//   task->computes(pPorosityLabel_preReloc,       matlset);
  task->computes(pLocalizedLabel_preReloc,      matlset);
  task->computes(pPlasticTempLabel_preReloc,    matlset);
  task->computes(pPlasticTempIncLabel_preReloc, matlset);
  task->computes(pFailureVariableLabel_preReloc, matlset);
  task->computes(lb->TotalLocalizedParticleLabel);
  // Add internal evolution variables computed by plasticity model
  d_plastic->addComputesAndRequires(task, matl, patches);

}

////////// Delegate as much to d_plastic as possible.
void 
ViscoPlastic::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
         
  //if ((patches->get(0))->getID() == 19)
  //  cout_CST << getpid() 
  //           << "ComputeStressTensor: In : Matl = " << matl << " id = " 
  //           << matl->getDWIndex() <<  " patch = " 
  //           << (patches->get(0))->getID();
  // General stuff
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  Matrix3 tensorL(0.0); // Velocity Gradient
  Matrix3 tensorD(0.0); // Rate of deformation
  Matrix3 tensorW(0.0); // Spin 
  Matrix3 tensorF; tensorF.Identity(); // Deformation gradient
  Matrix3 tensorV; tensorV.Identity(); // Left Cauchy-Green stretch
  Matrix3 tensorR; tensorR.Identity(); // Rotation 
  Matrix3 tensorSig(0.0); // The Cauchy stress
  Matrix3 tensorEta(0.0); // Deviatoric part of tensor D
  Matrix3 tensorS(0.0); // Devaitoric part of tensor Sig
  Matrix3 tensorF_new; tensorF_new.Identity(); // Deformation gradient
  Matrix3 stressRate(0.0); //stress rate from plasticity
  Matrix3 devTrialStress(0.0);
  int implicitFlag=0;   //0 for explicit, 0 for implicit;

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double alpha = d_initialData.alpha;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();

  double totalStrainEnergy = 0.0;
  double epdot; //plastic strain rate (1/sec)

  // Do thermal expansion?
  if(!flag->d_doThermalExpansion){
    alpha = 0;
  }

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    //cerr << getpid() << " patch = " << patch->getID() << endl;
    // Get grid size
    Vector dx = patch->dCell();

    long64 totalLocalizedParticle = 0;
    // Get the set of particles
    int dwi = matl->getDWIndex();

    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the particle location, particle, particle mass, particle volume
    constParticleVariable<double> pMass, pVolume;
    old_dw->get(pMass, lb->pMassLabel, pset);

    constParticleVariable<Vector> pVelocity;
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);

    // Get the particle stress and temperature
    constParticleVariable<Matrix3> pStress;
    constParticleVariable<double>  pTempPrev, pTemperature;
    old_dw->get(pStress,      lb->pStressLabel,       pset);
    old_dw->get(pTempPrev,    lb->pTempPreviousLabel, pset); 
    old_dw->get(pTemperature, lb->pTemperatureLabel,  pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

   //Get ParticleID
   constParticleVariable<long64> pParticleID;
   old_dw->get(pParticleID, lb->pParticleIDLabel, pset);

    // GET LOCAL DATA 

    // Get the left stretch (V) and rotation (R)
    constParticleVariable<Matrix3> pLeftStretch, pRotation;
    old_dw->get(pLeftStretch, pLeftStretchLabel, pset);
    old_dw->get(pRotation, pRotationLabel, pset);

    // Get the particle plastic temperature
    constParticleVariable<double> pPlasticTemperature, pPlasticTempInc;
    old_dw->get(pPlasticTemperature, pPlasticTempLabel, pset);
    old_dw->get(pPlasticTempInc, pPlasticTempIncLabel, pset);

    // Get the particle damage state
    constParticleVariable<double> pPlasticStrain, pDamage, pPorosity, 
      pStrainRate, pFailureVariable;

    old_dw->get(pFailureVariable, pFailureVariableLabel, pset);
    old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
    old_dw->get(pStrainRate, pStrainRateLabel, pset);

    // Get the particle localization state
    constParticleVariable<int> pLocalized;
    old_dw->get(pLocalized, pLocalizedLabel, pset);

    // Create and allocate arrays for storing the updated information
    // GLOBAL
    ParticleVariable<Matrix3>  pStress_new;
    constParticleVariable<double> pVolume_deformed;
    constParticleVariable<Matrix3>  pDeformGrad, pDeformGrad_new, velGrad;
    old_dw->get(pDeformGrad, lb->pDeformationMeasureLabel, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->get(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->get(pVolume_deformed, lb->pVolumeLabel_preReloc,      pset);
    new_dw->get(velGrad, lb->pVelGradLabel_preReloc,              pset);

    // LOCAL
    ParticleVariable<Matrix3> pLeftStretch_new, pRotation_new;
    ParticleVariable<double>  pPlasticStrain_new, pDamage_new, pPorosity_new, 
      pStrainRate_new;
    ParticleVariable<double>  pPlasticTemperature_new, pPlasticTempInc_new, pFailureVariable_new;
    ParticleVariable<int>     pLocalized_new;
    new_dw->allocateAndPut(pLeftStretch_new, 
                           pLeftStretchLabel_preReloc,            pset);
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pPlasticTemperature_new,      
                           pPlasticTempLabel_preReloc,            pset);
    new_dw->allocateAndPut(pPlasticTempInc_new,      
                           pPlasticTempIncLabel_preReloc,         pset);
    new_dw->allocateAndPut(pFailureVariable_new,      
                           pFailureVariableLabel_preReloc,            pset);

    // Allocate variable to store internal heating rate
    ParticleVariable<double> pdTdt, p_q;
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel_preReloc, 
                           pset);
    new_dw->allocateAndPut(p_q,   lb->p_qLabel_preReloc,          pset);

    // Get yield, drag, back stresses
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);

    // Copy failure variable to new dw
    pFailureVariable_new.copyData(pFailureVariable);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 

    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      double J = pDeformGrad_new[idx].Determinant();

      if(d_setStressToZero && pLocalized[idx]){
        J = pDeformGrad[idx].Determinant();
        tensorF_new = pDeformGrad[idx];
      }

      tensorL=velGrad[idx];

      if (!(J > 0.0)) {
        cerr << getpid() ;
        cerr << "**ERROR** Negative Jacobian of deformation gradient"
             << " in particle " << pParticleID[idx] << endl;
        cerr << "l = " << velGrad[idx] << endl;
        cerr << "F_old = " << pDeformGrad[idx] << endl;
        cerr << "F_new = " << pDeformGrad_new[idx] << endl;
        cerr << "J = " << J << endl;
        throw ParameterNotFound("**ERROR**:ViscoPlastic", __FILE__, __LINE__);
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;
      tensorF_new.polarDecompositionRMB(tensorV, tensorR);

      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorW = (tensorL - tensorL.Transpose())*0.5;
      for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
          tensorD(ii,jj)=(fabs(tensorD(ii,jj)) < d_tol) ? 0.0 : tensorD(ii,jj);
          tensorW(ii,jj)=(fabs(tensorW(ii,jj)) < d_tol) ? 0.0 : tensorW(ii,jj);
        }
      }

      // Update the kinematic variables
      pLeftStretch_new[idx] = tensorV;
      pRotation_new[idx] = tensorR;

      // If the particle is just sitting there, do nothing
      double defRateSq = tensorD.NormSquared();
      if (!(defRateSq > 0) || pLocalized[idx]==1) {
        pStress_new[idx] = pStress[idx];
        pStrainRate_new[idx] = 0.0;
//         pPlasticStrain_new[idx] = 0.0;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
//         pDamage_new[idx] = pDamage[idx];
//         pPorosity_new[idx] = pPorosity[idx];
        pLocalized_new[idx] = pLocalized[idx];
        if (pLocalized_new[idx]==1) totalLocalizedParticle+=1;
        pFailureVariable_new[idx] = pFailureVariable[idx];
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx];
        pPlasticTempInc_new[idx] = 0.0;
        p_q[idx] = 0.0;
        d_plastic->updateElastic(idx);
        continue;
      }

      // Rotate the total rate of deformation tensor back to the 
      // material configuration
      tensorD = (tensorR.Transpose())*(tensorD*tensorR);

      // Subtract the thermal expansion to get D_e + D_p
      double dT_dt = (pTemperature[idx] - pTempPrev[idx])/delT;
      tensorD -= one*(alpha*dT_dt);
      
      // Calculate the deviatoric part of the non-thermal part
      // of the rate of deformation tensor
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);
      pStrainRate_new[idx] = sqrt(tensorD.NormSquared()/1.5);
      
//       cout << "pStrainRate_new= " <<pStrainRate_new[idx] << " \n"; 

      // Rotate the Cauchy stress back to the 
      // material configuration and calculate the deviatoric part
      tensorSig = pStress[idx];
      tensorSig = (tensorR.Transpose())*(tensorSig*tensorR);
      double pressure = tensorSig.Trace()/3.0;
      Matrix3 tensorP = one*pressure;
      tensorS = tensorSig - tensorP;

//        cout << "tensorD=" << tensorD << "\n";
//        cout << "tensorS=" << tensorS << "\n";

      // Calculate the temperature at the start of the time step
      double temperature = pTemperature[idx];

//       // Calculate the equivalent plastic strain and strain rate
//       double epdot = sqrt(tensorEta.NormSquared()/1.5);
//       double ep = pPlasticStrain[idx] + epdot*delT;

      // Get the specific heat
      double C_p = matl->getSpecificHeat();

      // Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->strainRate = pStrainRate_new[idx];
//       state->plasticStrainRate = epdot;
//       state->plasticStrain = ep;
      state->pressure = pressure;
      state->temperature = temperature;
      state->initialTemperature = d_initialMaterialTemperature;
      state->density = rho_cur;
      state->initialDensity = rho_0;
      state->volume = pVolume_deformed[idx];
      state->initialVolume = pMass[idx]/rho_0;
      state->bulkModulus = bulk ;
      state->initialBulkModulus = bulk;
      state->shearModulus = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp = Tm ;
      state->initialMeltTemp = Tm;
      state->specificHeat = C_p;
    
      // Calculate the shear modulus and the melting temperature at the
      // start of the time step
      double mu_cur = d_plastic->computeShearModulus(state);
      double Tm_cur = d_plastic->computeMeltingTemp(state);

      // Update the plasticity state
      state->shearModulus = mu_cur ;
      state->meltingTemp = Tm_cur ;

      // compute the local wave speed
      double c_dil = sqrt((bulk + 4.0*mu_cur/3.0)/rho_cur);

     // what's:
     // a. Flow stress - to determine elastic or plastic
     // b. inelastic strain rate; total,yield, drag, back stresses
     // c. ElasPlasTangentModulus

     // Calculate flow stress at beginning of time step
      double flowStress = d_plastic->computeFlowStress(idx,
                                pStress[idx], tensorR,implicitFlag);

      //already localized; copy the old values to new
      if (pLocalized[idx]==1) {
         totalLocalizedParticle+=1;
         if (d_setStressToZero) {
             pStress_new[idx] = zero;
         } else if (d_allowNoTension) {
             double pressure = (1.0/3.0)*pStress_new[idx].Trace();
             if (pressure > 0.0) pStress_new[idx] = zero;
             else pStress_new[idx] = one*pressure;
         } else {
             pStress_new[idx] = pStress[idx];
         }

         pStrainRate_new[idx] =  pStrainRate[idx];
         pPlasticStrain_new[idx] = pPlasticStrain[idx];
         pLocalized_new[idx] = pLocalized[idx];
         pFailureVariable_new[idx] = pFailureVariable[idx];
         pPlasticTemperature_new[idx]=pPlasticTemperature[idx];
         pPlasticTempInc_new[idx] = 0.0;
         d_plastic->updateElastic(idx);
      } else {


      // Elastic - Compute the deviatoric stress
//       cout << "flowStress=" << flowStress << "\n";
      if (flowStress <= 0.0) {

      // Integrate the stress rate equation to get deviatoric stress
        Matrix3 trialS = tensorS + tensorEta*(2.0*mu_cur*delT);

        // Do the standard hypoelastic-plastic stress update
        // Calculate the updated hydrostatic stress
        double p = d_eos->computePressure(matl, state, tensorF_new, tensorD, 
                                        delT);
        //p -= qVisco;

        Matrix3 tensorHy = one*p;

        // Get the elastic stress
        tensorSig = trialS + tensorHy;

        // Rotate the stress rate back to the laboratory coordinates
        // to get the "true" Cauchy stress
        tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

        // Save the updated data
        pStress_new[idx] = tensorSig;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
//      cout << "elastic stress=" << pStress_new[idx] << "\n";
        
        // Update the internal variables
        d_plastic->updateElastic(idx);

        // Update the temperature
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx];
        pPlasticTempInc_new[idx] = 0.0;

        // Compute stability criterion
        pLocalized_new[idx] = pLocalized[idx];
        pFailureVariable_new[idx] = pFailureVariable[idx];
        if (pLocalized_new[idx]==1) totalLocalizedParticle+=1;  
      // plastic
      } else {

        //plastic strain rate, plastic strain; back,drag,yield stresses updated
        TangentModulusTensor Ce, Cep;
        computeElasticTangentModulus(bulk, shear, Ce);
        d_plastic->computeStressIncTangent(epdot,stressRate,Cep,delT,idx,Ce,
              tensorD,pStress[idx],implicitFlag, tensorR);
//         cout << "epInc=" << epdot*delT << "\n";
        
        // Calculate total stress
        tensorSig = pStress[idx]+stressRate*delT;
        
        state->plasticStrainRate = epdot;
        pPlasticStrain_new[idx]=pPlasticStrain[idx]+epdot*delT;
        state->plasticStrain = pPlasticStrain_new[idx]; 
        
//      cout << "plasticStrain= " << pPlasticStrain_new[idx] << " \n";
        // Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = 0.9;

        // Alternative approach 
        devTrialStress = tensorSig - one*(tensorSig.Trace()/3.0);
      
        // Calculate the equivalent stress
        double equivStress = sqrt((devTrialStress.NormSquared())*1.5);
        
        double Tdot = equivStress*epdot*taylorQuinney/(rho_cur*C_p);
        pdTdt[idx] = Tdot;
        double dT = Tdot*delT;
        pPlasticTempInc_new[idx] = dT;
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx] + dT; 
        double temp_new = temperature + dT;

        // Find if the particle has localized
        pLocalized_new[idx] = pLocalized[idx];

        bool isLocalized = false;

        if (d_checkFailure && pLocalized[idx]==0)  {
             isLocalized=updateFailedParticlesAndModifyStress(tensorF_new,
             pFailureVariable[idx], pLocalized[idx], pLocalized_new[idx], 
             tensorSig, pParticleID[idx], temp_new, Tm_cur);
        }
        if (pLocalized_new[idx]==1) totalLocalizedParticle+=1;
          // Rotate the stress back to the laboratory coordinates
          // Save the new data
          tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());
          pStress_new[idx] = tensorSig;
        if (isLocalized) {
          d_plastic->updateElastic(idx);
        } 
//      cout << "pStress= " << pStress_new[idx] << " \n";

      } // end plastic
   }  // end pLocalized
   
      // Rotate the deformation rate back to the laboratory coordinates
      tensorD = (tensorR*tensorD)*(tensorR.Transpose());

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        Matrix3 avgStress = (pStress_new[idx] + pStress[idx])*0.5;
        double pStrainEnergy = (tensorD(0,0)*avgStress(0,0) +
                                tensorD(1,1)*avgStress(1,1) +
                                tensorD(2,2)*avgStress(2,2) +
                                2.0*(tensorD(0,1)*avgStress(0,1) + 
                                     tensorD(0,2)*avgStress(0,2) +
                                     tensorD(1,2)*avgStress(1,2)))*
          pVolume_deformed[idx]*delT;
        totalStrainEnergy += pStrainEnergy;
      }  //else {          
      //   totalLocalizedParticle+=1;
      //}

      if (cout_visco.active()) {
	cout_visco << " totalLocalizedParticle = " << totalLocalizedParticle
	           << " pLocalized_new[idx] = " << pLocalized_new[idx] <<endl;
      }

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      delete state;
    } //end iterator

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }

     new_dw->put(sumlong_vartype(totalLocalizedParticle),
          lb->TotalLocalizedParticleLabel);

  } //end patch

  // cout_CST << getpid() << "... Out" << endl;
} //end method

void 
ViscoPlastic::computeStressTensorImplicit(const PatchSubset* patches,
                                                const MPMMaterial* matl,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
//   Constants
  int dwi = matl->getDWIndex();
  double sqrtTwoThird = sqrt(2.0/3.0);
  Ghost::GhostType gac = Ghost::AroundCells;
  Matrix3 One; One.Identity(); Matrix3 Zero(0.0);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double alpha = d_initialData.alpha;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();
  int implicitFlag = 1;

  // Do thermal expansion?
  if(!flag->d_doThermalExpansion){
    alpha = 0;
  }

//   Particle and Grid data
  delt_vartype delT;
  constParticleVariable<int>     pLocalized;
  constParticleVariable<double>  pMass, pVolume,
                                 pTempPrev, pTemperature,
                                 pPlasticTemp, pPlasticTempInc,
                                 pPlasticStrain, pDamage, pPorosity, 
                                 pStrainRate,  pFailureVariable;

  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pDeformGrad, pStress,
                                 pLeftStretch, pRotation;
  constNCVariable<Vector>        gDisp;

  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<Matrix3>      pDeformGrad_new, pStress_new,
                                 pLeftStretch_new, pRotation_new;
  ParticleVariable<double>       pVolume_deformed, pPlasticStrain_new, 
                                 pDamage_new, pPorosity_new, pStrainRate_new,
                                 pPlasticTemp_new, pPlasticTempInc_new,
                                 pdTdt,  pFailureVariable_new;

 constParticleVariable<long64>   pParticleID;


//   Local variables
  Matrix3 DispGrad(0.0); // Displacement gradient
  Matrix3 DefGrad, incDefGrad, incFFt, incFFtInv, LeftStretch, Rotation; 
  Matrix3 incTotalStrain(0.0), incThermalStrain(0.0), incStrain(0.0);
  Matrix3 oldStress(0.0), devStressOld(0.0), trialStress(0.0),
          devTrialStress(0.0);
  DefGrad.Identity(); incDefGrad.Identity(); incFFt.Identity(); 
  incFFtInv.Identity(); LeftStretch.Identity(); Rotation.Identity();

  CSTi << getpid() 
     << "ComputeStressTensorImplicit: In : Matl = " << matl << " id = " 
     << matl->getDWIndex() <<  " patch = " 
     << (patches->get(0))->getID();

  //Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

//     LinearInterpolator* interpolator = new LinearInterpolator(patch);
    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get the set of particles
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // GET GLOBAL DATA 
    old_dw->get(delT,         lb->delTLabel, getLevel(patches));
    new_dw->get(gDisp,        lb->dispNewLabel, dwi, patch, gac, 1);

    old_dw->get(pMass,        lb->pMassLabel,               pset);
    old_dw->get(pVolume,      lb->pVolumeLabel,             pset);
    old_dw->get(pTemperature, lb->pTemperatureLabel,        pset);
    old_dw->get(pTempPrev,    lb->pTempPreviousLabel,       pset); 
    old_dw->get(px,           lb->pXLabel,                  pset);
    old_dw->get(psize,        lb->pSizeLabel,               pset);    
    old_dw->get(pDeformGrad,  lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress,      lb->pStressLabel,             pset);
    old_dw->get(pParticleID,  lb->pParticleIDLabel,         pset);

    // GET LOCAL DATA 
    old_dw->get(pLeftStretch,        pLeftStretchLabel,    pset);
    old_dw->get(pRotation,           pRotationLabel,       pset);
    old_dw->get(pPlasticTemp,        pPlasticTempLabel,    pset);
    old_dw->get(pPlasticTempInc,     pPlasticTempIncLabel, pset);
    old_dw->get(pPlasticStrain,      pPlasticStrainLabel,  pset);
//     old_dw->get(pDamage,             pDamageLabel,         pset);
    old_dw->get(pStrainRate,         pStrainRateLabel,     pset);
//     old_dw->get(pPorosity,           pPorosityLabel,       pset);
    old_dw->get(pLocalized,          pLocalizedLabel,      pset);
    old_dw->get(pFailureVariable, pFailureVariableLabel, pset);

    //Create and allocate arrays for storing the updated information
    // GLOBAL
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolume_deformed, 
                           lb->pVolumeDeformedLabel,              pset);
    new_dw->allocateAndPut(pdTdt, 
                           lb->pdTdtLabel_preReloc,   pset);

//     LOCAL
    new_dw->allocateAndPut(pLeftStretch_new, 
                           pLeftStretchLabel_preReloc,            pset);
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
//     new_dw->allocateAndPut(pDamage_new,      
//                            pDamageLabel_preReloc,                 pset);
//     new_dw->allocateAndPut(pPorosity_new,      
//                            pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pPlasticTemp_new,      
                           pPlasticTempLabel_preReloc,            pset);
    new_dw->allocateAndPut(pPlasticTempInc_new,      
                           pPlasticTempIncLabel_preReloc,         pset);
    new_dw->allocateAndPut(pFailureVariable_new,
                           pFailureVariableLabel_preReloc,            pset);

//     Get the back, yield and drag stresses
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);

//     Special case for rigid materials
    double totalStrainEnergy = 0.0;
    if (matl->getIsRigid()) {
      ParticleSubset::iterator iter = pset->begin(); 
      for( ; iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pLeftStretch_new[idx] = pLeftStretch[idx];
        pRotation_new[idx] = pRotation[idx];
        pStrainRate_new[idx] = pStrainRate[idx];
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
//         pDamage_new[idx] = pDamage[idx];
//         pPorosity_new[idx] = pPorosity[idx];
        pLocalized_new[idx] = pLocalized[idx];
        pPlasticTemp_new[idx] = pPlasticTemp[idx];
        pPlasticTempInc_new[idx] = 0.0;
        pFailureVariable_new[idx] = pFailureVariable[idx];

        pStress_new[idx] = Zero;
        pDeformGrad_new[idx] = One; 
        pVolume_deformed[idx] = pMass[idx]/rho_0;
        pdTdt[idx] = 0.0;
      }
      
      if (flag->d_reductionVars->accStrainEnergy ||
          flag->d_reductionVars->strainEnergy) {
        new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
      }
      delete interpolator;
      continue;
    }

    // Copy failure variable to new dw
    pFailureVariable_new.copyData(pFailureVariable);


//     Standard case for deformable materials
//     Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

//       Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

//       Calculate the displacement gradient
//       interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S);
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDeformGrad[idx]);
      computeGrad(DispGrad, ni, d_S, oodx, gDisp);

//       Compute the deformation gradient increment
      incDefGrad = DispGrad + One;
//       double Jinc = incDefGrad.Determinant();

//       Update the deformation gradient
      DefGrad = incDefGrad*pDeformGrad[idx];
      pDeformGrad_new[idx] = DefGrad;
      double J = DefGrad.Determinant();

//       Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw ParameterNotFound("**ERROR**:ViscoPlastic:Implicit", __FILE__, __LINE__);
      }

//       Calculate the current density and deformed volume
      double rho_cur = rho_0/J;
      double volold = (pMass[idx]/rho_0);
      pVolume_deformed[idx]=volold*J;

//       Compute polar decomposition of F (F = VR)
//       (**NOTE** This is being done to provide reasonable starting 
//                 values for R and V if the incremental algorithm 
//                 for the polar decomposition is used in the explicit
//                 calculations following an implicit calculation.)
      DefGrad.polarDecomposition(LeftStretch, Rotation, d_tol, false);
      pLeftStretch_new[idx] = LeftStretch;
      pRotation_new[idx] = Rotation;

//       Compute the current strain and strain rate
      incFFt = incDefGrad*incDefGrad.Transpose(); 
      incFFtInv = incFFt.Inverse();
      incTotalStrain = (One - incFFtInv)*0.5;
      pStrainRate_new[idx] = incTotalStrain.Norm()*sqrtTwoThird/delT;
      
//       Compute thermal strain
      double incT = pTemperature[idx] - pTempPrev[idx];
      incThermalStrain = One*(alpha*incT);
      incStrain = incTotalStrain - incThermalStrain;
      
//       Compute pressure and deviatoric stress at t_n and
//       the volumetric strain and deviatoric strain increments at t_n+1
      oldStress = pStress[idx];
      double pressure = oldStress.Trace()/3.0;
      Matrix3 devStressOld = oldStress - One*pressure;
      
//       Get the specific heat
      double C_p = matl->getSpecificHeat();

//       Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->strainRate = pStrainRate_new[idx];
      state->plasticStrainRate = 0.0;
      state->plasticStrain = pPlasticStrain[idx];
      state->pressure = pressure;
      state->temperature = pTemperature[idx];
      state->initialTemperature = d_initialMaterialTemperature;
      state->density = rho_cur;
      state->initialDensity = rho_0;
      state->volume = pVolume_deformed[idx];
      state->initialVolume = volold;
      state->bulkModulus = bulk ;
      state->initialBulkModulus = bulk;
      state->shearModulus = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp = Tm ;
      state->initialMeltTemp = Tm;
      state->specificHeat = C_p;
    
//       Calculate the shear modulus and the melting temperature at the
//       start of the time step and update the plasticity state
      double Tm_cur = d_plastic->computeMeltingTemp(state);
      state->meltingTemp = Tm_cur ;
      double mu_cur = d_plastic->computeShearModulus(state);
      state->shearModulus = mu_cur ;

//       Compute trial stress
      double lambda = bulk - (2.0/3.0)*mu_cur;
      trialStress = oldStress + One*(lambda*incStrain.Trace()) + incStrain*(2.0*mu_cur);
      devTrialStress = trialStress - One*(trialStress.Trace()/3.0);

//       Calculate flow stress (strain driven problem)
      double flowStress = d_plastic->computeFlowStress(idx,
                                        pStress[idx], Rotation, implicitFlag);

      //already localized; copy the old values to new
      if (pLocalized[idx]==1) {

         if (d_setStressToZero) {
             pStress_new[idx] = Zero;
         } else {
             pStress_new[idx] = pStress[idx];
         }

         pStrainRate_new[idx] =  pStrainRate[idx];
         pPlasticStrain_new[idx] = pPlasticStrain[idx];
         pLocalized_new[idx] = pLocalized[idx];
         pFailureVariable_new[idx] = pFailureVariable[idx];
         pPlasticTemp_new[idx] = pPlasticTemp[idx];
         pPlasticTempInc_new[idx] = 0.0;
         d_plastic->updateElastic(idx);
      } else {

//       elastic - Compute the deviatoric stress
      if (flowStress <= 0.0) {

//         Save the updated data
        pStress_new[idx] = trialStress;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
//         pDamage_new[idx] = pDamage[idx];
//         pPorosity_new[idx] = pPorosity[idx];
        
//         Update the internal variables
        d_plastic->updateElastic(idx);

//         Update the temperature
        pPlasticTemp_new[idx] = pPlasticTemp[idx];
        pPlasticTempInc_new[idx] = 0.0;

//         Compute stability criterion
        pLocalized_new[idx] = pLocalized[idx];
        pFailureVariable_new[idx] = pFailureVariable[idx];
      } else {
      
        // back,drag,yield stresses updated
        double epdot;
        Matrix3 stressRate;
        TangentModulusTensor Ce, Cep;
        computeElasticTangentModulus(bulk, shear, Ce);
        d_plastic->computeStressIncTangent(epdot,stressRate,Cep,delT,idx,Ce,
              incStrain,pStress[idx],implicitFlag, Rotation);

        // Calculate total stress
        pStress_new[idx] = pStress[idx]+stressRate*delT;
        
        pPlasticStrain_new[idx]=pPlasticStrain[idx]+epdot*delT;

        state->plasticStrainRate = epdot;
        state->plasticStrain = pPlasticStrain_new[idx];

//         Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = 0.9;
        double Tdot = flowStress*state->plasticStrainRate*taylorQuinney/
                      (rho_cur*C_p);
        pdTdt[idx] = Tdot;
        double dT = Tdot*delT;
        pPlasticTempInc_new[idx] = dT;
        pPlasticTemp_new[idx] = pPlasticTemp[idx] + dT; 

        // Find if the particle has localized
        pLocalized_new[idx] = pLocalized[idx];

        bool isLocalized = false;

        if (d_checkFailure && pLocalized[idx]==0)  {
            isLocalized=updateFailedParticlesAndModifyStress(DefGrad,
            pFailureVariable[idx], pLocalized[idx], pLocalized_new[idx], 
            pStress_new[idx], pParticleID[idx], pPlasticTemp_new[idx], Tm_cur);
        }
          // Rotate the stress back to the laboratory coordinates
          // Save the new data - no need for implicit
          //tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());
          //pStress_new[idx] = tensorSig;

        if (isLocalized) {
          d_plastic->updateElastic(idx);
        } 
        
      } // end if plastic or elastic
   } //end pLocalized

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        Matrix3 avgStress = (pStress_new[idx] + pStress[idx])*0.5;
        double pStrainEnergy = (incStrain(0,0)*avgStress(0,0) +
                                incStrain(1,1)*avgStress(1,1) +
                                incStrain(2,2)*avgStress(2,2) +
                                2.0*(incStrain(0,1)*avgStress(0,1) + 
                                     incStrain(0,2)*avgStress(0,2) +
                                     incStrain(1,2)*avgStress(1,2)))*
        pVolume_deformed[idx]*delT;
        totalStrainEnergy += pStrainEnergy;  
      }                

      delete state;
    } //end iterator
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }
    delete interpolator;
  } //end patch
}  //end method

void 
ViscoPlastic::addComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* patches,
                                           const bool recurse,
                                           const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForImplicitHypo(task, matlset, true, recurse,SchedParent);

  // Local stuff
  Ghost::GhostType  gnone = Ghost::None;
  if(SchedParent){
    task->requires(Task::ParentOldDW, lb->pTempPreviousLabel, matlset, gnone); 
    task->requires(Task::ParentOldDW, lb->pTemperatureLabel,  matlset, gnone);
    task->requires(Task::ParentOldDW, pPlasticStrainLabel,    matlset, gnone);
  }else{
    task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 
    task->requires(Task::OldDW, lb->pTemperatureLabel,  matlset, gnone);
    task->requires(Task::OldDW, pPlasticStrainLabel,    matlset, gnone);
  }
//   task->requires(Task::ParentOldDW, pPorosityLabel,         matlset, gnone);
//JONAHDEBUG
  d_plastic->addComputesAndRequires(task, matl, patches, recurse);
}

void 
ViscoPlastic::computeStressTensorImplicit(const PatchSubset* patches,
                                          const MPMMaterial* matl,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw,
                                          Solver* solver,
                                          const bool recurs)
{
  // Constants
  Ghost::GhostType gac = Ghost::AroundCells;
  Matrix3 One; One.Identity(); Matrix3 Zero(0.0);

//    cout << "computeStressTensor bool version\n";
   
  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double alpha = d_initialData.alpha;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();
  int implicitFlag = 1;

  // Do thermal expansion?
  if(!flag->d_doThermalExpansion){
    alpha = 0;
  }

  // Data location
  int dwi = matl->getDWIndex();
  DataWarehouse* parent_old_dw = 
    new_dw->getOtherDataWarehouse(Task::ParentOldDW);

  // Particle and Grid data
  delt_vartype delT;
  constParticleVariable<double>  pMass,
                                 pTempPrev, pTemperature,
                                 pPlasticStrain, pPorosity;

  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;  
  constParticleVariable<Matrix3> pDeformGrad, pStress;
  constNCVariable<Vector>        gDisp;

  ParticleVariable<Matrix3>      pDeformGrad_new, pStress_new;
  ParticleVariable<double>       pVolume_deformed, pPlasticStrain_new; 

  // Local variables
  Matrix3 DispGrad(0.0); // Displacement gradient
  Matrix3 DefGrad, incDefGrad, incFFt, incFFtInv;
  Matrix3 incTotalStrain(0.0), incThermalStrain(0.0), incStrain(0.0);
  Matrix3 oldStress(0.0), devStressOld(0.0), trialStress(0.0),
          devTrialStress(0.0);
  DefGrad.Identity(); incDefGrad.Identity(); incFFt.Identity(); 
  incFFtInv.Identity(); 

  // For B matrices
  double D[6][6];
  double B[6][24];
  double Bnl[3][24];
  double Kmatrix[24][24];
  int dof[24];
  double v[576];

  //CSTir << getpid() 
  //      << "ComputeStressTensorIteration: In : Matl = " << matl << " id = " 
  //      << matl->getDWIndex() <<  " patch = " 
  //      << (patches->get(0))->getID();

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get interpolation functions
//     LinearInterpolator* interpolator = new LinearInterpolator(patch);
    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);    
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    // Get patch indices for parallel solver
//    IntVector lowIndex = patch->getInteriorNodeLowIndex();
//    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);

    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get the set of particles
    ParticleSubset* pset = parent_old_dw->getParticleSubset(dwi, patch);

    // GET GLOBAL DATA 
    old_dw->get(gDisp,        lb->dispNewLabel, dwi, patch, gac, 1);
    
    // the following line - JONAH
    parent_old_dw->get(pMass,               lb->pMassLabel,   pset);
    parent_old_dw->get(delT,         lb->delTLabel, getLevel(patches));    

    parent_old_dw->get(pTempPrev,    lb->pTempPreviousLabel,       pset); 
    parent_old_dw->get(pTemperature, lb->pTemperatureLabel,        pset);
    parent_old_dw->get(px,           lb->pXLabel,                  pset);
    parent_old_dw->get(psize,        lb->pSizeLabel,               pset);    
    parent_old_dw->get(pDeformGrad,  lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pStress,      lb->pStressLabel,             pset);

    // GET LOCAL DATA 
    parent_old_dw->get(pPlasticStrain,      pPlasticStrainLabel,  pset);
//     parent_old_dw->get(pPorosity,           pPorosityLabel,       pset);

    // Create and allocate arrays for storing the updated information
    // GLOBAL
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolume_deformed, 
                           lb->pVolumeDeformedLabel,              pset);

    // LOCAL
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);

    // Special case for rigid materials
    if (matl->getIsRigid()) {
      ParticleSubset::iterator iter = pset->begin(); 
      for( ; iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];

        pStress_new[idx] = Zero;
        pDeformGrad_new[idx] = One; 
        pVolume_deformed[idx] = pMass[idx]/rho_0;
      }
      delete interpolator;
      continue;
    }

    // Standard case for deformable materials
    
    //following two lines added by JONAH
    d_plastic->getInternalVars(pset, parent_old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);
    
    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      //CSTir << " patch = " << patch << " particle = " << idx << endl;

      // Calculate the displacement gradient
//       interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S);
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDeformGrad[idx]);      
      computeGradAndBmats(DispGrad,ni,d_S, oodx, gDisp, l2g,B, Bnl, dof);

      // Compute the deformation gradient increment
      incDefGrad = DispGrad + One;
      //double Jinc = incDefGrad.Determinant();

      //CSTir << " particle = " << idx << " Jinc = " << Jinc << endl;

      // Update the deformation gradient
      DefGrad = incDefGrad*pDeformGrad[idx];
      pDeformGrad_new[idx] = DefGrad;
      double J = DefGrad.Determinant();

      // Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw ParameterNotFound("**ERROR**:ViscoPlastic:Implicit", __FILE__, __LINE__);
      }

      //CSTir << " particle = " << idx << " J = " << J << endl;

//       cout << "rho_0= " << rho_0 << " J= " << J << " idx= " << idx << "\n";
//       cout << "pMass[idx]= " << pMass[idx] << " \n";
      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;
      double volold = (pMass[idx]/rho_0);
      pVolume_deformed[idx] = volold*J;

      //CSTir << " particle = " << idx << " rho = " << rho_cur 
      //      << " vol_new = " << pVolume_deformed[idx] << endl;

      // Compute the current strain and strain rate
      incFFt = incDefGrad*incDefGrad.Transpose(); 
      incFFtInv = incFFt.Inverse();
      incTotalStrain = (One - incFFtInv)*0.5;
      
      double pStrainRate_new = incTotalStrain.Norm()*sqrt(2.0/3.0)/delT;

      // Compute thermal strain
      double incT = pTemperature[idx] - pTempPrev[idx];
      incThermalStrain = One*(alpha*incT);
      incStrain = incTotalStrain - incThermalStrain;
      
      //CSTir << " particle = " << idx << " incStrain = " << incStrain  << endl;

      // Compute pressure and deviatoric stress at t_n and
      // the volumetric strain and deviatoric strain increments at t_n+1
      oldStress = pStress[idx];
      double pressure = oldStress.Trace()/3.0;
      //Matrix3 devStressOld = oldStress - One*pressure;
      
      // Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->strainRate = pStrainRate_new;
      state->plasticStrainRate = 0.0;
      state->plasticStrain = pPlasticStrain[idx];
      state->pressure = pressure;
      state->temperature = pTemperature[idx];
      state->initialTemperature = d_initialMaterialTemperature;
      state->density = rho_cur;
      state->initialDensity = rho_0;
      state->volume = pVolume_deformed[idx];
      state->initialVolume = volold;
      state->bulkModulus = bulk ;
      state->initialBulkModulus = bulk;
      state->shearModulus = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp = Tm ;
      state->initialMeltTemp = Tm;
    
      // Calculate the shear modulus and the melting temperature at the
      // start of the time step and update the plasticity state
      double Tm_cur = d_plastic->computeMeltingTemp(state);
      state->meltingTemp = Tm_cur ;
      double mu_cur = d_plastic->computeShearModulus(state);
      state->shearModulus = mu_cur ;

      // Compute trial stress
      double lambda = bulk - (2.0/3.0)*mu_cur;
      trialStress = oldStress + One*(lambda*incStrain.Trace()) + incStrain*(2.0*mu_cur);
      devTrialStress = trialStress - One*(trialStress.Trace()/3.0);
      
      // Calculate the equivalent stress
//       double equivStress = sqrt((devTrialStress.NormSquared())*1.5);

      // Calculate flow stress (strain driven problem) - Zero is fake Rotation
      double flowStress = d_plastic->computeFlowStress(idx,
                                 pStress[idx], Zero, implicitFlag);

      // Compute the deviatoric stress
      if (flowStress <= 0.0) {

        // Save the updated data
        pStress_new[idx] = trialStress;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        
        computeElasticTangentModulus(bulk, shear, D);
        
        //CSTir << " Elastic particle = " << idx 
        //      << " stress = " << pStress_new[idx] << endl;  

      } else {

        //plastic strain rate, plastic strain; back,drag,yield stresses updated
        double epdot;
        Matrix3 stressRate;
        TangentModulusTensor Ce, Cep;
        computeElasticTangentModulus(bulk, shear, Ce);
        
        d_plastic->computeStressIncTangent(epdot,stressRate,Cep,delT,idx,Ce,
              incStrain,pStress[idx],implicitFlag,Zero);
        convertToVoigtForm(Ce,D);

        // Calculate total stress
        pStress_new[idx] = pStress[idx]+stressRate*delT;
        pPlasticStrain_new[idx]=pPlasticStrain[idx]+epdot*delT;

        state->plasticStrainRate = epdot;
        state->plasticStrain = pPlasticStrain_new[idx];


//         computeEPlasticTangentModulus(bulk, shear, delGamma, normTrialS,
//                                       idx, nn, state, D);

        //CSTir << " Plastic particle = " << idx 
        //      << " stress = " << pStress_new[idx] << endl;  
      }

      // Compute K matrix = Kmat + Kgeo
      computeStiffnessMatrix(B, Bnl, D, pStress[idx], volold, 
                             pVolume_deformed[idx], Kmatrix);

      //CSTir << " particle = " << idx << " Computed K matrix " << endl;

      // Assemble into global K matrix
      for (int ii = 0; ii < 24; ii++){
        for (int jj = 0; jj < 24; jj++){
          v[24*ii+jj] = Kmatrix[ii][jj];
        }
      }
      solver->fillMatrix(24,dof,24,dof,v);

      //CSTir << " particle = " << idx << " Sent to solver " << endl;

      delete state;
    }
    delete interpolator;
  }
//  solver->flushMatrix();
}


/*! Compute the elastic tangent modulus tensor for isotropic
    materials
    Assume: [stress] = [s11 s22 s33 s23 s31 s12]
            [strain] = [e11 e22 e33 2e23 2e31 2e12] 
*/
void 
ViscoPlastic::computeElasticTangentModulus(const double& K,
                                                 const double& mu,
                                                 double Ce[6][6])
{
  // Form the elastic tangent modulus tensor
  double twomu = 2.0*mu;
  double lambda = K - (twomu/3.0);
  double lambda_twomu = lambda + twomu;

  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = 0; jj < 6; ++jj) {
      Ce[ii][jj] = 0.0;
    }
  }
  Ce[0][0] = lambda_twomu;
  Ce[1][1] = lambda_twomu;
  Ce[2][2] = lambda_twomu;
  Ce[3][3] = mu;
  Ce[4][4] = mu;
  Ce[5][5] = mu;
  Ce[0][1] = lambda;
  Ce[0][2] = lambda;
  Ce[1][2] = lambda;
  for (int ii = 1; ii < 3; ++ii) {
    for (int jj = 0; jj < ii; ++jj) {
      Ce[ii][jj] = Ce[jj][ii];
    }
  }
}

/*! Compute the elastic-plastic tangent modulus tensor for isotropic
    materials for use in the implicit stress update
    Assume: [stress] = [s11 s22 s33 s23 s31 s12]
            [strain] = [e11 e22 e33 2e23 2e31 2e12] 
    Uses alogorithm for small strain plasticity (Simo 1998, p.124)
*/
void 
ViscoPlastic::computeEPlasticTangentModulus(const double& K,
                                                  const double& mu,
                                                  const double& delGamma,
                                                  const double& normTrialS,
                                                  const particleIndex idx,
                                                  const Matrix3& n,
                                                  PlasticityState* state,
                                                  double Cep[6][6])
{

}

/*! Compute K matrix */
void 
ViscoPlastic::computeStiffnessMatrix(const double B[6][24],
                                           const double Bnl[3][24],
                                           const double D[6][6],
                                           const Matrix3& sig,
                                           const double& vol_old,
                                           const double& vol_new,
                                           double Kmatrix[24][24])
{
  // Kmat = B.transpose()*D*B*volold
  double Kmat[24][24];
  BtDB(B, D, Kmat);

  // Kgeo = Bnl.transpose*sig*Bnl*volnew;
  double Kgeo[24][24];
  BnlTSigBnl(sig, Bnl, Kgeo);

  for(int ii = 0;ii<24;ii++){
    for(int jj = 0;jj<24;jj++){
      Kmatrix[ii][jj] =  Kmat[ii][jj]*vol_old + Kgeo[ii][jj]*vol_new;
    }
  }
}

void 
ViscoPlastic::BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                               double Kgeo[24][24]) const
{
  double t1, t10, t11, t12, t13, t14, t15, t16, t17;
  double t18, t19, t2, t20, t21, t22, t23, t24, t25;
  double t26, t27, t28, t29, t3, t30, t31, t32, t33;
  double t34, t35, t36, t37, t38, t39, t4, t40, t41;
  double t42, t43, t44, t45, t46, t47, t48, t49, t5;
  double t50, t51, t52, t53, t54, t55, t56, t57, t58;
  double t59, t6, t60, t61, t62, t63, t64, t65, t66;
  double t67, t68, t69, t7, t70, t71, t72, t73, t74;
  double t75, t77, t78, t8, t81, t85, t88, t9, t90;
  double t79, t82, t83, t86, t87, t89;

  t1 = Bnl[0][0]*sig(0,0);
  t4 = Bnl[0][0]*sig(0,0);
  t2 = Bnl[0][0]*sig(0,1);
  t3 = Bnl[0][0]*sig(0,2);
  t5 = Bnl[1][1]*sig(1,1);
  t8 = Bnl[1][1]*sig(1,1);
  t6 = Bnl[1][1]*sig(1,2);
  t7 = Bnl[1][1]*sig(0,1);
  t9 = Bnl[2][2]*sig(2,2);
  t12 = Bnl[2][2]*sig(2,2);
  t10 = Bnl[2][2]*sig(0,2);
  t11 = Bnl[2][2]*sig(1,2);
  t13 = Bnl[0][3]*sig(0,0);
  t16 = Bnl[0][3]*sig(0,0);
  t14 = Bnl[0][3]*sig(0,1);
  t15 = Bnl[0][3]*sig(0,2);
  t17 = Bnl[1][4]*sig(1,1);
  t20 = Bnl[1][4]*sig(1,1);
  t18 = Bnl[1][4]*sig(1,2);
  t19 = Bnl[1][4]*sig(0,1);
  t21 = Bnl[2][5]*sig(2,2);
  t22 = Bnl[2][5]*sig(0,2);
  t23 = Bnl[2][5]*sig(1,2);
  t24 = Bnl[2][5]*sig(2,2);
  t25 = Bnl[0][6]*sig(0,0);
  t26 = Bnl[0][6]*sig(0,1);
  t27 = Bnl[0][6]*sig(0,2);
  t28 = Bnl[0][6]*sig(0,0);
  t29 = Bnl[1][7]*sig(1,1);
  t30 = Bnl[1][7]*sig(1,2);
  t31 = Bnl[1][7]*sig(0,1);
  t32 = Bnl[1][7]*sig(1,1);
  t33 = Bnl[2][8]*sig(2,2);
  t34 = Bnl[2][8]*sig(0,2);
  t35 = Bnl[2][8]*sig(1,2);
  t36 = Bnl[2][8]*sig(2,2);
  t37 = Bnl[0][9]*sig(0,0);
  t38 = Bnl[0][9]*sig(0,1);
  t39 = Bnl[0][9]*sig(0,2);
  t40 = Bnl[0][9]*sig(0,0);
  t41 = Bnl[1][10]*sig(1,1);
  t42 = Bnl[1][10]*sig(1,2);
  t43 = Bnl[1][10]*sig(0,1);
  t44 = Bnl[1][10]*sig(1,1);
  t45 = Bnl[2][11]*sig(2,2);
  t46 = Bnl[2][11]*sig(0,2);
  t47 = Bnl[2][11]*sig(1,2);
  t48 = Bnl[2][11]*sig(2,2);
  t49 = Bnl[0][12]*sig(0,0);
  t50 = Bnl[0][12]*sig(0,1);
  t51 = Bnl[0][12]*sig(0,2);
  t52 = Bnl[0][12]*sig(0,0);
  t53 = Bnl[1][13]*sig(1,1);
  t54 = Bnl[1][13]*sig(1,2);
  t55 = Bnl[1][13]*sig(0,1);
  t56 = Bnl[1][13]*sig(1,1);
  t57 = Bnl[2][14]*sig(2,2);
  t58 = Bnl[2][14]*sig(0,2);
  t59 = Bnl[2][14]*sig(1,2);
  t60 = Bnl[2][14]*sig(2,2);
  t61 = Bnl[0][15]*sig(0,0);
  t62 = Bnl[0][15]*sig(0,1);
  t63 = Bnl[0][15]*sig(0,2);
  t64 = Bnl[0][15]*sig(0,0);
  t65 = Bnl[1][16]*sig(1,1);
  t66 = Bnl[1][16]*sig(1,2);
  t67 = Bnl[1][16]*sig(0,1);
  t68 = Bnl[1][16]*sig(1,1);
  t69 = Bnl[2][17]*sig(2,2);
  t70 = Bnl[2][17]*sig(0,2);
  t71 = Bnl[2][17]*sig(1,2);
  t72 = Bnl[2][17]*sig(2,2);
  t73 = Bnl[0][18]*sig(0,0);
  t74 = Bnl[0][18]*sig(0,1);
  t75 = Bnl[0][18]*sig(0,2);
  t77 = Bnl[1][19]*sig(1,1);
  t78 = Bnl[1][19]*sig(1,2);
  t79 = Bnl[1][19]*sig(0,1);
  t81 = Bnl[2][20]*sig(2,2);
  t82 = Bnl[2][20]*sig(0,2);
  t83 = Bnl[2][20]*sig(1,2);
  t85 = Bnl[0][21]*sig(0,0);
  t86 = Bnl[0][21]*sig(0,1);
  t87 = Bnl[0][21]*sig(0,2);
  t88 = Bnl[1][22]*sig(1,1);
  t89 = Bnl[1][22]*sig(1,2);
  t90 = Bnl[2][23]*sig(2,2);

  Kgeo[0][0] = t1*Bnl[0][0];
  Kgeo[0][1] = t2*Bnl[1][1];
  Kgeo[0][2] = t3*Bnl[2][2];
  Kgeo[0][3] = t4*Bnl[0][3];
  Kgeo[0][4] = t2*Bnl[1][4];
  Kgeo[0][5] = t3*Bnl[2][5];
  Kgeo[0][6] = t4*Bnl[0][6];
  Kgeo[0][7] = t2*Bnl[1][7];
  Kgeo[0][8] = t3*Bnl[2][8];
  Kgeo[0][9] = t4*Bnl[0][9];
  Kgeo[0][10] = t2*Bnl[1][10];
  Kgeo[0][11] = t3*Bnl[2][11];
  Kgeo[0][12] = t4*Bnl[0][12];
  Kgeo[0][13] = t2*Bnl[1][13];
  Kgeo[0][14] = t3*Bnl[2][14];
  Kgeo[0][15] = t4*Bnl[0][15];
  Kgeo[0][16] = t2*Bnl[1][16];
  Kgeo[0][17] = t3*Bnl[2][17];
  Kgeo[0][18] = t4*Bnl[0][18];
  Kgeo[0][19] = t2*Bnl[1][19];
  Kgeo[0][20] = t3*Bnl[2][20];
  Kgeo[0][21] = t4*Bnl[0][21];
  Kgeo[0][22] = t2*Bnl[1][22];
  Kgeo[0][23] = t3*Bnl[2][23];
  Kgeo[1][0] = Kgeo[0][1];
  Kgeo[1][1] = t5*Bnl[1][1];
  Kgeo[1][2] = t6*Bnl[2][2];
  Kgeo[1][3] = t7*Bnl[0][3];
  Kgeo[1][4] = Bnl[1][4]*t8;
  Kgeo[1][5] = t6*Bnl[2][5];
  Kgeo[1][6] = t7*Bnl[0][6];
  Kgeo[1][7] = Bnl[1][7]*t8;
  Kgeo[1][8] = t6*Bnl[2][8];
  Kgeo[1][9] = t7*Bnl[0][9];
  Kgeo[1][10] = Bnl[1][10]*t8;
  Kgeo[1][11] = t6*Bnl[2][11];
  Kgeo[1][12] = t7*Bnl[0][12];
  Kgeo[1][13] = Bnl[1][13]*t8;
  Kgeo[1][14] = t6*Bnl[2][14];
  Kgeo[1][15] = t7*Bnl[0][15];
  Kgeo[1][16] = Bnl[1][16]*t8;
  Kgeo[1][17] = t6*Bnl[2][17];
  Kgeo[1][18] = t7*Bnl[0][18];
  Kgeo[1][19] = Bnl[1][19]*t8;
  Kgeo[1][20] = t6*Bnl[2][20];
  Kgeo[1][21] = t7*Bnl[0][21];
  Kgeo[1][22] = Bnl[1][22]*t8;
  Kgeo[1][23] = t6*Bnl[2][23];
  Kgeo[2][0] = Kgeo[0][2];
  Kgeo[2][1] = Kgeo[1][2];
  Kgeo[2][2] = t9*Bnl[2][2];
  Kgeo[2][3] = t10*Bnl[0][3];
  Kgeo[2][4] = Bnl[1][4]*t11;
  Kgeo[2][5] = t12*Bnl[2][5];
  Kgeo[2][6] = t10*Bnl[0][6];
  Kgeo[2][7] = Bnl[1][7]*t11;
  Kgeo[2][8] = t12*Bnl[2][8];
  Kgeo[2][9] = t10*Bnl[0][9];
  Kgeo[2][10] = Bnl[1][10]*t11;
  Kgeo[2][11] = t12*Bnl[2][11];
  Kgeo[2][12] = t10*Bnl[0][12];
  Kgeo[2][13] = Bnl[1][13]*t11;
  Kgeo[2][14] = t12*Bnl[2][14];
  Kgeo[2][15] = t10*Bnl[0][15];
  Kgeo[2][16] = Bnl[1][16]*t11;
  Kgeo[2][17] = t12*Bnl[2][17];
  Kgeo[2][18] = t10*Bnl[0][18];
  Kgeo[2][19] = t11*Bnl[1][19];
  Kgeo[2][20] = t12*Bnl[2][20];
  Kgeo[2][21] = t10*Bnl[0][21];
  Kgeo[2][22] = t11*Bnl[1][22];
  Kgeo[2][23] = t12*Bnl[2][23];
  Kgeo[3][0] = Kgeo[0][3];
  Kgeo[3][1] = Kgeo[1][3];
  Kgeo[3][2] = Kgeo[2][3];
  Kgeo[3][3] = t13*Bnl[0][3];
  Kgeo[3][4] = t14*Bnl[1][4];
  Kgeo[3][5] = Bnl[2][5]*t15;
  Kgeo[3][6] = t16*Bnl[0][6];
  Kgeo[3][7] = t14*Bnl[1][7];
  Kgeo[3][8] = Bnl[2][8]*t15;
  Kgeo[3][9] = t16*Bnl[0][9];
  Kgeo[3][10] = t14*Bnl[1][10];
  Kgeo[3][11] = Bnl[2][11]*t15;
  Kgeo[3][12] = t16*Bnl[0][12];
  Kgeo[3][13] = t14*Bnl[1][13];
  Kgeo[3][14] = Bnl[2][14]*t15;
  Kgeo[3][15] = t16*Bnl[0][15];
  Kgeo[3][16] = t14*Bnl[1][16];
  Kgeo[3][17] = Bnl[2][17]*t15;
  Kgeo[3][18] = t16*Bnl[0][18];
  Kgeo[3][19] = t14*Bnl[1][19];
  Kgeo[3][20] = Bnl[2][20]*t15;
  Kgeo[3][21] = t16*Bnl[0][21];
  Kgeo[3][22] = t14*Bnl[1][22];
  Kgeo[3][23] = Bnl[2][23]*t15;
  Kgeo[4][0] = Kgeo[0][4];
  Kgeo[4][1] = Kgeo[1][4];
  Kgeo[4][2] = Kgeo[2][4];
  Kgeo[4][3] = Kgeo[3][4];
  Kgeo[4][4] = t17*Bnl[1][4];
  Kgeo[4][5] = t18*Bnl[2][5];
  Kgeo[4][6] = t19*Bnl[0][6];
  Kgeo[4][7] = t20*Bnl[1][7];
  Kgeo[4][8] = t18*Bnl[2][8];
  Kgeo[4][9] = t19*Bnl[0][9];
  Kgeo[4][10] = t20*Bnl[1][10];
  Kgeo[4][11] = t18*Bnl[2][11];
  Kgeo[4][12] = t19*Bnl[0][12];
  Kgeo[4][13] = t20*Bnl[1][13];
  Kgeo[4][14] = t18*Bnl[2][14];
  Kgeo[4][15] = t19*Bnl[0][15];
  Kgeo[4][16] = t20*Bnl[1][16];
  Kgeo[4][17] = t18*Bnl[2][17];
  Kgeo[4][18] = t19*Bnl[0][18];
  Kgeo[4][19] = t20*Bnl[1][19];
  Kgeo[4][20] = t18*Bnl[2][20];
  Kgeo[4][21] = t19*Bnl[0][21];
  Kgeo[4][22] = t20*Bnl[1][22];
  Kgeo[4][23] = t18*Bnl[2][23];
  Kgeo[5][0] = Kgeo[0][5];
  Kgeo[5][1] = Kgeo[1][5];
  Kgeo[5][2] = Kgeo[2][5];
  Kgeo[5][3] = Kgeo[3][5];
  Kgeo[5][4] = Kgeo[4][5];
  Kgeo[5][5] = t21*Bnl[2][5];
  Kgeo[5][6] = t22*Bnl[0][6];
  Kgeo[5][7] = t23*Bnl[1][7];
  Kgeo[5][8] = t24*Bnl[2][8];
  Kgeo[5][9] = t22*Bnl[0][9];
  Kgeo[5][10] = t23*Bnl[1][10];
  Kgeo[5][11] = t24*Bnl[2][11];
  Kgeo[5][12] = t22*Bnl[0][12];
  Kgeo[5][13] = t23*Bnl[1][13];
  Kgeo[5][14] = t24*Bnl[2][14];
  Kgeo[5][15] = t22*Bnl[0][15];
  Kgeo[5][16] = t23*Bnl[1][16];
  Kgeo[5][17] = t24*Bnl[2][17];
  Kgeo[5][18] = t22*Bnl[0][18];
  Kgeo[5][19] = t23*Bnl[1][19];
  Kgeo[5][20] = t24*Bnl[2][20];
  Kgeo[5][21] = t22*Bnl[0][21];
  Kgeo[5][22] = t23*Bnl[1][22];
  Kgeo[5][23] = t24*Bnl[2][23];
  Kgeo[6][0] = Kgeo[0][6];
  Kgeo[6][1] = Kgeo[1][6];
  Kgeo[6][2] = Kgeo[2][6];
  Kgeo[6][3] = Kgeo[3][6];
  Kgeo[6][4] = Kgeo[4][6];
  Kgeo[6][5] = Kgeo[5][6];
  Kgeo[6][6] = t25*Bnl[0][6];
  Kgeo[6][7] = t26*Bnl[1][7];
  Kgeo[6][8] = t27*Bnl[2][8];
  Kgeo[6][9] = t28*Bnl[0][9];
  Kgeo[6][10] = t26*Bnl[1][10];
  Kgeo[6][11] = t27*Bnl[2][11];
  Kgeo[6][12] = t28*Bnl[0][12];
  Kgeo[6][13] = t26*Bnl[1][13];
  Kgeo[6][14] = t27*Bnl[2][14];
  Kgeo[6][15] = t28*Bnl[0][15];
  Kgeo[6][16] = t26*Bnl[1][16];
  Kgeo[6][17] = t27*Bnl[2][17];
  Kgeo[6][18] = t28*Bnl[0][18];
  Kgeo[6][19] = t26*Bnl[1][19];
  Kgeo[6][20] = t27*Bnl[2][20];
  Kgeo[6][21] = t28*Bnl[0][21];
  Kgeo[6][22] = t26*Bnl[1][22];
  Kgeo[6][23] = t27*Bnl[2][23];
  Kgeo[7][0] = Kgeo[0][7];
  Kgeo[7][1] = Kgeo[1][7];
  Kgeo[7][2] = Kgeo[2][7];
  Kgeo[7][3] = Kgeo[3][7];
  Kgeo[7][4] = Kgeo[4][7];
  Kgeo[7][5] = Kgeo[5][7];
  Kgeo[7][6] = Kgeo[6][7];
  Kgeo[7][7] = t29*Bnl[1][7];
  Kgeo[7][8] = t30*Bnl[2][8];
  Kgeo[7][9] = t31*Bnl[0][9];
  Kgeo[7][10] = t32*Bnl[1][10];
  Kgeo[7][11] = t30*Bnl[2][11];
  Kgeo[7][12] = t31*Bnl[0][12];
  Kgeo[7][13] = t32*Bnl[1][13];
  Kgeo[7][14] = t30*Bnl[2][14];
  Kgeo[7][15] = t31*Bnl[0][15];
  Kgeo[7][16] = t32*Bnl[1][16];
  Kgeo[7][17] = t30*Bnl[2][17];
  Kgeo[7][18] = t31*Bnl[0][18];
  Kgeo[7][19] = t32*Bnl[1][19];
  Kgeo[7][20] = t30*Bnl[2][20];
  Kgeo[7][21] = t31*Bnl[0][21];
  Kgeo[7][22] = t32*Bnl[1][22];
  Kgeo[7][23] = t30*Bnl[2][23];
  Kgeo[8][0] = Kgeo[0][8];
  Kgeo[8][1] = Kgeo[1][8];
  Kgeo[8][2] = Kgeo[2][8];
  Kgeo[8][3] = Kgeo[3][8];
  Kgeo[8][4] = Kgeo[4][8];
  Kgeo[8][5] = Kgeo[5][8];
  Kgeo[8][6] = Kgeo[6][8];
  Kgeo[8][7] = Kgeo[7][8];
  Kgeo[8][8] = t33*Bnl[2][8];
  Kgeo[8][9] = t34*Bnl[0][9];
  Kgeo[8][10] = t35*Bnl[1][10];
  Kgeo[8][11] = t36*Bnl[2][11];
  Kgeo[8][12] = t34*Bnl[0][12];
  Kgeo[8][13] = t35*Bnl[1][13];
  Kgeo[8][14] = t36*Bnl[2][14];
  Kgeo[8][15] = t34*Bnl[0][15];
  Kgeo[8][16] = t35*Bnl[1][16];
  Kgeo[8][17] = t36*Bnl[2][17];
  Kgeo[8][18] = t34*Bnl[0][18];
  Kgeo[8][19] = t35*Bnl[1][19];
  Kgeo[8][20] = t36*Bnl[2][20];
  Kgeo[8][21] = t34*Bnl[0][21];
  Kgeo[8][22] = t35*Bnl[1][22];
  Kgeo[8][23] = t36*Bnl[2][23];
  Kgeo[9][0] = Kgeo[0][9];
  Kgeo[9][1] = Kgeo[1][9];
  Kgeo[9][2] = Kgeo[2][9];
  Kgeo[9][3] = Kgeo[3][9];
  Kgeo[9][4] = Kgeo[4][9];
  Kgeo[9][5] = Kgeo[5][9];
  Kgeo[9][6] = Kgeo[6][9];
  Kgeo[9][7] = Kgeo[7][9];
  Kgeo[9][8] = Kgeo[8][9];
  Kgeo[9][9] = t37*Bnl[0][9];
  Kgeo[9][10] = t38*Bnl[1][10];
  Kgeo[9][11] = t39*Bnl[2][11];
  Kgeo[9][12] = t40*Bnl[0][12];
  Kgeo[9][13] = t38*Bnl[1][13];
  Kgeo[9][14] = t39*Bnl[2][14];
  Kgeo[9][15] = t40*Bnl[0][15];
  Kgeo[9][16] = t38*Bnl[1][16];
  Kgeo[9][17] = t39*Bnl[2][17];
  Kgeo[9][18] = t40*Bnl[0][18];
  Kgeo[9][19] = t38*Bnl[1][19];
  Kgeo[9][20] = t39*Bnl[2][20];
  Kgeo[9][21] = t40*Bnl[0][21];
  Kgeo[9][22] = t38*Bnl[1][22];
  Kgeo[9][23] = t39*Bnl[2][23];
  Kgeo[10][0] = Kgeo[0][10];
  Kgeo[10][1] = Kgeo[1][10];
  Kgeo[10][2] = Kgeo[2][10];
  Kgeo[10][3] = Kgeo[3][10];
  Kgeo[10][4] = Kgeo[4][10];
  Kgeo[10][5] = Kgeo[5][10];
  Kgeo[10][6] = Kgeo[6][10];
  Kgeo[10][7] = Kgeo[7][10];
  Kgeo[10][8] = Kgeo[8][10];
  Kgeo[10][9] = Kgeo[9][10];
  Kgeo[10][10] = t41*Bnl[1][10];
  Kgeo[10][11] = t42*Bnl[2][11];
  Kgeo[10][12] = t43*Bnl[0][12];
  Kgeo[10][13] = t44*Bnl[1][13];
  Kgeo[10][14] = t42*Bnl[2][14];
  Kgeo[10][15] = t43*Bnl[0][15];
  Kgeo[10][16] = t44*Bnl[1][16];
  Kgeo[10][17] = t42*Bnl[2][17];
  Kgeo[10][18] = t43*Bnl[0][18];
  Kgeo[10][19] = t44*Bnl[1][19];
  Kgeo[10][20] = t42*Bnl[2][20];
  Kgeo[10][21] = t43*Bnl[0][21];
  Kgeo[10][22] = t44*Bnl[1][22];
  Kgeo[10][23] = t42*Bnl[2][23];
  Kgeo[11][0] = Kgeo[0][11];
  Kgeo[11][1] = Kgeo[1][11];
  Kgeo[11][2] = Kgeo[2][11];
  Kgeo[11][3] = Kgeo[3][11];
  Kgeo[11][4] = Kgeo[4][11];
  Kgeo[11][5] = Kgeo[5][11];
  Kgeo[11][6] = Kgeo[6][11];
  Kgeo[11][7] = Kgeo[7][11];
  Kgeo[11][8] = Kgeo[8][11];
  Kgeo[11][9] = Kgeo[9][11];
  Kgeo[11][10] = Kgeo[10][11];
  Kgeo[11][11] = t45*Bnl[2][11];
  Kgeo[11][12] = t46*Bnl[0][12];
  Kgeo[11][13] = t47*Bnl[1][13];
  Kgeo[11][14] = t48*Bnl[2][14];
  Kgeo[11][15] = t46*Bnl[0][15];
  Kgeo[11][16] = t47*Bnl[1][16];
  Kgeo[11][17] = t48*Bnl[2][17];
  Kgeo[11][18] = t46*Bnl[0][18];
  Kgeo[11][19] = t47*Bnl[1][19];
  Kgeo[11][20] = t48*Bnl[2][20];
  Kgeo[11][21] = t46*Bnl[0][21];
  Kgeo[11][22] = t47*Bnl[1][22];
  Kgeo[11][23] = t48*Bnl[2][23];
  Kgeo[12][0] = Kgeo[0][12];
  Kgeo[12][1] = Kgeo[1][12];
  Kgeo[12][2] = Kgeo[2][12];
  Kgeo[12][3] = Kgeo[3][12];
  Kgeo[12][4] = Kgeo[4][12];
  Kgeo[12][5] = Kgeo[5][12];
  Kgeo[12][6] = Kgeo[6][12];
  Kgeo[12][7] = Kgeo[7][12];
  Kgeo[12][8] = Kgeo[8][12];
  Kgeo[12][9] = Kgeo[9][12];
  Kgeo[12][10] = Kgeo[10][12];
  Kgeo[12][11] = Kgeo[11][12];
  Kgeo[12][12] = t49*Bnl[0][12];
  Kgeo[12][13] = t50*Bnl[1][13];
  Kgeo[12][14] = t51*Bnl[2][14];
  Kgeo[12][15] = t52*Bnl[0][15];
  Kgeo[12][16] = t50*Bnl[1][16];
  Kgeo[12][17] = t51*Bnl[2][17];
  Kgeo[12][18] = t52*Bnl[0][18];
  Kgeo[12][19] = t50*Bnl[1][19];
  Kgeo[12][20] = t51*Bnl[2][20];
  Kgeo[12][21] = t52*Bnl[0][21];
  Kgeo[12][22] = t50*Bnl[1][22];
  Kgeo[12][23] = t51*Bnl[2][23];
  Kgeo[13][0] = Kgeo[0][13];
  Kgeo[13][1] = Kgeo[1][13];
  Kgeo[13][2] = Kgeo[2][13];
  Kgeo[13][3] = Kgeo[3][13];
  Kgeo[13][4] = Kgeo[4][13];
  Kgeo[13][5] = Kgeo[5][13];
  Kgeo[13][6] = Kgeo[6][13];
  Kgeo[13][7] = Kgeo[7][13];
  Kgeo[13][8] = Kgeo[8][13];
  Kgeo[13][9] = Kgeo[9][13];
  Kgeo[13][10] = Kgeo[10][13];
  Kgeo[13][11] = Kgeo[11][13];
  Kgeo[13][12] = Kgeo[12][13];
  Kgeo[13][13] = t53*Bnl[1][13];
  Kgeo[13][14] = t54*Bnl[2][14];
  Kgeo[13][15] = t55*Bnl[0][15];
  Kgeo[13][16] = t56*Bnl[1][16];
  Kgeo[13][17] = t54*Bnl[2][17];
  Kgeo[13][18] = t55*Bnl[0][18];
  Kgeo[13][19] = t56*Bnl[1][19];
  Kgeo[13][20] = t54*Bnl[2][20];
  Kgeo[13][21] = t55*Bnl[0][21];
  Kgeo[13][22] = t56*Bnl[1][22];
  Kgeo[13][23] = t54*Bnl[2][23];
  Kgeo[14][0] = Kgeo[0][14];
  Kgeo[14][1] = Kgeo[1][14];
  Kgeo[14][2] = Kgeo[2][14];
  Kgeo[14][3] = Kgeo[3][14];
  Kgeo[14][4] = Kgeo[4][14];
  Kgeo[14][5] = Kgeo[5][14];
  Kgeo[14][6] = Kgeo[6][14];
  Kgeo[14][7] = Kgeo[7][14];
  Kgeo[14][8] = Kgeo[8][14];
  Kgeo[14][9] = Kgeo[9][14];
  Kgeo[14][10] = Kgeo[10][14];
  Kgeo[14][11] = Kgeo[11][14];
  Kgeo[14][12] = Kgeo[12][14];
  Kgeo[14][13] = Kgeo[13][14];
  Kgeo[14][14] = t57*Bnl[2][14];
  Kgeo[14][15] = t58*Bnl[0][15];
  Kgeo[14][16] = t59*Bnl[1][16];
  Kgeo[14][17] = t60*Bnl[2][17];
  Kgeo[14][18] = t58*Bnl[0][18];
  Kgeo[14][19] = t59*Bnl[1][19];
  Kgeo[14][20] = t60*Bnl[2][20];
  Kgeo[14][21] = t58*Bnl[0][21];
  Kgeo[14][22] = t59*Bnl[1][22];
  Kgeo[14][23] = t60*Bnl[2][23];
  Kgeo[15][0] = Kgeo[0][15];
  Kgeo[15][1] = Kgeo[1][15];
  Kgeo[15][2] = Kgeo[2][15];
  Kgeo[15][3] = Kgeo[3][15];
  Kgeo[15][4] = Kgeo[4][15];
  Kgeo[15][5] = Kgeo[5][15];
  Kgeo[15][6] = Kgeo[6][15];
  Kgeo[15][7] = Kgeo[7][15];
  Kgeo[15][8] = Kgeo[8][15];
  Kgeo[15][9] = Kgeo[9][15];
  Kgeo[15][10] = Kgeo[10][15];
  Kgeo[15][11] = Kgeo[11][15];
  Kgeo[15][12] = Kgeo[12][15];
  Kgeo[15][13] = Kgeo[13][15];
  Kgeo[15][14] = Kgeo[14][15];
  Kgeo[15][15] = t61*Bnl[0][15];
  Kgeo[15][16] = t62*Bnl[1][16];
  Kgeo[15][17] = t63*Bnl[2][17];
  Kgeo[15][18] = t64*Bnl[0][18];
  Kgeo[15][19] = t62*Bnl[1][19];
  Kgeo[15][20] = t63*Bnl[2][20];
  Kgeo[15][21] = t64*Bnl[0][21];
  Kgeo[15][22] = t62*Bnl[1][22];
  Kgeo[15][23] = t63*Bnl[2][23];
  Kgeo[16][0] = Kgeo[0][16];
  Kgeo[16][1] = Kgeo[1][16];
  Kgeo[16][2] = Kgeo[2][16];
  Kgeo[16][3] = Kgeo[3][16];
  Kgeo[16][4] = Kgeo[4][16];
  Kgeo[16][5] = Kgeo[5][16];
  Kgeo[16][6] = Kgeo[6][16];
  Kgeo[16][7] = Kgeo[7][16];
  Kgeo[16][8] = Kgeo[8][16];
  Kgeo[16][9] = Kgeo[9][16];
  Kgeo[16][10] = Kgeo[10][16];
  Kgeo[16][11] = Kgeo[11][16];
  Kgeo[16][12] = Kgeo[12][16];
  Kgeo[16][13] = Kgeo[13][16];
  Kgeo[16][14] = Kgeo[14][16];
  Kgeo[16][15] = Kgeo[15][16];
  Kgeo[16][16] = t65*Bnl[1][16];
  Kgeo[16][17] = t66*Bnl[2][17];
  Kgeo[16][18] = t67*Bnl[0][18];
  Kgeo[16][19] = t68*Bnl[1][19];
  Kgeo[16][20] = t66*Bnl[2][20];
  Kgeo[16][21] = t67*Bnl[0][21];
  Kgeo[16][22] = t68*Bnl[1][22];
  Kgeo[16][23] = t66*Bnl[2][23];
  Kgeo[17][0] = Kgeo[0][17];
  Kgeo[17][1] = Kgeo[1][17];
  Kgeo[17][2] = Kgeo[2][17];
  Kgeo[17][3] = Kgeo[3][17];
  Kgeo[17][4] = Kgeo[4][17];
  Kgeo[17][5] = Kgeo[5][17];
  Kgeo[17][6] = Kgeo[6][17];
  Kgeo[17][7] = Kgeo[7][17];
  Kgeo[17][8] = Kgeo[8][17];
  Kgeo[17][9] = Kgeo[9][17];
  Kgeo[17][10] = Kgeo[10][17];
  Kgeo[17][11] = Kgeo[11][17];
  Kgeo[17][12] = Kgeo[12][17];
  Kgeo[17][13] = Kgeo[13][17];
  Kgeo[17][14] = Kgeo[14][17];
  Kgeo[17][15] = Kgeo[15][17];
  Kgeo[17][16] = Kgeo[16][17];
  Kgeo[17][17] = t69*Bnl[2][17];
  Kgeo[17][18] = t70*Bnl[0][18];
  Kgeo[17][19] = t71*Bnl[1][19];
  Kgeo[17][20] = t72*Bnl[2][20];
  Kgeo[17][21] = t70*Bnl[0][21];
  Kgeo[17][22] = t71*Bnl[1][22];
  Kgeo[17][23] = t72*Bnl[2][23];
  Kgeo[18][0] = Kgeo[0][18];
  Kgeo[18][1] = Kgeo[1][18];
  Kgeo[18][2] = Kgeo[2][18];
  Kgeo[18][3] = Kgeo[3][18];
  Kgeo[18][4] = Kgeo[4][18];
  Kgeo[18][5] = Kgeo[5][18];
  Kgeo[18][6] = Kgeo[6][18];
  Kgeo[18][7] = Kgeo[7][18];
  Kgeo[18][8] = Kgeo[8][18];
  Kgeo[18][9] = Kgeo[9][18];
  Kgeo[18][10] = Kgeo[10][18];
  Kgeo[18][11] = Kgeo[11][18];
  Kgeo[18][12] = Kgeo[12][18];
  Kgeo[18][13] = Kgeo[13][18];
  Kgeo[18][14] = Kgeo[14][18];
  Kgeo[18][15] = Kgeo[15][18];
  Kgeo[18][16] = Kgeo[16][18];
  Kgeo[18][17] = Kgeo[17][18];
  Kgeo[18][18] = t73*Bnl[0][18];
  Kgeo[18][19] = t74*Bnl[1][19];
  Kgeo[18][20] = t75*Bnl[2][20];
  Kgeo[18][21] = t73*Bnl[0][21];
  Kgeo[18][22] = t74*Bnl[1][22];
  Kgeo[18][23] = t75*Bnl[2][23];
  Kgeo[19][0] = Kgeo[0][19];
  Kgeo[19][1] = Kgeo[1][19];
  Kgeo[19][2] = Kgeo[2][19];
  Kgeo[19][3] = Kgeo[3][19];
  Kgeo[19][4] = Kgeo[4][19];
  Kgeo[19][5] = Kgeo[5][19];
  Kgeo[19][6] = Kgeo[6][19];
  Kgeo[19][7] = Kgeo[7][19];
  Kgeo[19][8] = Kgeo[8][19];
  Kgeo[19][9] = Kgeo[9][19];
  Kgeo[19][10] = Kgeo[10][19];
  Kgeo[19][11] = Kgeo[11][19];
  Kgeo[19][12] = Kgeo[12][19];
  Kgeo[19][13] = Kgeo[13][19];
  Kgeo[19][14] = Kgeo[14][19];
  Kgeo[19][15] = Kgeo[15][19];
  Kgeo[19][16] = Kgeo[16][19];
  Kgeo[19][17] = Kgeo[17][19];
  Kgeo[19][18] = Kgeo[18][19];
  Kgeo[19][19] = t77*Bnl[1][19];
  Kgeo[19][20] = t78*Bnl[2][20];
  Kgeo[19][21] = t79*Bnl[0][21];
  Kgeo[19][22] = t77*Bnl[1][22];
  Kgeo[19][23] = t78*Bnl[2][23];
  Kgeo[20][0] = Kgeo[0][20];
  Kgeo[20][1] = Kgeo[1][20];
  Kgeo[20][2] = Kgeo[2][20];
  Kgeo[20][3] = Kgeo[3][20];
  Kgeo[20][4] = Kgeo[4][20];
  Kgeo[20][5] = Kgeo[5][20];
  Kgeo[20][6] = Kgeo[6][20];
  Kgeo[20][7] = Kgeo[7][20];
  Kgeo[20][8] = Kgeo[8][20];
  Kgeo[20][9] = Kgeo[9][20];
  Kgeo[20][10] = Kgeo[10][20];
  Kgeo[20][11] = Kgeo[11][20];
  Kgeo[20][12] = Kgeo[12][20];
  Kgeo[20][13] = Kgeo[13][20];
  Kgeo[20][14] = Kgeo[14][20];
  Kgeo[20][15] = Kgeo[15][20];
  Kgeo[20][16] = Kgeo[16][20];
  Kgeo[20][17] = Kgeo[17][20];
  Kgeo[20][18] = Kgeo[18][20];
  Kgeo[20][19] = Kgeo[19][20];
  Kgeo[20][20] = t81*Bnl[2][20];
  Kgeo[20][21] = t82*Bnl[0][21];
  Kgeo[20][22] = t83*Bnl[1][22];
  Kgeo[20][23] = t81*Bnl[2][23];
  Kgeo[21][0] = Kgeo[0][21];
  Kgeo[21][1] = Kgeo[1][21];
  Kgeo[21][2] = Kgeo[2][21];
  Kgeo[21][3] = Kgeo[3][21];
  Kgeo[21][4] = Kgeo[4][21];
  Kgeo[21][5] = Kgeo[5][21];
  Kgeo[21][6] = Kgeo[6][21];
  Kgeo[21][7] = Kgeo[7][21];
  Kgeo[21][8] = Kgeo[8][21];
  Kgeo[21][9] = Kgeo[9][21];
  Kgeo[21][10] = Kgeo[10][21];
  Kgeo[21][11] = Kgeo[11][21];
  Kgeo[21][12] = Kgeo[12][21];
  Kgeo[21][13] = Kgeo[13][21];
  Kgeo[21][14] = Kgeo[14][21];
  Kgeo[21][15] = Kgeo[15][21];
  Kgeo[21][16] = Kgeo[16][21];
  Kgeo[21][17] = Kgeo[17][21];
  Kgeo[21][18] = Kgeo[18][21];
  Kgeo[21][19] = Kgeo[19][21];
  Kgeo[21][20] = Kgeo[20][21];
  Kgeo[21][21] = t85*Bnl[0][21];
  Kgeo[21][22] = t86*Bnl[1][22];
  Kgeo[21][23] = t87*Bnl[2][23];
  Kgeo[22][0] = Kgeo[0][22];
  Kgeo[22][1] = Kgeo[1][22];
  Kgeo[22][2] = Kgeo[2][22];
  Kgeo[22][3] = Kgeo[3][22];
  Kgeo[22][4] = Kgeo[4][22];
  Kgeo[22][5] = Kgeo[5][22];
  Kgeo[22][6] = Kgeo[6][22];
  Kgeo[22][7] = Kgeo[7][22];
  Kgeo[22][8] = Kgeo[8][22];
  Kgeo[22][9] = Kgeo[9][22];
  Kgeo[22][10] = Kgeo[10][22];
  Kgeo[22][11] = Kgeo[11][22];
  Kgeo[22][12] = Kgeo[12][22];
  Kgeo[22][13] = Kgeo[13][22];
  Kgeo[22][14] = Kgeo[14][22];
  Kgeo[22][15] = Kgeo[15][22];
  Kgeo[22][16] = Kgeo[16][22];
  Kgeo[22][17] = Kgeo[17][22];
  Kgeo[22][18] = Kgeo[18][22];
  Kgeo[22][19] = Kgeo[19][22];
  Kgeo[22][20] = Kgeo[20][22];
  Kgeo[22][21] = Kgeo[21][22];
  Kgeo[22][22] = t88*Bnl[1][22];
  Kgeo[22][23] = t89*Bnl[2][23];
  Kgeo[23][0] = Kgeo[0][23];
  Kgeo[23][1] = Kgeo[1][23];
  Kgeo[23][2] = Kgeo[2][23];
  Kgeo[23][3] = Kgeo[3][23];
  Kgeo[23][4] = Kgeo[4][23];
  Kgeo[23][5] = Kgeo[5][23];
  Kgeo[23][6] = Kgeo[6][23];
  Kgeo[23][7] = Kgeo[7][23];
  Kgeo[23][8] = Kgeo[8][23];
  Kgeo[23][9] = Kgeo[9][23];
  Kgeo[23][10] = Kgeo[10][23];
  Kgeo[23][11] = Kgeo[11][23];
  Kgeo[23][12] = Kgeo[12][23];
  Kgeo[23][13] = Kgeo[13][23];
  Kgeo[23][14] = Kgeo[14][23];
  Kgeo[23][15] = Kgeo[15][23];
  Kgeo[23][16] = Kgeo[16][23];
  Kgeo[23][17] = Kgeo[17][23];
  Kgeo[23][18] = Kgeo[18][23];
  Kgeo[23][19] = Kgeo[19][23];
  Kgeo[23][20] = Kgeo[20][23];
  Kgeo[23][21] = Kgeo[21][23];
  Kgeo[23][22] = Kgeo[22][23];
  Kgeo[23][23] = t90*Bnl[2][23];
}

void
ViscoPlastic::carryForward(const PatchSubset* patches,
                                 const MPMMaterial* matl,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model 
    constParticleVariable<Matrix3> pLeftStretch, pRotation;
    constParticleVariable<double>  pPlasticStrain, pDamage, pPorosity, 
      pStrainRate;
    constParticleVariable<int>     pLocalized;
    constParticleVariable<double>  pPlasticTemp, pPlasticTempInc, pFailureVariable;

    old_dw->get(pLeftStretch,    pLeftStretchLabel,    pset);
    old_dw->get(pRotation,       pRotationLabel,       pset);
    old_dw->get(pStrainRate,     pStrainRateLabel,     pset);
    old_dw->get(pPlasticStrain,  pPlasticStrainLabel,  pset);
//     old_dw->get(pDamage,         pDamageLabel,         pset);
//     old_dw->get(pPorosity,       pPorosityLabel,       pset);
    old_dw->get(pLocalized,      pLocalizedLabel,      pset);
    old_dw->get(pPlasticTemp,    pPlasticTempLabel,    pset);
    old_dw->get(pPlasticTempInc, pPlasticTempIncLabel, pset);
    old_dw->get(pFailureVariable, pFailureVariableLabel, pset);

    ParticleVariable<Matrix3>      pLeftStretch_new, pRotation_new;
    ParticleVariable<double>       pPlasticStrain_new, pDamage_new, 
      pPorosity_new, pStrainRate_new;
    ParticleVariable<int>          pLocalized_new;
    ParticleVariable<double>       pPlasticTemp_new, pPlasticTempInc_new, pFailureVariable_new;

    new_dw->allocateAndPut(pLeftStretch_new, 
                           pLeftStretchLabel_preReloc,            pset);
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
//     new_dw->allocateAndPut(pDamage_new,      
//                            pDamageLabel_preReloc,                 pset);
//     new_dw->allocateAndPut(pPorosity_new,      
//                            pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pPlasticTemp_new,
                           pPlasticTempLabel_preReloc,            pset);
    new_dw->allocateAndPut(pPlasticTempInc_new,
                           pPlasticTempIncLabel_preReloc,         pset);
    new_dw->allocateAndPut(pFailureVariable_new,
                           pFailureVariableLabel_preReloc,         pset);

    // Get the plastic strain
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutRigid(pset, new_dw);

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pLeftStretch_new[idx] = pLeftStretch[idx];
      pRotation_new[idx] = pRotation[idx];
      pStrainRate_new[idx] = pStrainRate[idx];
      pPlasticStrain_new[idx] = pPlasticStrain[idx];
//       pDamage_new[idx] = pDamage[idx];
//       pPorosity_new[idx] = pPorosity[idx];
      pLocalized_new[idx] = pLocalized[idx];
      pPlasticTemp_new[idx] = pPlasticTemp[idx];
      pPlasticTempInc_new[idx] = 0.0;
      pFailureVariable_new[idx] = pFailureVariable[idx];
    }

    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

void 
ViscoPlastic::getPlasticTemperatureIncrement(ParticleSubset* pset,
                                                   DataWarehouse* new_dw,
                                                   ParticleVariable<double>& T) 
{
  constParticleVariable<double> pPlasticTempInc;
  new_dw->get(pPlasticTempInc, pPlasticTempIncLabel_preReloc, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++) T[*iter] = pPlasticTempInc[*iter];
}

void
ViscoPlastic::addRequiresDamageParameter(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* ) const
{
//const MaterialSubset* matlset = matl->thisMaterial();
//task-*/>requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}

void
ViscoPlastic::getDamageParameter(const Patch* patch,
                                       ParticleVariable<int>& damage,
                                       int dwi,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
//   ParticleSubset* pset = old_dw->getParticleSubset(dwi,patch);
//   constParticleVariable<int> pLocalized;
//   new_dw->get(pLocalized, pLocalizedLabel_preReloc, pset);
 
//   ParticleSubset::iterator iter;
//   for (iter = pset->begin(); iter != pset->end(); iter++) {
//     damage[*iter] = pLocalized[*iter];
  }
   
// }
         
// Actually calculate rotation
void
ViscoPlastic::computeUpdatedVR(const double& delT,
                                     const Matrix3& DD, 
                                     const Matrix3& WW,
                                     Matrix3& VV, 
                                     Matrix3& RR)  
{
  // Note:  The incremental polar decomposition algorithm is from
  // Flanagan and Taylor, 1987, Computer Methods in Applied Mechanics and
  // Engineering, v. 62, p.315.

  // Set up identity matrix
  Matrix3 one; one.Identity();

  // Calculate rate of rotation tensor (Omega)
  Matrix3 Omega = computeRateofRotation(VV, DD, WW);

  // Update the rotation tensor (R)
  Matrix3 oneMinusOmega = one - Omega*(0.5*delT);
  ASSERT(oneMinusOmega.Determinant() != 0.0);
  Matrix3 oneMinusOmegaInv = oneMinusOmega.Inverse();
  Matrix3 onePlusOmega = one + Omega*(0.5*delT);
  RR = (oneMinusOmegaInv*onePlusOmega)*RR;

  // Check the ortogonality of R
  //if (!RR.Orthogonal()) {
  // Do something here that restores orthogonality
  //}

  // Update the left Cauchy-Green stretch tensor (V)
  VV = VV + ((DD+WW)*VV - VV*Omega)*delT;

  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      VV(ii,jj) = (fabs(VV(ii,jj)) < d_tol) ? 0.0 : VV(ii,jj);
      RR(ii,jj) = (fabs(RR(ii,jj)) < d_tol) ? 0.0 : RR(ii,jj);
      //if (fabs(VV(ii,jj)) < d_tol) VV(ii,jj) = 0.0;
      //if (fabs(RR(ii,jj)) < d_tol) RR(ii,jj) = 0.0;
    }
  }
}

Matrix3 
ViscoPlastic::computeRateofRotation(const Matrix3& tensorV, 
                                          const Matrix3& tensorD,
                                          const Matrix3& tensorW)
{
  // Algorithm based on :
  // Dienes, J.K., 1979, Acta Mechanica, 32, p.222.
  // Belytschko, T. and others, 2000, Nonlinear finite elements ..., p.86.

  // Calculate vector w 
  double w[3];
  w[0] = -0.5*(tensorW(1,2)-tensorW(2,1));
  w[1] = -0.5*(tensorW(2,0)-tensorW(0,2));
  w[2] = -0.5*(tensorW(0,1)-tensorW(1,0));

  // Calculate tensor Z
  Matrix3 tensorZ = (tensorD*tensorV) - (tensorV*tensorD);

  // Calculate vector z
  double z[3];
  z[0] = -0.5*(tensorZ(1,2)-tensorZ(2,1));
  z[1] = -0.5*(tensorZ(2,0)-tensorZ(0,2));
  z[2] = -0.5*(tensorZ(0,1)-tensorZ(1,0));

  // Calculate I Trace(V) - V
  Matrix3 one;   one.Identity();
  Matrix3 temp = one*(tensorV.Trace()) - tensorV;
  if (temp.Determinant() == 0.0) {
    cout << "HEP:computeRdot:Determinant less than zero. ** ERROR ** " << endl;
  }
  ASSERT(temp.Determinant() != 0.0);
  temp = temp.Inverse();

  // Calculate vector omega = w + temp*z
  double omega[3];
  for (int ii = 0; ii < 3; ++ii) {
    double sum = 0.0;
    for (int jj = 0; jj < 3; ++jj) {
      sum += temp(ii,jj)*z[jj]; 
    }
    omega[ii] = w[ii] + sum;
  }

  // Calculate tensor Omega
  Matrix3 tensorOmega;
  tensorOmega(0,1) = -omega[2];  
  tensorOmega(0,2) = omega[1];  
  tensorOmega(1,0) = omega[2];  
  tensorOmega(1,2) = -omega[0];  
  tensorOmega(2,0) = -omega[1];  
  tensorOmega(2,1) = omega[0];  

  return tensorOmega;
}

// Compute the elastic tangent modulus tensor for isotropic
// materials (**NOTE** can get rid of one copy operation if needed)
void 
ViscoPlastic::computeElasticTangentModulus(double bulk,
                                                 double shear,
                                                 TangentModulusTensor& Ce)
{
  // Form the elastic tangent modulus tensor
  double E = 9.0*bulk*shear/(3.0*bulk+shear);
  double nu = E/(2.0*shear) - 1.0;
  double fac = E/((1.0+nu)*(1.0-2.0*nu));
  double C11 = fac*(1.0-nu);
  double C12 = fac*nu;
  FastMatrix C_6x6(6,6);
  for (int ii = 0; ii < 6; ++ii) 
    for (int jj = 0; jj < 6; ++jj) C_6x6(ii,jj) = 0.0;
  C_6x6(0,0) = C11; C_6x6(1,1) = C11; C_6x6(2,2) = C11;
  C_6x6(0,1) = C12; C_6x6(0,2) = C12; 
  C_6x6(1,0) = C12; C_6x6(1,2) = C12; 
  C_6x6(2,0) = C12; C_6x6(2,1) = C12; 
  C_6x6(3,3) = shear; C_6x6(4,4) = shear; C_6x6(5,5) = shear;
  
  Ce.convertToTensorForm(C_6x6);
}

void 
ViscoPlastic::convertToVoigtForm(const TangentModulusTensor Ce, 
      double D[6][6])
{
  int index[6][2];
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 2; ++jj) {
      index[ii][jj] = ii;
    }
  }
  index[3][0] = 1; index[3][1] = 2;
  index[4][0] = 2; index[4][1] = 0;
  index[5][0] = 0; index[5][1] = 1;

  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = 0; jj < 6; ++jj) {
      D[ii][jj] = Ce(index[ii][0], index[ii][1], 
                             index[jj][0], index[jj][1]);
    }
  }



}


double
ViscoPlastic::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;

  double p_gauge = pressure - p_ref;
  double rho_cur;

  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;  // modified EOS
    double n = p_ref/bulk;
    rho_cur  = rho_orig*pow(pressure/A,n);
  } else {             // Standard EOS
    rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  }
  return rho_cur;
}
//{
//  double rho_orig = matl->getInitialDensity();
//  double bulk = d_initialData.Bulk;
//  double p_gauge = pressure - p_ref;
//  return (rho_orig/(1.0-p_gauge/bulk));
//}

void
ViscoPlastic::computePressEOSCM(double rho_cur,double& pressure,
                                      double p_ref,  
                                      double& dp_drho, double& tmp,
                                      const MPMMaterial* matl,
                                      double temperature)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS
    double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = bulk/rho_cur;  // speed of sound squared
  }
}
//{
//  double rho_orig = matl->getInitialDensity();
//  double bulk = d_initialData.Bulk;
//  double p_g = bulk*(1.0 - rho_orig/rho_cur);
//  pressure = p_ref + p_g;
//  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
//  tmp = bulk/rho_cur;  // speed of sound squared
//}

double
ViscoPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

// check for failure
bool
ViscoPlastic::updateFailedParticlesAndModifyStress(const Matrix3& FF,
                                                const double& pFailureVariable,
                                                const int& pLocalized,
                                                int& pLocalized_new,
                                                Matrix3& pStress_new,
                                                const long64 particleID,
                                                const double temp_new,
                                                const double Tm_cur)
{
  Matrix3 Identity, zero(0.0); Identity.Identity();

bool isLocalized = false;
if (d_removeParticles) {
        
  // Check 1: Look at the temperature
   if (temp_new > Tm_cur && !isLocalized) {
       cout_CST << getpid() << "Particle localized. "
       << " Tm_cur = " << Tm_cur << " temp_new = " << temp_new
       << endl;
       isLocalized = true;
       } //end check 1

  // Check 4: Stability criterion
  
  // Initialize localization direction
//      Vector direction(0.0,0.0,0.0);
//     isLocalized = d_stable->checkStability(tensorSig, tensorD, Cep,
//                                                   direction);
//      if (isLocalized)
//          cout_CST << getpid() << "Particle " << idx << " localized. "
//                       << " using stability criterion" << endl;
  } //end removeParticles


Vector  eigval(0.0, 0.0, 0.0);
Matrix3 eigvec(0.0), ee(0.0);

double pressure = (1.0/3.0)*pStress_new.Trace();

if (!d_varf.failureByStress) {  //failure by strain only

  // Compute Finger tensor (left Cauchy-Green)
  Matrix3 bb = FF*FF.Transpose();

  // Compute Eulerian strain tensor
  ee = (Identity - bb.Inverse())*0.5;    
}

  double maxEigen=0.0, medEigen=0.0, minEigen=0.0;
  if (d_varf.failureByStress) {
      pStress_new.getEigenValues(maxEigen,medEigen,minEigen); //principal stress
  } else if (!d_varf.failureByPressure) { //failure by strain 
      ee.getEigenValues(maxEigen,medEigen,minEigen);;          //principal strain 
  }

double epsMax;

//  double epsMax = Max(fabs(eigval[0]),fabs(eigval[2]));
  if (d_varf.failureByPressure) {
      epsMax = pressure;
  } else {
    epsMax = maxEigen; //max principal stress or strain
  }

//  cout << "e0= " << eigval[0] << ", e2=" << eigval[2] << endl;
  // Find if the particle has failed
  pLocalized_new = pLocalized;

  if (epsMax > pFailureVariable) pLocalized_new = 1;
  if (pLocalized != pLocalized_new) {
     cout << "Particle " << particleID << " has failed: current value = " << epsMax 
          << ", max allowable  = " << pFailureVariable << endl;
     isLocalized = true;

     if (d_setStressToZero) pStress_new = zero;
     else if (d_allowNoTension) {
        if (pressure > 0.0) pStress_new = zero;
        else pStress_new = Identity*pressure;
     }
  }

return isLocalized;
}
