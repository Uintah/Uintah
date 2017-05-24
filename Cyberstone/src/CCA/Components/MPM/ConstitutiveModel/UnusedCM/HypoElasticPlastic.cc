/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/ConstitutiveModel/HypoElasticPlastic.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/YieldConditionFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/StabilityCheckFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PlasticityModelFactory.h>
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

#include <cmath>

#include <iostream>


using namespace std;
using namespace Uintah;

static DebugStream cout_CST("HEP",false);
static DebugStream cout_CST1("HEP1",false);
static DebugStream CSTi("HEPi",false);
static DebugStream CSTir("HEPir",false);


HypoElasticPlastic::HypoElasticPlastic(ProblemSpecP& ps, MPMFlags* Mflag) :
  ConstitutiveModel(Mflag), ImplicitCM()
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  d_initialData.alpha = 1.0e-5; // default is per K
  ps->get("coeff_thermal_expansion", d_initialData.alpha);

  ps->get("useModifiedEOS",d_useModifiedEOS);
  d_removeParticles = true;
  ps->get("remove_particles",d_removeParticles);
  d_setStressToZero = true;
  ps->get("zero_stress_upon_failure",d_setStressToZero);
  d_evolvePorosity = true;
  ps->get("evolve_porosity",d_evolvePorosity);
  d_evolveDamage = true;
  ps->get("evolve_damage",d_evolveDamage);
  d_checkTeplaFailureCriterion = true;
  ps->get("check_TEPLA_failure_criterion",d_checkTeplaFailureCriterion);
  d_tol = 1.0e-10;
  ps->get("tolerance",d_tol);
  d_initialMaterialTemperature = 294.0;
  ps->get("initial_material_temperature",d_initialMaterialTemperature);

  d_porosity.f0 = 0.002;  // Initial porosity
  d_porosity.f0_std = 0.002;  // Initial STD porosity
  d_porosity.fc = 0.5;    // Critical porosity
  d_porosity.fn = 0.1;    // Volume fraction of void nucleating particles
  d_porosity.en = 0.3;    // Mean strain for nucleation
  d_porosity.sn = 0.1;    // Standard deviation strain for nucleation
  d_porosity.porosityDist = "constant";
  ps->get("initial_mean_porosity",         d_porosity.f0);
  ps->get("initial_std_porosity",          d_porosity.f0_std);
  ps->get("critical_porosity",             d_porosity.fc);
  ps->get("frac_nucleation",               d_porosity.fn);
  ps->get("meanstrain_nucleation",         d_porosity.en);
  ps->get("stddevstrain_nucleation",       d_porosity.sn);
  ps->get("initial_porosity_distrib",      d_porosity.porosityDist);

  d_scalarDam.D0 = 0.0; // Initial scalar damage
  d_scalarDam.D0_std = 0.0; // Initial STD scalar damage
  d_scalarDam.Dc = 1.0; // Critical scalar damage
  d_scalarDam.scalarDamageDist = "constant";
  ps->get("initial_mean_scalar_damage",        d_scalarDam.D0);
  ps->get("initial_std_scalar_damage",         d_scalarDam.D0_std);
  ps->get("critical_scalar_damage",            d_scalarDam.Dc);
  ps->get("initial_scalar_damage_distrib",     d_scalarDam.scalarDamageDist);

  d_yield = YieldConditionFactory::create(ps);
  if(!d_yield){
    ostringstream desc;
    desc << "An error occured in the YieldConditionFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_stable = StabilityCheckFactory::create(ps);
  if(!d_stable) cerr << "Stability check disabled\n";

  d_plastic = PlasticityModelFactory::create(ps);
  if(!d_plastic){
    ostringstream desc;
    desc << "An error occured in the PlasticityModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_damage = DamageModelFactory::create(ps);
  if(!d_damage){
    ostringstream desc;
    desc << "An error occured in the DamageModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  d_eos = MPMEquationOfStateFactory::create(ps);
  d_eos->setBulkModulus(d_initialData.Bulk);
  if(!d_eos){
    ostringstream desc;
    desc << "An error occured in the EquationOfStateFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  // Initialize local VarLabels
  initializeLocalMPMLabels();

}

HypoElasticPlastic::HypoElasticPlastic(const HypoElasticPlastic* cm)
  : ConstitutiveModel(cm), ImplicitCM(cm)
{
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  d_initialData.alpha = cm->d_initialData.alpha;
  d_useModifiedEOS = cm->d_useModifiedEOS;
  d_removeParticles = cm->d_removeParticles;
  d_setStressToZero = cm->d_setStressToZero;
  d_evolvePorosity = cm->d_evolvePorosity;
  d_evolveDamage = cm->d_evolveDamage;
  d_checkTeplaFailureCriterion = cm->d_checkTeplaFailureCriterion;
  d_tol = cm->d_tol ;
  d_initialMaterialTemperature = cm->d_initialMaterialTemperature ;

  d_porosity.f0 = cm->d_porosity.f0 ;
  d_porosity.f0_std = cm->d_porosity.f0_std ;
  d_porosity.fc = cm->d_porosity.fc ;
  d_porosity.fn = cm->d_porosity.fn ;
  d_porosity.en = cm->d_porosity.en ;
  d_porosity.sn = cm->d_porosity.sn ;
  d_porosity.porosityDist = cm->d_porosity.porosityDist ;

  d_scalarDam.D0 = cm->d_scalarDam.D0 ;
  d_scalarDam.D0_std = cm->d_scalarDam.D0_std ;
  d_scalarDam.Dc = cm->d_scalarDam.Dc ;
  d_scalarDam.scalarDamageDist = cm->d_scalarDam.scalarDamageDist ;

  d_yield = YieldConditionFactory::createCopy(cm->d_yield);
  d_stable = StabilityCheckFactory::createCopy(cm->d_stable);
  d_plastic = PlasticityModelFactory::createCopy(cm->d_plastic);
  d_damage = DamageModelFactory::createCopy(cm->d_damage);
  d_eos = MPMEquationOfStateFactory::createCopy(cm->d_eos);
  d_eos->setBulkModulus(d_initialData.Bulk);
  
  // Initialize local VarLabels
  initializeLocalMPMLabels();

}

HypoElasticPlastic::~HypoElasticPlastic()
{
  // Destructor 
  VarLabel::destroy(pLeftStretchLabel);
  VarLabel::destroy(pRotationLabel);
  VarLabel::destroy(pStrainRateLabel);
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pPorosityLabel);
  VarLabel::destroy(pLocalizedLabel);
  VarLabel::destroy(pPlasticTempLabel);
  VarLabel::destroy(pPlasticTempIncLabel);

  VarLabel::destroy(pLeftStretchLabel_preReloc);
  VarLabel::destroy(pRotationLabel_preReloc);
  VarLabel::destroy(pStrainRateLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pDamageLabel_preReloc);
  VarLabel::destroy(pPorosityLabel_preReloc);
  VarLabel::destroy(pLocalizedLabel_preReloc);
  VarLabel::destroy(pPlasticTempLabel_preReloc);
  VarLabel::destroy(pPlasticTempIncLabel_preReloc);

  delete d_plastic;
  delete d_yield;
  delete d_stable;
  delete d_damage;
  delete d_eos;
}

void HypoElasticPlastic::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","hypoelastic_plastic");
  }

  cm_ps->appendElement("bulk_modulus",d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",d_initialData.Shear);
  cm_ps->appendElement("coeff_thermal_expansion", d_initialData.alpha);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
  cm_ps->appendElement("remove_particles",d_removeParticles);
  cm_ps->appendElement("zero_stress_upon_failure",d_setStressToZero);
  cm_ps->appendElement("evolve_porosity",d_evolvePorosity);
  cm_ps->appendElement("evolve_damage",d_evolveDamage);
  cm_ps->appendElement("check_TEPLA_failure_criterion",
                       d_checkTeplaFailureCriterion);
  cm_ps->appendElement("tolerance",d_tol);
  cm_ps->appendElement("initial_material_temperature",
                       d_initialMaterialTemperature);

  cm_ps->appendElement("initial_mean_porosity",d_porosity.f0);
  cm_ps->appendElement("initial_std_porosity",d_porosity.f0_std);
  cm_ps->appendElement("critical_porosity",d_porosity.fc);
  cm_ps->appendElement("frac_nucleation",d_porosity.fn);
  cm_ps->appendElement("meanstrain_nucleation",d_porosity.en);
  cm_ps->appendElement("stddevstrain_nucleation",d_porosity.sn);
  cm_ps->appendElement("initial_porosity_distrib",d_porosity.porosityDist);

  cm_ps->appendElement("initial_mean_scalar_damage",d_scalarDam.D0);
  cm_ps->appendElement("initial_std_scalar_damage",d_scalarDam.D0_std);
  cm_ps->appendElement("critical_scalar_damage",d_scalarDam.Dc);
  cm_ps->appendElement("initial_scalar_damage_distrib",
                       d_scalarDam.scalarDamageDist);

  d_yield->outputProblemSpec(cm_ps);
  d_stable->outputProblemSpec(cm_ps);
  d_plastic->outputProblemSpec(cm_ps);
  d_damage->outputProblemSpec(cm_ps);
  d_eos->outputProblemSpec(cm_ps);
}



HypoElasticPlastic* HypoElasticPlastic::clone()
{
  return scinew HypoElasticPlastic(*this);
}

void
HypoElasticPlastic::initializeLocalMPMLabels()
{
  pLeftStretchLabel = VarLabel::create("p.leftStretch",
        ParticleVariable<Matrix3>::getTypeDescription());
  pRotationLabel = VarLabel::create("p.rotation",
        ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel = VarLabel::create("p.strainRate",
        ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
        ParticleVariable<double>::getTypeDescription());
  pDamageLabel = VarLabel::create("p.damage",
        ParticleVariable<double>::getTypeDescription());
  pPorosityLabel = VarLabel::create("p.porosity",
        ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel = VarLabel::create("p.localized",
        ParticleVariable<int>::getTypeDescription());
  pPlasticTempLabel = VarLabel::create("p.plasticTemp",
        ParticleVariable<double>::getTypeDescription());
  pPlasticTempIncLabel = VarLabel::create("p.plasticTempInc",
        ParticleVariable<double>::getTypeDescription());

  pLeftStretchLabel_preReloc = VarLabel::create("p.leftStretch+",
        ParticleVariable<Matrix3>::getTypeDescription());
  pRotationLabel_preReloc = VarLabel::create("p.rotation+",
        ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel_preReloc = VarLabel::create("p.strainRate+",
        ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
        ParticleVariable<double>::getTypeDescription());
  pDamageLabel_preReloc = VarLabel::create("p.damage+",
        ParticleVariable<double>::getTypeDescription());
  pPorosityLabel_preReloc = VarLabel::create("p.porosity+",
        ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel_preReloc = VarLabel::create("p.localized+",
        ParticleVariable<int>::getTypeDescription());
  pPlasticTempLabel_preReloc = VarLabel::create("p.plasticTemp+",
        ParticleVariable<double>::getTypeDescription());
  pPlasticTempIncLabel_preReloc = VarLabel::create("p.plasticTempInc+",
        ParticleVariable<double>::getTypeDescription());
}

void 
HypoElasticPlastic::addParticleState(std::vector<const VarLabel*>& from,
                                     std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pLeftStretchLabel);
  from.push_back(pRotationLabel);
  from.push_back(pStrainRateLabel);
  from.push_back(pPlasticStrainLabel);
  from.push_back(pDamageLabel);
  from.push_back(pPorosityLabel);
  from.push_back(pLocalizedLabel);
  from.push_back(pPlasticTempLabel);
  from.push_back(pPlasticTempIncLabel);

  to.push_back(pLeftStretchLabel_preReloc);
  to.push_back(pRotationLabel_preReloc);
  to.push_back(pStrainRateLabel_preReloc);
  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pDamageLabel_preReloc);
  to.push_back(pPorosityLabel_preReloc);
  to.push_back(pLocalizedLabel_preReloc);
  to.push_back(pPlasticTempLabel_preReloc);
  to.push_back(pPlasticTempIncLabel_preReloc);

  // Add the particle state for the plasticity model
  d_plastic->addParticleState(from, to);
}

void 
HypoElasticPlastic::addInitialComputesAndRequires(Task* task,
                                                  const MPMMaterial* matl,
                                                  const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pLeftStretchLabel, matlset);
  task->computes(pRotationLabel, matlset);
  task->computes(pStrainRateLabel, matlset);
  task->computes(pPlasticStrainLabel, matlset);
  task->computes(pDamageLabel, matlset);
  task->computes(pPorosityLabel, matlset);
  task->computes(pLocalizedLabel, matlset);
  task->computes(pPlasticTempLabel, matlset);
  task->computes(pPlasticTempIncLabel, matlset);

  // Add internal evolution variables computed by plasticity model
  d_plastic->addInitialComputesAndRequires(task, matl, patch);
}

void 
HypoElasticPlastic::initializeCMData(const Patch* patch,
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
  //cout << "Initialize CM Data in HypoElasticPlastic" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pLeftStretch, pRotation;
  ParticleVariable<double> pPlasticStrain, pDamage, pPorosity, pStrainRate;
  ParticleVariable<int> pLocalized;
  ParticleVariable<double> pPlasticTemperature, pPlasticTempInc;

  new_dw->allocateAndPut(pLeftStretch, pLeftStretchLabel, pset);
  new_dw->allocateAndPut(pRotation, pRotationLabel, pset);
  new_dw->allocateAndPut(pStrainRate, pStrainRateLabel, pset);
  new_dw->allocateAndPut(pPlasticStrain, pPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pDamage, pDamageLabel, pset);
  new_dw->allocateAndPut(pLocalized, pLocalizedLabel, pset);
  new_dw->allocateAndPut(pPorosity, pPorosityLabel, pset);
  new_dw->allocateAndPut(pPlasticTemperature, pPlasticTempLabel, pset);
  new_dw->allocateAndPut(pPlasticTempInc, pPlasticTempIncLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    pLeftStretch[*iter] = one;
    pRotation[*iter] = one;
    pStrainRate[*iter] = 0.0;
    pPlasticStrain[*iter] = 0.0;
    pDamage[*iter] = d_damage->initialize();
    pPorosity[*iter] = d_porosity.f0;
    pLocalized[*iter] = 0;
    pPlasticTemperature[*iter] = d_initialMaterialTemperature;
    pPlasticTempInc[*iter] = 0.0;
  }

  // Do some extra things if the porosity or the damage distribution
  // is not uniform.  
  // ** WARNING ** Weibull distribution needs to be implemented.
  //               At present only Gaussian available.
  if (d_porosity.porosityDist != "constant") {

    Uintah::Gaussian gaussGen(d_porosity.f0, d_porosity.f0_std, 0, 1, DBL_MAX);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){

      // Generate a Gaussian distributed random number given the mean
      // porosity and the std.
      pPorosity[*iter] = fabs(gaussGen.rand(1.0));
    }
  }

  if (d_scalarDam.scalarDamageDist != "constant") {

    Uintah::Gaussian gaussGen(d_scalarDam.D0, d_scalarDam.D0_std, 0, 1,DBL_MAX);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){

      // Generate a Gaussian distributed random number given the mean
      // damage and the std.
      pDamage[*iter] = fabs(gaussGen.rand(1.0));
    }
  }

  // Initialize the data for the plasticity model
  d_plastic->initializeInternalVars(pset, new_dw);
}

void 
HypoElasticPlastic::computeStableTimestep(const Patch* patch,
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
HypoElasticPlastic::addComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  if (flag->d_integrator == MPMFlags::Implicit) {
    addSharedCRForImplicitHypo(task, matlset, true);
  } else {
    addSharedCRForHypoExplicit(task, matlset, patches);
  }

  // Other constitutive model and input dependent computes and requires
  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 

  task->requires(Task::OldDW, pLeftStretchLabel,     matlset, gnone);
  task->requires(Task::OldDW, pRotationLabel,        matlset, gnone);
  task->requires(Task::OldDW, pStrainRateLabel,      matlset, gnone);
  task->requires(Task::OldDW, pPlasticStrainLabel,   matlset, gnone);
  task->requires(Task::OldDW, pDamageLabel,          matlset, gnone);
  task->requires(Task::OldDW, pPorosityLabel,        matlset, gnone);
  task->requires(Task::OldDW, pLocalizedLabel,       matlset, gnone);
  task->requires(Task::OldDW, pPlasticTempLabel,     matlset, gnone);
  task->requires(Task::OldDW, pPlasticTempIncLabel,  matlset, gnone);

  task->computes(pLeftStretchLabel_preReloc,    matlset);
  task->computes(pRotationLabel_preReloc,       matlset);
  task->computes(pStrainRateLabel_preReloc,     matlset);
  task->computes(pPlasticStrainLabel_preReloc,  matlset);
  task->computes(pDamageLabel_preReloc,         matlset);
  task->computes(pPorosityLabel_preReloc,       matlset);
  task->computes(pLocalizedLabel_preReloc,      matlset);
  task->computes(pPlasticTempLabel_preReloc,    matlset);
  task->computes(pPlasticTempIncLabel_preReloc, matlset);

  // Add internal evolution variables computed by plasticity model
  d_plastic->addComputesAndRequires(task, matl, patches);

}

void 
HypoElasticPlastic::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  if (flag->d_integrator == MPMFlags::Implicit) {
    computeStressTensorImplicit(patches, matl, old_dw, new_dw);
    return;
  }

  //if ((patches->get(0))->getID() == 19)
  //  cout_CST << getpid() 
  //           << "ComputeStressTensor: In : Matl = " << matl << " id = " 
  //           << matl->getDWIndex() <<  " patch = " 
  //           << (patches->get(0))->getID();
  // General stuff
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  Matrix3 tensorL(0.0); // Velocity gradient
  Matrix3 tensorD(0.0); // Rate of deformation
//  Matrix3 tensorW(0.0); // Spin 
  Matrix3 tensorF; tensorF.Identity(); // Deformation gradient
  Matrix3 tensorV; tensorV.Identity(); // Left Cauchy-Green stretch
  Matrix3 tensorR; tensorR.Identity(); // Rotation 
  Matrix3 tensorSig(0.0); // The Cauchy stress
  Matrix3 tensorEta(0.0); // Deviatoric part of tensor D
  Matrix3 tensorS(0.0); // Devaitoric part of tensor Sig
  Matrix3 tensorF_new; tensorF_new.Identity(); // Deformation gradient

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double alpha = d_initialData.alpha;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();
  double sqrtTwo = sqrt(2.0);
  double totalStrainEnergy = 0.0;

  // Do thermal expansion?
  if(!flag->d_doThermalExpansion){
    alpha = 0;
  }

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    //cerr << getpid() << " patch = " << patch->getID() << endl;
    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

    // Get the set of particles
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // GET GLOBAL DATA 

    // Get the deformation gradient (F)
    // Note : The deformation gradient from the old datawarehouse is no
    // longer used, but it is updated for possible use elsewhere
    constParticleVariable<Matrix3>  pDeformGrad;
    old_dw->get(pDeformGrad, lb->pDeformationMeasureLabel, pset);

    // Get the particle location, particle size, particle mass, particle volume
    constParticleVariable<Point> px;
    constParticleVariable<Vector> psize;
    constParticleVariable<double> pMass;
    old_dw->get(px,    lb->pXLabel,    pset);
    old_dw->get(psize, lb->pSizeLabel, pset);
    old_dw->get(pMass, lb->pMassLabel, pset);

    // Get the velocity from the grid and particle velocity
    constParticleVariable<Vector> pVelocity;
    constNCVariable<Vector> gVelocity;
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(gVelocity, lb->gVelocityStarLabel, dwi, patch, gac, NGN);

    // Get the particle stress and temperature
    constParticleVariable<Matrix3> pStress;
    constParticleVariable<double>  pTempPrev, pTemperature;
    old_dw->get(pStress,      lb->pStressLabel,       pset);
    old_dw->get(pTempPrev,    lb->pTempPreviousLabel, pset); 
    old_dw->get(pTemperature, lb->pTemperatureLabel,  pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    constParticleVariable<Short27> pgCode;
    constNCVariable<Vector> GVelocity;
    if (flag->d_fracture) {
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      new_dw->get(GVelocity,lb->GVelocityStarLabel, dwi, patch, gac, NGN);
    }

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
      pStrainRate;
    old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
    old_dw->get(pDamage, pDamageLabel, pset);
    old_dw->get(pStrainRate, pStrainRateLabel, pset);
    old_dw->get(pPorosity, pPorosityLabel, pset);

    // Get the particle localization state
    constParticleVariable<int> pLocalized;
    old_dw->get(pLocalized, pLocalizedLabel, pset);

    // Create and allocate arrays for storing the updated information
    // GLOBAL
    ParticleVariable<Matrix3> pDeformGrad_new, pStress_new;
    ParticleVariable<double> pVolume_deformed;
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolume_deformed, 
                           lb->pVolumeLabel_preReloc,             pset);

    // LOCAL
    ParticleVariable<Matrix3> pLeftStretch_new, pRotation_new;
    ParticleVariable<double>  pPlasticStrain_new, pDamage_new, pPorosity_new, 
      pStrainRate_new;
    ParticleVariable<double>  pPlasticTemperature_new, pPlasticTempInc_new;
    ParticleVariable<int>     pLocalized_new;
    ParticleVariable<double> pdTdt,p_q;
    new_dw->allocateAndPut(pLeftStretch_new, 
                           pLeftStretchLabel_preReloc,            pset);
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pPorosity_new,      
                           pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pPlasticTemperature_new,      
                           pPlasticTempLabel_preReloc,            pset);
    new_dw->allocateAndPut(pPlasticTempInc_new,      
                           pPlasticTempIncLabel_preReloc,         pset);
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel_preReloc,        pset);
    new_dw->allocateAndPut(p_q,   lb->p_qLabel_preReloc,          pset);

    // Get the plastic strain
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Calculate the velocity gradient (L) from the grid velocity
      Matrix3 tensorL(0.0);
      short pgFld[27];
      if (flag->d_fracture) {
        for(int k=0; k<27; k++){ 
          pgFld[k]=pgCode[idx][k];
        }
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDeformGrad[idx]);
        computeVelocityGradient(tensorL,ni,d_S,oodx,pgFld,gVelocity,GVelocity);
      } else {
        if(!flag->d_axisymmetric){
         // Get the node indices that surround the cell
         interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDeformGrad[idx]);

         computeVelocityGradient(tensorL,ni,d_S, oodx, gVelocity);
        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                    psize[idx],pDeformGrad[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gVelocity,px[idx]);
        }
      }

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      Matrix3 tensorFinc = tensorL*delT + one;
      tensorF_new = tensorFinc*pDeformGrad[idx];
      pDeformGrad_new[idx] = tensorF_new;
      double J = tensorF_new.Determinant();

      // Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw ParameterNotFound("**ERROR**:HypoElasticPlastic", __FILE__, __LINE__);
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;
      pVolume_deformed[idx]=pMass[idx]/rho_cur;

      // Compute polar decomposition of F (F = VR)
      pDeformGrad[idx].polarDecompositionRMB(tensorV, tensorR);

      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
          tensorD(ii,jj)=(fabs(tensorD(ii,jj)) < d_tol) ? 0.0 : tensorD(ii,jj);
        }
      }

      // If the particle is just sitting there, do nothing
      double defRateSq = tensorD.NormSquared();
      if (!(defRateSq > 0)) {
        pStress_new[idx] = pStress[idx];
        pStrainRate_new[idx] = 0.0;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pDamage_new[idx] = pDamage[idx];
        pPorosity_new[idx] = pPorosity[idx];
        pLocalized_new[idx] = pLocalized[idx];
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

      // Rotate the Cauchy stress back to the 
      // material configuration and calculate the deviatoric part
      tensorSig = pStress[idx];
      tensorSig = (tensorR.Transpose())*(tensorSig*tensorR);
      double pressure = tensorSig.Trace()/3.0;
      Matrix3 tensorP = one*pressure;
      tensorS = tensorSig - tensorP;

      // Calculate the temperature at the start of the time step
      double temperature = pTemperature[idx];

      // Calculate the plastic strain rate and plastic strain
      double epdot = sqrt(tensorEta.NormSquared()/1.5);
      double ep = pPlasticStrain[idx] + epdot*delT;

      // Get the specific heat
      double C_p = matl->getSpecificHeat();

      // Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->strainRate = pStrainRate_new[idx];
      state->plasticStrainRate = epdot;
      state->plasticStrain = ep;
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

      // Integrate the stress rate equation to get a trial deviatoric stress
      Matrix3 trialS = tensorS + tensorEta*(2.0*mu_cur*delT);

      // Calculate the equivalent stress
      double equivStress = sqrt((trialS.NormSquared())*1.5);

      // Calculate flow stress (strain driven problem)
      double flowStress = d_plastic->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);

      // Get the current porosity 
      double porosity = pPorosity[idx];

      // Evaluate yield condition
      double traceOfTrialStress = 3.0*pressure + 
        tensorD.Trace()*(2.0*mu_cur*delT);
      double sig = flowStress;
      double Phi = d_yield->evalYieldCondition(equivStress, flowStress,
                                               traceOfTrialStress, 
                                               porosity, sig);
      
      // Compute bulk viscosity
      double de_s=0.;
      if (flag->d_artificial_viscosity) {
        double Dkk = 0.5*(tensorL+tensorL.Transpose()).Trace();
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
        de_s = -p_q[idx]*Dkk/rho_cur;
      } else {
        p_q[idx] = 0.;
        de_s = 0.;
      }

      // Calculate Tdot due to artificial viscosity
      double Tdot_AV = de_s/C_p;
      pdTdt[idx] += Tdot_AV;

      // Compute the deviatoric stress
      if (Phi <= 0.0 || flowStress <= 0.0) {

        // Do the standard hypoelastic-plastic stress update
        // Calculate the updated hydrostatic stress
        double p = d_eos->computePressure(matl, state, tensorF_new, tensorD, 
                                        delT);
        //p -= qVisco;

        Matrix3 tensorHy = one*p;

        // Get the elastic stress
        tensorSig = trialS + tensorHy;

        // Compute polar decomposition of Fnew (F = VR)
        tensorF_new.polarDecompositionRMB(tensorV, tensorR);

        // Update the kinematic variables
        pLeftStretch_new[idx] = tensorV;
        pRotation_new[idx] = tensorR;


        // Rotate the stress rate back to the laboratory coordinates
        // to get the "true" Cauchy stress
        tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

        // Save the updated data
        pStress_new[idx] = tensorSig;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pDamage_new[idx] = pDamage[idx];
        pPorosity_new[idx] = pPorosity[idx];
        
        // Update the internal variables
        d_plastic->updateElastic(idx);

        // Update the temperature
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx];
        pPlasticTempInc_new[idx] = 0.0;

        // Compute stability criterion
        pLocalized_new[idx] = pLocalized[idx];

      } else {

        // Using the algorithm from Zocher, Maudlin, Chen, Flower-Maudlin
        // European Congress on Computational Methods in Applied Sciences 
        // and Engineering,  September 11-14, 2000.
        // Basic assumption is that all strain rate is plastic strain rate
        ASSERT(flowStress != 0);
        if (flowStress == 0) {
          cout << getpid() << " HEP:flowstress = " << flowStress << endl;
          throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
        }

        Matrix3 Stilde(0.0);
        double delGamma = 0.0;
        double sqrtSxS = tensorS.Norm(); 
        if (sqrtSxS == 0 || tensorS.Determinant() == 0.0) { 
          // If the material goes plastic in the first step, 
          Stilde = trialS;
          delGamma = ((equivStress-flowStress)/(2.0*mu_cur))/
            (1.0+bulk/(3.0*mu_cur));

        } else {

          // Calculate the derivative of the yield function (using the 
          // previous time step (n) values)
          Matrix3 q(0.0);
          d_yield->evalDevDerivOfYieldFunction(tensorSig, flowStress, 
                                               porosity, q);

          // Calculate the tensor u (at start of time interval)
          double sqrtqs = sqrt(q.Contract(tensorS));
          ASSERT(sqrtqs != 0);
          if (sqrtqs == 0) {
            cout << getpid() << " HEP:sqrtqs = " << sqrtqs << " q = " << q
                 << " S = " << tensorS << endl;
            throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
          }
          Matrix3 u = q/sqrtqs;

          // Calculate c and d at the beginning of time step
          double cplus = u.NormSquared();
          double dplus = u.Contract(tensorEta);
         
          // Calculate gamma_dot at the beginning of the time step
          ASSERT(cplus != 0);
          if (cplus == 0) {
            cout << getpid() << " HEP:cplus = " << cplus 
                 << " u = " << u << endl;
            throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
          }
          double gammadotplus = dplus/cplus;

          // Set initial theta
          double theta = 0.0;

          // Calculate u_q and u_eta
          double etaeta = sqrt(tensorEta.NormSquared());
          ASSERT(etaeta != 0);
          if (etaeta == 0) {
            cout << getpid() << " HEP:etaeta = " << etaeta << " L = " << tensorL
                 << " D = " << tensorD  << " Eta = " << tensorEta << endl;
            throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
          }
          Matrix3 u_eta = tensorEta/etaeta;
          double qq = sqrt(q.NormSquared());
          ASSERT(qq != 0);
          if (qq == 0) {
            cout << getpid() << " HEP:qq = " << qq << " q = " << q << endl;
            throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
          }
          Matrix3 u_q = q/qq;

          // Calculate new dstar
          int count = 1;
          double dStarOld = 0.0;
          double dStar = dplus;
          while (count < 10) {
            dStarOld = dStar;

            // Calculate dStar
            dStar = ((1.0-0.5*theta)*u_eta.Contract(tensorEta) + 
                     0.5*theta*u_q.Contract(tensorEta))*sqrt(cplus);

            // Update theta
            ASSERT(dStar != 0);
            if (dStar == 0) {
              cout << getpid() << " HEP:dStar = " << dStar 
                   << " theta = " << theta
                   << " u_eta = " << u_eta 
                   << " Eta = " << tensorEta << " u_q = " << u_q
                   << " cplus = " << cplus << endl;
              throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
            }
            theta = (dStar - cplus*gammadotplus)/dStar;
            ++count;
            //cout << "dStar = " << dStar << " dStarOld " << dStarOld 
            //     << " count " << count << endl;
            double tol_dStar = dStar*1.0e-6;
            if (fabs(dStar-dStarOld) < tol_dStar) break;
          } 

          // Calculate delGammaEr
          double delGammaEr =  (sqrtTwo*sig - sqrtqs)/(2.0*mu_cur*cplus);

          // Calculate delGamma
          delGamma = dStar/cplus*delT - delGammaEr;

          // Calculate Stilde
          ASSERT(sig != 0);
          if (sig == 0) {
            cout << getpid() << " HEP:sig = " << sig << endl;
            throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
          }
          double denom = 1.0 + (3.0*sqrtTwo*mu_cur*delGamma)/sig; 
          ASSERT(denom != 0);
          if (denom == 0) {
            cout << getpid() << " HEP:denom = " << denom 
                 << " mu_cur = " << mu_cur
                 << " delGamma = " << delGamma << " sig = " << sig << endl;
            throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
          }
          Stilde = trialS/denom;
        }
        
        // Do radial return adjustment
        double stst = sqrt(1.5*Stilde.NormSquared());
        ASSERT(stst != 0);
        if (stst == 0) {
          cout << getpid() << " HEP:stst = " << stst 
               << " Stilde = " << Stilde << endl;
          throw ParameterNotFound("**ERROR**", __FILE__, __LINE__);
        }
        tensorS = Stilde*(sig/stst);
        equivStress = sqrt((tensorS.NormSquared())*1.5);

        // Do the standard hypoelastic-plastic stress update
        // Calculate the updated hydrostatic stress
        double p = d_eos->computePressure(matl, state, tensorF_new, tensorD, 
                                        delT);
        //p -= qVisco;
        Matrix3 tensorHy = one*p;

        // Calculate total stress
        tensorSig = tensorS + tensorHy;

        // Update the plastic strain
        pPlasticStrain_new[idx] = ep;

        // Update the porosity
        if (d_evolvePorosity) 
          pPorosity_new[idx] = updatePorosity(tensorD, delT, porosity, ep);
        else
          pPorosity_new[idx] = pPorosity[idx];

        // Calculate the updated scalar damage parameter
        if (d_evolveDamage) 
          pDamage_new[idx] = d_damage->computeScalarDamage(epdot, tensorSig, 
                                                           temperature,
                                                           delT, matl, d_tol, 
                                                           pDamage[idx]);
        else
          pDamage_new[idx] = pDamage[idx];

        // Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = 0.9;

        // ** WARNING ** Special for steel (remove for other materials)
        //double T = temperature;
        //C_p = 1.0e3*(0.09278 + 7.454e-4*T + 12404.0/(T*T));

        // Alternative approach
        double Tdot_PW = flowStress*epdot*taylorQuinney/(rho_cur*C_p);

        pdTdt[idx] += Tdot_PW;
        double dT = Tdot_PW*delT;
        pPlasticTempInc_new[idx] = dT;
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx] + dT; 
        double temp_new = temperature + dT;

            // cout_CST << "HEP::Particle = " << idx 
                 //  << " T_old = " << temperature
                 //  << " Tdot = " << Tdot
                 //  << " dT = " << dT 
                 //  << " T_new = " << temp_new << endl;

        /*
        // Calculate Tdot (do not allow negative Tdot)
        // (this is the internal heating rate)
        // Update the plastic temperature
        Tdot = tensorS.Contract(tensorEta)*(taylorQuinney/(rho_cur*C_p));
        Tdot = max(Tdot, 0.0);
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx] + Tdot*delT; 
        pPlasticTempInc_new[idx] = Tdot*delT; 
        double temp_new = temperature + pPlasticTempInc_new[idx];
        */

        // Find if the particle has localized
        pLocalized_new[idx] = pLocalized[idx];

        bool isLocalized = false;
        if (d_removeParticles) {

          // Check 0: See is the particle is already localized
          // If this is done the kinetic energy goes through the roof
          //if (pLocalized[idx]) isLocalized = true;

          // Check 1: Look at the temperature
          if (temp_new > Tm_cur && !isLocalized) {

            cout_CST << getpid() << "Particle " << idx << " localized. " 
                     << " Tm_cur = " << Tm_cur << " temp_new = " << temp_new
                     << endl;
            isLocalized = true;
          } 

          // Check 3: Modified Tepla rule
          if (d_checkTeplaFailureCriterion && !isLocalized) {
            double tepla = pPorosity_new[idx]/d_porosity.fc + pDamage_new[idx];
            if (tepla > 1.0) 
              isLocalized = true;

            if (isLocalized)
               cout_CST << getpid() << "Particle " << idx << " localized. " 
                        << " porosity_new = " << pPorosity_new[idx]
                        << " damage_new = " << pDamage_new[idx]
                        << " TEPLA = " << tepla << endl;
          } 

          // Check 4: Stability criterion
          if (d_stable->doIt() && !isLocalized) {

            // Calculate the elastic tangent modulus
            TangentModulusTensor Ce;
            computeElasticTangentModulus(bulk, mu_cur, Ce);
  
            // Calculate values needed for tangent modulus calculation
            state->temperature = temp_new;
            mu_cur = d_plastic->computeShearModulus(state);
            Tm_cur = d_plastic->computeMeltingTemp(state);
            double sigY = d_plastic->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);
            double dsigYdep = 
              d_plastic->evalDerivativeWRTPlasticStrain(state, idx);
            double A = voidNucleationFactor(ep);

            // Calculate the elastic-plastic tangent modulus
            TangentModulusTensor Cep;
            d_yield->computeElasPlasTangentModulus(Ce, tensorSig, sigY, 
                                                   dsigYdep,
                                                   pPorosity_new[idx], A, Cep);
          
            // Initialize localization direction
            Vector direction(0.0,0.0,0.0);
            isLocalized = d_stable->checkStability(tensorSig, tensorD, Cep, 
                                                   direction);
            
            if (isLocalized)
              cout_CST << getpid() << "Particle " << idx << " localized. " 
                       << " using stability criterion" << endl;
          }
        }

        // set the stress to zero
        if (pLocalized[idx] && d_setStressToZero) tensorSig = zero;

        if (isLocalized) {

          // set the particle localization flag to true  
          pLocalized_new[idx] = 1;
          pDamage_new[idx] = 0.0;
          pPorosity_new[idx] = 0.0;

          // set the stress to zero
          if (d_setStressToZero) tensorSig = zero;

          // Rotate the stress back to the laboratory coordinates
          // Save the new data
          tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());
          pStress_new[idx] = tensorSig;

          d_plastic->updateElastic(idx);
         
        } else {

          // Rotate the stress back to the laboratory coordinates
          tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

          // Save the new data
          pStress_new[idx] = tensorSig;
        
          // Update internal variables
          d_plastic->updatePlastic(idx, delGamma);
        }

      }

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
      }                 

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      delete state;
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
  // cout_CST << getpid() << "... Out" << endl;
}

void 
HypoElasticPlastic::computeStressTensorImplicit(const PatchSubset* patches,
                                                const MPMMaterial* matl,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  // Constants
  int dwi = matl->getDWIndex();
  double sqrtTwoThird = sqrt(2.0/3.0);
  Ghost::GhostType gac = Ghost::AroundCells;
  Matrix3 One; One.Identity(); Matrix3 Zero(0.0);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double alpha = d_initialData.alpha;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();

  // Do thermal expansion?
  if(!flag->d_doThermalExpansion){
    alpha = 0;
  }

  // Particle and Grid data
  delt_vartype delT;
  constParticleVariable<int>     pLocalized;
  constParticleVariable<double>  pMass, pVolume,
                                 pTempPrev, pTemperature,
                                 pPlasticTemp, pPlasticTempInc,
                                 pPlasticStrain, pDamage, pPorosity, 
                                 pStrainRate;

  constParticleVariable<Point>   px;
  constParticleVariable<Vector>  psize;
  constParticleVariable<Matrix3> pDeformGrad, pStress,
                                 pLeftStretch, pRotation;
  constNCVariable<Vector>        gDisp;

  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<Matrix3>      pDeformGrad_new, pStress_new,
                                 pLeftStretch_new, pRotation_new;
  ParticleVariable<double>       pVolume_deformed, pPlasticStrain_new, 
                                 pDamage_new, pPorosity_new, pStrainRate_new,
                                 pPlasticTemp_new, pPlasticTempInc_new,
                                 pdTdt;

  // Local variables
  Matrix3 DispGrad(0.0); // Displacement gradient
  Matrix3 DefGrad, incDefGrad, incFFt, incFFtInv, LeftStretch, Rotation; 
  Matrix3 incTotalStrain(0.0), incThermalStrain(0.0), incStrain(0.0);
  Matrix3 oldStress(0.0), devStressOld(0.0), trialStress(0.0),
          devTrialStress(0.0);
  DefGrad.Identity(); incDefGrad.Identity(); incFFt.Identity(); 
  incFFtInv.Identity(); LeftStretch.Identity(); Rotation.Identity();

  //CSTi << getpid() 
  //     << "ComputeStressTensorImplicit: In : Matl = " << matl << " id = " 
  //     << matl->getDWIndex() <<  " patch = " 
  //     << (patches->get(0))->getID();

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

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

    // GET LOCAL DATA 
    old_dw->get(pLeftStretch,        pLeftStretchLabel,    pset);
    old_dw->get(pRotation,           pRotationLabel,       pset);
    old_dw->get(pPlasticTemp,        pPlasticTempLabel,    pset);
    old_dw->get(pPlasticTempInc,     pPlasticTempIncLabel, pset);
    old_dw->get(pPlasticStrain,      pPlasticStrainLabel,  pset);
    old_dw->get(pDamage,             pDamageLabel,         pset);
    old_dw->get(pStrainRate,         pStrainRateLabel,     pset);
    old_dw->get(pPorosity,           pPorosityLabel,       pset);
    old_dw->get(pLocalized,          pLocalizedLabel,      pset);

    // Create and allocate arrays for storing the updated information
    // GLOBAL
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolume_deformed, 
                           lb->pVolumeDeformedLabel,              pset);
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel_preReloc,        pset);

    // LOCAL
    new_dw->allocateAndPut(pLeftStretch_new, 
                           pLeftStretchLabel_preReloc,            pset);
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pPorosity_new,      
                           pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pPlasticTemp_new,      
                           pPlasticTempLabel_preReloc,            pset);
    new_dw->allocateAndPut(pPlasticTempInc_new,      
                           pPlasticTempIncLabel_preReloc,         pset);

    // Get the plastic strain
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);

    // Special case for rigid materials
    double totalStrainEnergy = 0.0;
    if (matl->getIsRigid()) {
      ParticleSubset::iterator iter = pset->begin(); 
      for( ; iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pLeftStretch_new[idx] = pLeftStretch[idx];
        pRotation_new[idx] = pRotation[idx];
        pStrainRate_new[idx] = pStrainRate[idx];
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pDamage_new[idx] = pDamage[idx];
        pPorosity_new[idx] = pPorosity[idx];
        pLocalized_new[idx] = pLocalized[idx];
        pPlasticTemp_new[idx] = pPlasticTemp[idx];
        pPlasticTempInc_new[idx] = 0.0;

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

    // Standard case for deformable materials
    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Calculate the displacement gradient
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDeformGrad[idx]);
      computeGrad(DispGrad, ni, d_S, oodx, gDisp);

      // Compute the deformation gradient increment
      incDefGrad = DispGrad + One;
      //double Jinc = incDefGrad.Determinant();

      // Update the deformation gradient
      DefGrad = incDefGrad*pDeformGrad[idx];
      pDeformGrad_new[idx] = DefGrad;
      double J = DefGrad.Determinant();

      // Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw ParameterNotFound("**ERROR**:HypoElasticPlastic:Implicit", __FILE__, __LINE__);
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;
      double volold = (pMass[idx]/rho_0);
      pVolume_deformed[idx]=volold*J;

      // Compute polar decomposition of F (F = VR)
      // (**NOTE** This is being done to provide reasonable starting 
      //           values for R and V if the incremental algorithm 
      //           for the polar decomposition is used in the explicit
      //           calculations following an implicit calculation.)
      DefGrad.polarDecompositionRMB(LeftStretch, Rotation);
      pLeftStretch_new[idx] = LeftStretch;
      pRotation_new[idx] = Rotation;

      // Compute the current strain and strain rate
      incFFt = incDefGrad*incDefGrad.Transpose(); 
      incFFtInv = incFFt.Inverse();
      incTotalStrain = (One - incFFtInv)*0.5;
      pStrainRate_new[idx] = incTotalStrain.Norm()*sqrtTwoThird/delT;
      
      // Compute thermal strain
      double incT = pTemperature[idx] - pTempPrev[idx];
      incThermalStrain = One*(alpha*incT);
      incStrain = incTotalStrain - incThermalStrain;
      
      // Compute pressure and deviatoric stress at t_n and
      // the volumetric strain and deviatoric strain increments at t_n+1
      oldStress = pStress[idx];
      double pressure = oldStress.Trace()/3.0;
      //Matrix3 devStressOld = oldStress - One*pressure;
      
      // Get the specific heat
      double C_p = matl->getSpecificHeat();

      // Set up the PlasticityState
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
      double equivStress = sqrt((devTrialStress.NormSquared())*1.5);

      // Calculate flow stress (strain driven problem)
      double flowStress = d_plastic->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);

      // Get the current porosity 
      double porosity = pPorosity[idx];

      // Evaluate yield condition
      double traceOfTrialStress = trialStress.Trace();
      double sig = flowStress;
      double Phi = d_yield->evalYieldCondition(equivStress, flowStress,
                                               traceOfTrialStress, 
                                               porosity, sig);
      
      // Compute the deviatoric stress
      if (Phi < 0.0) {

        // Save the updated data
        pStress_new[idx] = trialStress;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pDamage_new[idx] = pDamage[idx];
        pPorosity_new[idx] = pPorosity[idx];
        
        // Update the internal variables
        d_plastic->updateElastic(idx);

        // Update the temperature
        pPlasticTemp_new[idx] = pPlasticTemp[idx];
        pPlasticTempInc_new[idx] = 0.0;

        // Compute stability criterion
        pLocalized_new[idx] = pLocalized[idx];

      } else {

        // Do Newton iteration to compute delGamma and updated 
        // plastic strain, plastic strain rate, and yield stress
        double normTrialS = devTrialStress.Norm();
        double tolerance = min((double) delT, 1.0e-6);
        double delGamma = computeDeltaGamma(delT, tolerance, normTrialS, matl,
                                            idx, state);
        Matrix3 nn = devTrialStress/normTrialS;
        pStress_new[idx] = trialStress - nn*(2.0*state->shearModulus*delGamma);
        pPlasticStrain_new[idx] = state->plasticStrain;

        // Update the porosity
        if (d_evolvePorosity) {
          Matrix3 tensorD = incStrain/delT;
          double ep = state->plasticStrain;
          pPorosity_new[idx] = updatePorosity(tensorD, delT, porosity, ep);
        } else {
          pPorosity_new[idx] = pPorosity[idx];
        }

        // Calculate the updated scalar damage parameter
        if (d_evolveDamage) 
          pDamage_new[idx] = 
            d_damage->computeScalarDamage(state->plasticStrainRate, 
                                          pStress_new[idx],
                                          pTemperature[idx],
                                          delT, matl, d_tol, 
                                          pDamage[idx]);
        else
          pDamage_new[idx] = pDamage[idx];

        // Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = 0.9;
        double Tdot = flowStress*state->plasticStrainRate*taylorQuinney/
                      (rho_cur*C_p);
        pdTdt[idx] = Tdot;
        double dT = Tdot*delT;
        pPlasticTempInc_new[idx] = dT;
        pPlasticTemp_new[idx] = pPlasticTemp[idx] + dT; 

        // No failure implemented for implcit time integration
        pLocalized_new[idx] = pLocalized[idx];

        // Update internal variables in the plasticity model
        d_plastic->updatePlastic(idx, delGamma);
      }

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
    }
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void 
HypoElasticPlastic::addComputesAndRequires(Task* task,
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
    task->requires(Task::ParentOldDW, pPorosityLabel,         matlset, gnone);
  }else{
    task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 
    task->requires(Task::OldDW, lb->pTemperatureLabel,  matlset, gnone);
    task->requires(Task::OldDW, pPlasticStrainLabel,    matlset, gnone);
    task->requires(Task::OldDW, pPorosityLabel,         matlset, gnone);
  }
}

void 
HypoElasticPlastic::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        Solver* solver,
                                        const bool )
{
  // Constants
  Ghost::GhostType gac = Ghost::AroundCells;
  Matrix3 One; One.Identity(); Matrix3 Zero(0.0);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double alpha = d_initialData.alpha;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();

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
  constParticleVariable<Vector>  psize;
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
    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    // Get patch indices for parallel solver
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

    parent_old_dw->get(pTempPrev,    lb->pTempPreviousLabel,       pset); 
    parent_old_dw->get(pTemperature, lb->pTemperatureLabel,        pset);
    parent_old_dw->get(px,           lb->pXLabel,                  pset);
    parent_old_dw->get(psize,        lb->pSizeLabel,               pset);
    parent_old_dw->get(pDeformGrad,  lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pStress,      lb->pStressLabel,             pset);

    // GET LOCAL DATA 
    parent_old_dw->get(pPlasticStrain,      pPlasticStrainLabel,  pset);
    parent_old_dw->get(pPorosity,           pPorosityLabel,       pset);

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
    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      //CSTir << " patch = " << patch << " particle = " << idx << endl;

      // Calculate the displacement gradient
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
        throw ParameterNotFound("**ERROR**:HypoElasticPlastic:Implicit", __FILE__, __LINE__);
      }

      //CSTir << " particle = " << idx << " J = " << J << endl;

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
      double equivStress = sqrt((devTrialStress.NormSquared())*1.5);

      // Calculate flow stress (strain driven problem)
      double flowStress = d_plastic->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);

      // Get the current porosity 
      double porosity = pPorosity[idx];

      // Evaluate yield condition
      double traceOfTrialStress = trialStress.Trace();
      double sig = flowStress;
      double Phi = d_yield->evalYieldCondition(equivStress, flowStress,
                                               traceOfTrialStress, 
                                               porosity, sig);
      
      //CSTir << " particle = " << idx << " Seqv = " << equivStress  
      //      << " Sflow = " << flowStress << " Phi = " << Phi << endl;

      // Compute the deviatoric stress
      if (Phi < 0.0) {

        // Save the updated data
        pStress_new[idx] = trialStress;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        
        computeElasticTangentModulus(bulk, shear, D);

        //CSTir << " Elastic particle = " << idx 
        //      << " stress = " << pStress_new[idx] << endl;  

      } else {

        // Do Newton iteration to compute delGamma and updated 
        // plastic strain, plastic strain rate, and yield stress
        double normTrialS = devTrialStress.Norm();
        double tolerance = min((double)delT, 1.0e-6);
        double delGamma = computeDeltaGamma(delT, tolerance, normTrialS, matl,
                                            idx, state);
        Matrix3 nn = devTrialStress/normTrialS;
        pStress_new[idx] = trialStress - nn*(2.0*state->shearModulus*delGamma);
        pPlasticStrain_new[idx] = state->plasticStrain;

        computeEPlasticTangentModulus(bulk, shear, delGamma, normTrialS,
                                      idx, nn, state, D);

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
}

////////////////////////////////////////////////////////////////////////
// Compute the quantity 
//             \f$d(\gamma)/dt * \Delta T = \Delta \gamma \f$ 
//             using Newton iterative root finder */
////////////////////////////////////////////////////////////////////////
double 
HypoElasticPlastic::computeDeltaGamma(const double& delT,
                                  const double& tolerance,
                                  const double& normTrialS,
                                  const MPMMaterial* matl,
                                  const particleIndex idx,
                                  PlasticityState* state)
{
  // Initialize constants
  double twothird = 2.0/3.0;
  double stwothird = sqrt(twothird);
  double sthreetwo = 1.0/stwothird;
  double twomu = 2.0*state->shearModulus;
  double ep = state->plasticStrain;

  // Initialize variables
  //localState->plasticStrainRate = 0.0;
  double deltaGamma = state->plasticStrainRate*delT*sthreetwo;
  double deltaGammaOld = deltaGamma;
  double g = 0.0;
  double Dg = 1.0;

  // iterate
  int count = 0;
  do {

    ++count;

    // Compute sigma_y
    double sigma_y = d_plastic->computeFlowStress(state, delT, tolerance, 
                                                  matl, idx);

    // Update the yield stress in the local state
    state->yieldStress = sigma_y;

    // Compute g
    g = normTrialS - stwothird*sigma_y - twomu*deltaGamma;

    // Compute d(sigma_y)/d(epdot)
    double dsigy_depdot = d_plastic->evalDerivativeWRTStrainRate(state,
                                                                 idx);

    // Compute d(sigma_y)/d(ep)
    double dsigy_dep = d_plastic->evalDerivativeWRTPlasticStrain(state,
                                                                 idx);

    // Compute d(g)/d(deltaGamma)
    Dg = -twothird*(dsigy_depdot/delT + dsigy_dep) - twomu;

    // Update deltaGamma
    deltaGammaOld = deltaGamma;
    deltaGamma -= g/Dg;

    // Update local plastic strain rate
    double stt_deltaGamma = max(stwothird*deltaGamma, 0.0);
    state->plasticStrainRate = stt_deltaGamma/delT;
    
    // Update local plastic strain 
    state->plasticStrain = ep + stt_deltaGamma;

    if (std::isnan(g) || std::isnan(deltaGamma)) {
      cout << "idx = " << idx << " iter = " << count 
         << " g = " << g << " Dg = " << Dg << " deltaGamma = " << deltaGamma 
         << " sigy = " << sigma_y 
         << " dsigy/depdot = " << dsigy_depdot << " dsigy/dep = " << dsigy_dep 
         << " epdot = " << state->plasticStrainRate 
         << " ep = " << state->plasticStrain << endl;
      throw ParameterNotFound("**ERROR**:ElasticPlastic: Found nan.", __FILE__, __LINE__);
    }

    if (fabs(deltaGamma-deltaGammaOld) < tolerance || count > 100) break;

  } while (fabs(g) > tolerance);

  return deltaGamma;
}

/*! Compute the elastic tangent modulus tensor for isotropic
    materials
    Assume: [stress] = [s11 s22 s33 s23 s31 s12]
            [strain] = [e11 e22 e33 2e23 2e31 2e12] 
*/
void 
HypoElasticPlastic::computeElasticTangentModulus(const double& K,
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
HypoElasticPlastic::computeEPlasticTangentModulus(const double& K,
                                                  const double& mu,
                                                  const double& delGamma,
                                                  const double& normTrialS,
                                                  const particleIndex idx,
                                                  const Matrix3& n,
                                                  PlasticityState* state,
                                                  double Cep[6][6])
{
  // Compute theta and theta_bar
  double twomu = 2.0*mu;
  double theta = 1.0 - (twomu*delGamma)/normTrialS;
  double dsigYdep = d_plastic->evalDerivativeWRTPlasticStrain(state, idx);
  double thetabar = 1.0/(1.0 + dsigYdep/(3.0*mu)) - (1.0 - theta);

  // Form the elastic-plastic tangent modulus tensor
  double twomu3 = twomu/3.0;
  double twomu3theta = twomu3*theta;
  double kfourmu3theta = K + 2.0*twomu3theta;
  double twomutheta = twomu*theta;
  double ktwomu3theta = K - twomu3theta;
  double twomuthetabar = twomu*thetabar;
  double twomuthetabarn11 = twomuthetabar*n(0,0);
  double twomuthetabarn22 = twomuthetabar*n(1,1);
  double twomuthetabarn33 = twomuthetabar*n(2,2);
  double twomuthetabarn23 = twomuthetabar*n(1,2);
  double twomuthetabarn31 = twomuthetabar*n(2,0);
  double twomuthetabarn12 = twomuthetabar*n(0,1);

  Cep[0][0] = kfourmu3theta - twomuthetabarn11*n(0,0); 
  Cep[0][1] = ktwomu3theta - twomuthetabarn11*n(1,1);
  Cep[0][2] = ktwomu3theta - twomuthetabarn11*n(2,2);
  Cep[0][3] = -0.5*twomuthetabarn11*n(1,2);
  Cep[0][4] = -0.5*twomuthetabarn11*n(2,0);
  Cep[0][5] = -0.5*twomuthetabarn11*n(0,1);
  
  Cep[1][0] = Cep[0][1];
  Cep[1][1] = kfourmu3theta - twomuthetabarn22*n(1,1); 
  Cep[1][2] = ktwomu3theta - twomuthetabarn22*n(2,2);
  Cep[1][3] = -0.5*twomuthetabarn22*n(1,2);
  Cep[1][4] = -0.5*twomuthetabarn22*n(2,0);
  Cep[1][5] = -0.5*twomuthetabarn22*n(0,1);

  Cep[2][0] = Cep[0][2];
  Cep[2][1] = Cep[1][2];
  Cep[2][2] = kfourmu3theta - twomuthetabarn33*n(2,2); 
  Cep[2][3] = -0.5*twomuthetabarn33*n(1,2);
  Cep[2][4] = -0.5*twomuthetabarn33*n(2,0);
  Cep[2][5] = -0.5*twomuthetabarn33*n(0,1);

  Cep[3][0] = Cep[0][3];
  Cep[3][1] = Cep[1][3];
  Cep[3][2] = Cep[2][3];
  Cep[3][3] =  0.5*(twomutheta - twomuthetabarn23*n(1,2)); 
  Cep[3][4] = -0.5*twomuthetabarn23*n(2,0);
  Cep[3][5] = -0.5*twomuthetabarn23*n(0,1);

  Cep[4][0] = Cep[0][4];
  Cep[4][1] = Cep[1][4];
  Cep[4][2] = Cep[2][4];
  Cep[4][3] = Cep[3][4];
  Cep[4][4] =  0.5*(twomutheta - twomuthetabarn31*n(2,0)); 
  Cep[4][5] = -0.5*twomuthetabarn31*n(0,1);

  Cep[5][0] = Cep[0][5];
  Cep[5][1] = Cep[1][5];
  Cep[5][2] = Cep[2][5];
  Cep[5][3] = Cep[3][5];
  Cep[5][4] = Cep[4][5];
  Cep[5][5] =  0.5*(twomutheta - twomuthetabarn12*n(0,1)); 
}

/*! Compute K matrix */
void 
HypoElasticPlastic::computeStiffnessMatrix(const double B[6][24],
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
HypoElasticPlastic::BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
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
HypoElasticPlastic::carryForward(const PatchSubset* patches,
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
    constParticleVariable<double>  pPlasticTemp, pPlasticTempInc;

    old_dw->get(pLeftStretch,    pLeftStretchLabel,    pset);
    old_dw->get(pRotation,       pRotationLabel,       pset);
    old_dw->get(pStrainRate,     pStrainRateLabel,     pset);
    old_dw->get(pPlasticStrain,  pPlasticStrainLabel,  pset);
    old_dw->get(pDamage,         pDamageLabel,         pset);
    old_dw->get(pPorosity,       pPorosityLabel,       pset);
    old_dw->get(pLocalized,      pLocalizedLabel,      pset);
    old_dw->get(pPlasticTemp,    pPlasticTempLabel,    pset);
    old_dw->get(pPlasticTempInc, pPlasticTempIncLabel, pset);

    ParticleVariable<Matrix3>      pLeftStretch_new, pRotation_new;
    ParticleVariable<double>       pPlasticStrain_new, pDamage_new, 
      pPorosity_new, pStrainRate_new;
    ParticleVariable<int>          pLocalized_new;
    ParticleVariable<double>       pPlasticTemp_new, pPlasticTempInc_new;

    new_dw->allocateAndPut(pLeftStretch_new, 
                           pLeftStretchLabel_preReloc,            pset);
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pPorosity_new,      
                           pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pPlasticTemp_new,
                           pPlasticTempLabel_preReloc,            pset);
    new_dw->allocateAndPut(pPlasticTempInc_new,
                           pPlasticTempIncLabel_preReloc,         pset);

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
      pDamage_new[idx] = pDamage[idx];
      pPorosity_new[idx] = pPorosity[idx];
      pLocalized_new[idx] = pLocalized[idx];
      pPlasticTemp_new[idx] = pPlasticTemp[idx];
      pPlasticTempInc_new[idx] = 0.0;
    }

    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

void
HypoElasticPlastic::allocateCMDataAddRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patch,
                                              MPMLabel* lb) const
{
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patch);

  // Add requires local to this model
  task->requires(Task::NewDW, pLeftStretchLabel_preReloc,    matlset, gnone);
  task->requires(Task::NewDW, pRotationLabel_preReloc,       matlset, gnone);
  task->requires(Task::NewDW, pStrainRateLabel_preReloc,     matlset, gnone);
  task->requires(Task::NewDW, pPlasticStrainLabel_preReloc,  matlset, gnone);
  task->requires(Task::NewDW, pDamageLabel_preReloc,         matlset, gnone);
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,      matlset, gnone);
  task->requires(Task::NewDW, pPorosityLabel_preReloc,       matlset, gnone);
  task->requires(Task::NewDW, pPlasticTempLabel_preReloc,    matlset, gnone);
  task->requires(Task::NewDW, pPlasticTempIncLabel_preReloc, matlset, gnone);
  d_plastic->allocateCMDataAddRequires(task,matl,patch,lb);
}

void
HypoElasticPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
                                      ParticleSubset* addset,
                                      map<const VarLabel*, 
                                      ParticleVariableBase*>* newState,
                                      ParticleSubset* delset,
                                      DataWarehouse* old_dw)
{
  // Copy the data common to all constitutive models from the particle to be 
  // deleted to the particle to be added. 
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);

  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
  ParticleSubset::iterator n,o;

  ParticleVariable<Matrix3> pLeftStretch, pRotation;
  ParticleVariable<double> pPlasticStrain, pDamage,pPorosity, pStrainRate;
  ParticleVariable<int> pLocalized;
  ParticleVariable<double> pPlasticTemperature, pPlasticTempInc;

  constParticleVariable<Matrix3> o_LeftStretch, o_Rotation;
  constParticleVariable<double> o_PlasticStrain, o_Damage,o_Porosity, 
    o_StrainRate;
  constParticleVariable<int> o_Localized;
  constParticleVariable<double> o_PlasticTemperature, o_PlasticTempInc;

  new_dw->allocateTemporary(pLeftStretch,addset);
  new_dw->allocateTemporary(pRotation,addset);
  new_dw->allocateTemporary(pPlasticStrain,addset);
  new_dw->allocateTemporary(pDamage,addset);
  new_dw->allocateTemporary(pStrainRate,addset);
  new_dw->allocateTemporary(pLocalized,addset);
  new_dw->allocateTemporary(pPorosity,addset);
  new_dw->allocateTemporary(pPlasticTemperature,addset);
  new_dw->allocateTemporary(pPlasticTempInc,addset);

  new_dw->get(o_LeftStretch,pLeftStretchLabel_preReloc,delset);
  new_dw->get(o_Rotation,pRotationLabel_preReloc,delset);
  new_dw->get(o_StrainRate,pStrainRateLabel_preReloc,delset);
  new_dw->get(o_PlasticStrain,pPlasticStrainLabel_preReloc,delset);
  new_dw->get(o_Damage,pDamageLabel_preReloc,delset);
  new_dw->get(o_Localized,pLocalizedLabel_preReloc,delset);
  new_dw->get(o_Porosity,pPorosityLabel_preReloc,delset);
  new_dw->get(o_PlasticTemperature,pPlasticTempLabel_preReloc,delset);
  new_dw->get(o_PlasticTempInc,pPlasticTempIncLabel_preReloc,delset);

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {

    pLeftStretch[*n] = o_LeftStretch[*o];
    pRotation[*n] = o_Rotation[*o];
    pStrainRate[*n] = o_StrainRate[*o];
    pPlasticStrain[*n] = o_PlasticStrain[*o];
    pDamage[*n] = o_Damage[*o];
    pLocalized[*n] = o_Localized[*o];
    pPorosity[*n] = o_Porosity[*o];
    pPlasticTemperature[*n] = o_PlasticTemperature[*o];
    pPlasticTempInc[*n] = o_PlasticTempInc[*o];
  }

  (*newState)[pLeftStretchLabel]=pLeftStretch.clone();
  (*newState)[pRotationLabel]=pRotation.clone();
  (*newState)[pStrainRateLabel]=pStrainRate.clone();
  (*newState)[pPlasticStrainLabel]=pPlasticStrain.clone();
  (*newState)[pDamageLabel]=pDamage.clone();
  (*newState)[pLocalizedLabel]=pLocalized.clone();
  (*newState)[pPorosityLabel]=pPorosity.clone();
  (*newState)[pPlasticTempLabel]=pPlasticTemperature.clone();
  (*newState)[pPlasticTempIncLabel]=pPlasticTempInc.clone();
  
  // Initialize the data for the plasticity model
  d_plastic->allocateCMDataAdd(new_dw,addset, newState, delset, old_dw);
}


void 
HypoElasticPlastic::getPlasticTemperatureIncrement(ParticleSubset* pset,
                                                   DataWarehouse* new_dw,
                                                   ParticleVariable<double>& T) 
{
  constParticleVariable<double> pPlasticTempInc;
  new_dw->get(pPlasticTempInc, pPlasticTempIncLabel_preReloc, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++) T[*iter] = pPlasticTempInc[*iter];
}

void
HypoElasticPlastic::addRequiresDamageParameter(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}

void
HypoElasticPlastic::getDamageParameter(const Patch* patch,
                                       ParticleVariable<int>& damage,
                                       int dwi,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  ParticleSubset* pset = old_dw->getParticleSubset(dwi,patch);
  constParticleVariable<int> pLocalized;
  new_dw->get(pLocalized, pLocalizedLabel_preReloc, pset);

  ParticleSubset::iterator iter;
  for (iter = pset->begin(); iter != pset->end(); iter++) {
    damage[*iter] = pLocalized[*iter];
  }
   
}
         
// Actually calculate rotation
void
HypoElasticPlastic::computeUpdatedVR(const double& delT,
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
HypoElasticPlastic::computeRateofRotation(const Matrix3& tensorV, 
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
HypoElasticPlastic::computeElasticTangentModulus(double bulk,
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

// Update the porosity of the material
double 
HypoElasticPlastic::updatePorosity(const Matrix3& D,
                                   double delT, 
                                   double f,
                                   double ep)
{
  // Growth
  // Calculate trace of D
  double Dkk = D.Trace();
  Matrix3 one; one.Identity();
  Matrix3 eta = D - one*(Dkk/3.0);

  // Calculate rate of growth
  double fdot_grow = 0.0;
  if (Dkk > 0.0) fdot_grow = (1.0-f)*Dkk;

  // Nucleation 
  // Calculate A
  double A = voidNucleationFactor(ep);

  // Calculate plastic strain rate
  double epdot = sqrt(eta.NormSquared()/1.5);

  // Calculate rate of nucleation
  double fdot_nucl = A*epdot;

  // Update void volume fraction using forward euler
  double f_new = f + delT*(fdot_nucl + fdot_grow);
  //cout << "Porosity: D = " << D << endl;
  //cout << "Porosity: eta = " << eta << endl;
  //cout << "Porosity: Dkk = " << Dkk << endl;
  //cout << "Porosity::fdot_gr = " << fdot_grow 
  //     << " fdot_nucl = " << fdot_nucl << " f = " << f 
  //     << " f_new = " << f_new << endl;
  return f_new;
}

// Calculate the void nucleation factor
double 
HypoElasticPlastic::voidNucleationFactor(double ep)
{
  double temp = (ep - d_porosity.en)/d_porosity.sn;
  double A = d_porosity.fn/(d_porosity.sn*sqrt(2.0*M_PI))*
    exp(-0.5*temp*temp);
  return A;
}

double
HypoElasticPlastic::computeRhoMicroCM(double pressure,
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
HypoElasticPlastic::computePressEOSCM(double rho_cur,double& pressure,
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
HypoElasticPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

void
HypoElasticPlastic::scheduleCheckNeedAddMPMMaterial(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* ) const
{
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pPlasticStrainLabel_preReloc,   matlset, gnone);

  task->computes(lb->NeedAddMPMMaterialLabel);
}

void
HypoElasticPlastic::checkNeedAddMPMMaterial(const PatchSubset* patches,
                                            const MPMMaterial* matl,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  if (cout_CST.active()) {
    cout_CST << getpid() << "checkNeedAddMPMMaterial: In : Matl = " << matl
             << " id = " << matl->getDWIndex() <<  " patch = "
             << (patches->get(0))->getID();
  }

  double need_add=0.;
                                                                                
  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<double> pPlasticStrain;
    new_dw->get(pPlasticStrain, pPlasticStrainLabel_preReloc, pset);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;
      if(pPlasticStrain[idx]>5.e-2){
        need_add = -1.;
      }
    }
  }

  new_dw->put(sum_vartype(need_add),     lb->NeedAddMPMMaterialLabel);
}



