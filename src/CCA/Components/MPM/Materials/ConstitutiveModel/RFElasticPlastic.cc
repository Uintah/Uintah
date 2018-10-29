/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/RFElasticPlastic.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/YieldConditionFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/StabilityCheckFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/FlowStressModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/MPMEquationOfStateFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ShearModulusModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/MeltingTempModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/SpecificHeatModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DevStressModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/PlasticityState.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DeformationState.h>


#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/SymmMatrix3.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/TangentModulusTensor.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <cmath>
#include <iostream>

#include <unistd.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>

using namespace std;
using namespace Uintah;

static DebugStream cout_EP("EP",false);
static DebugStream cout_EP1("EP1",false);
static DebugStream CSTi("EPi",false);
static DebugStream CSTir("EPir",false);

RFElasticPlastic::RFElasticPlastic(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  //********** Concentration Component****************************
  if(flag->d_doScalarDiffusion){
    ps->require("volume_expansion_coeff",d_initialData.vol_exp_coeff);
  }else{
    d_initialData.vol_exp_coeff = 0.0;
    ostringstream warn;
    warn << "RFElasticPlastic:: This Constitutive Model requires the use\n"
         << "of scalar diffusion" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  //********** Concentration Component****************************

  d_initialData.alpha = 0.0; // default is per K.  Only used in implicit code
  ps->get("coeff_thermal_expansion", d_initialData.alpha);
  d_initialData.Chi = 0.9;
  ps->get("taylor_quinney_coeff",d_initialData.Chi);
  d_initialData.sigma_crit = 2.0e99; // Make huge to do nothing by default
  ps->get("critical_stress", d_initialData.sigma_crit);

  d_tol = 1.0e-10;
  ps->get("tolerance",d_tol);

  d_useModifiedEOS = false;
  ps->get("useModifiedEOS",d_useModifiedEOS);

  d_initialMaterialTemperature = 294.0;
  ps->get("initial_material_temperature",d_initialMaterialTemperature);

  d_checkTeplaFailureCriterion = true;
  ps->get("check_TEPLA_failure_criterion",d_checkTeplaFailureCriterion);

  d_doMelting = true;
  ps->get("do_melting",d_doMelting);

  d_checkStressTriax = true;
  ps->get("check_max_stress_failure",d_checkStressTriax);
  
  // plasticity convergence Algorithm
  d_plasticConvergenceAlgo = "radialReturn";   // default
  bool usingRR = true;
  
  if(d_plasticConvergenceAlgo != "radialReturn"){
    ostringstream warn;
    warn << "RFElasticPlastic:: Invalid plastic_convergence_algo option ("
         << d_plasticConvergenceAlgo << ") Valid options are: radialReturn"
         << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  // 
  d_yield = YieldConditionFactory::create(ps, usingRR );
  if(!d_yield){
    ostringstream desc;
    desc << "An error occured in the YieldConditionFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Chris.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_stable = StabilityCheckFactory::create(ps);
  if(!d_stable) cerr << "Stability check disabled\n";

  d_flow = FlowStressModelFactory::create(ps);
  if(!d_flow){
    ostringstream desc;
    desc << "An error occured in the FlowModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Chris.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_eos = MPMEquationOfStateFactory::create(ps);
  d_eos->setBulkModulus(d_initialData.Bulk);
  if(!d_eos){
    ostringstream desc;
    desc << "An error occured in the EquationOfStateFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Jim.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_shear = ShearModulusModelFactory::create(ps);
  if (!d_shear) {
    ostringstream desc;
    desc << "RFElasticPlastic::Error in shear modulus model factory" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  d_melt = MeltingTempModelFactory::create(ps);
  if (!d_melt) {
    ostringstream desc;
    desc << "RFElasticPlastic::Error in melting temp model factory" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  d_devStress = DevStressModelFactory::create(ps);
  if (!d_devStress) {
    ostringstream desc;
    desc << "RFElasticPlastic::Error creating deviatoric stress model" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_computeSpecificHeat = false;
  ps->get("compute_specific_heat",d_computeSpecificHeat);
  d_Cp = SpecificHeatModelFactory::create(ps);
  
  //getSpecificHeatData(ps);
  initializeLocalMPMLabels();
}

RFElasticPlastic::~RFElasticPlastic()
{
  // Destructor 
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainRateLabel);
  VarLabel::destroy(pEnergyLabel);

  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pPlasticStrainRateLabel_preReloc);
  VarLabel::destroy(pEnergyLabel_preReloc);

  delete d_flow;
  delete d_yield;
  delete d_stable;
  delete d_eos;
  delete d_shear;
  delete d_melt;
  delete d_Cp;
  delete d_devStress;
}

//______________________________________________________________________
//
void RFElasticPlastic::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","rf_elastic_plastic");
  }
  
  cm_ps->appendElement("bulk_modulus",                  d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",                 d_initialData.Shear);
  //********** Concentration Component****************************
  if(flag->d_doScalarDiffusion){
    cm_ps->appendElement("volume_expansion_coeff",      d_initialData.vol_exp_coeff);
  }
  //********** Concentration Component****************************
  cm_ps->appendElement("coeff_thermal_expansion",       d_initialData.alpha);
  cm_ps->appendElement("taylor_quinney_coeff",          d_initialData.Chi);
  cm_ps->appendElement("critical_stress",               d_initialData.sigma_crit);
  cm_ps->appendElement("tolerance",                     d_tol);
  cm_ps->appendElement("useModifiedEOS",                d_useModifiedEOS);
  cm_ps->appendElement("initial_material_temperature",  d_initialMaterialTemperature);
  cm_ps->appendElement("check_TEPLA_failure_criterion", d_checkTeplaFailureCriterion);
  cm_ps->appendElement("do_melting",                    d_doMelting);
  cm_ps->appendElement("check_max_stress_failure",      d_checkStressTriax);
  cm_ps->appendElement("plastic_convergence_algo",      d_plasticConvergenceAlgo);
  cm_ps->appendElement("compute_specific_heat",         d_computeSpecificHeat);

  d_yield      ->outputProblemSpec(cm_ps);
  d_stable     ->outputProblemSpec(cm_ps);
  d_flow       ->outputProblemSpec(cm_ps);
  d_devStress  ->outputProblemSpec(cm_ps);
  d_eos        ->outputProblemSpec(cm_ps);
  d_shear      ->outputProblemSpec(cm_ps);
  d_melt       ->outputProblemSpec(cm_ps);
  d_Cp         ->outputProblemSpec(cm_ps);
}


RFElasticPlastic* RFElasticPlastic::clone()
{
  return scinew RFElasticPlastic(*this);
}

//______________________________________________________________________
//
void
RFElasticPlastic::initializeLocalMPMLabels()
{
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainRateLabel = VarLabel::create("p.plasticStrainRate",
    ParticleVariable<double>::getTypeDescription());
  pEnergyLabel = VarLabel::create("p.energy",
    ParticleVariable<double>::getTypeDescription());

  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainRateLabel_preReloc = VarLabel::create("p.plasticStrainRate+",
    ParticleVariable<double>::getTypeDescription());
  pEnergyLabel_preReloc = VarLabel::create("p.energy+",
    ParticleVariable<double>::getTypeDescription());
}

//______________________________________________________________________
//
void 
RFElasticPlastic::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pPlasticStrainLabel);
  from.push_back(pPlasticStrainRateLabel);
  from.push_back(pEnergyLabel);

  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pPlasticStrainRateLabel_preReloc);
  to.push_back(pEnergyLabel_preReloc);

  // Add the particle state for the flow & deviatoric stress model
  d_flow     ->addParticleState(from, to);
  d_devStress->addParticleState(from, to);
}
//______________________________________________________________________
//
void 
RFElasticPlastic::addInitialComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(pPlasticStrainLabel, matlset);
  task->computes(pPlasticStrainRateLabel, matlset);
  task->computes(pEnergyLabel,        matlset);
 
  // Add internal evolution variables computed by flow & deviatoric stress model
  d_flow     ->addInitialComputesAndRequires(task, matl, patch);
  d_devStress->addInitialComputesAndRequires(task, matl);
}
//______________________________________________________________________
//
void 
RFElasticPlastic::initializeCMData(const Patch* patch,
                                 const MPMMaterial* matl,
                                 DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);
  computeStableTimeStep(patch, matl, new_dw);

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  //cout << "Initialize CM Data in RFElasticPlastic" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pPlasticStrain, pPlasticStrainRate, pEnergy;

  new_dw->allocateAndPut(pPlasticStrain,     pPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pPlasticStrainRate, pPlasticStrainRateLabel, pset);
  new_dw->allocateAndPut(pEnergy,            pEnergyLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    pPlasticStrain[*iter] = 0.0;
    pPlasticStrainRate[*iter] = 0.0;
    pEnergy[*iter] = 0.;
  }

  // Initialize the data for the flow model
  d_flow->initializeInternalVars(pset, new_dw);
  
  // Deviatoric Stress Model
  d_devStress->initializeInternalVars(pset, new_dw);
  
}
//______________________________________________________________________
//
void 
RFElasticPlastic::computeStableTimeStep(const Patch* patch,
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
//______________________________________________________________________
//
void 
RFElasticPlastic::addComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 
  task->requires(Task::OldDW, pPlasticStrainLabel,    matlset, gnone);
  task->requires(Task::OldDW, pPlasticStrainRateLabel,matlset, gnone);
  task->requires(Task::OldDW, lb->pLocalizedMPMLabel, matlset, gnone);
  task->requires(Task::OldDW, lb->pParticleIDLabel,   matlset, gnone);
  task->requires(Task::OldDW, pEnergyLabel,           matlset, gnone);

  //********** Concentration Component****************************
  if(flag->d_doScalarDiffusion){
    task->requires(Task::OldDW, lb->diffusion->pConcPrevious, matlset, gnone);
    task->requires(Task::OldDW, lb->diffusion->pConcentration, matlset, gnone);
  }
  //********** Concentration Component****************************

  task->computes(pPlasticStrainLabel_preReloc,      matlset);
  task->computes(pPlasticStrainRateLabel_preReloc,  matlset);
  task->computes(lb->pLocalizedMPMLabel_preReloc,   matlset);
  task->computes(pEnergyLabel_preReloc,             matlset);

  // Add internal evolution variables computed by flow model
  d_flow->addComputesAndRequires(task, matl, patches);
  
  // Deviatoric stress model
  d_devStress->addComputesAndRequires(task, matl);

}
//______________________________________________________________________
//
void 
RFElasticPlastic::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  if (cout_EP.active()) {
    cout_EP << getpid() 
            << " RFElasticPlastic:ComputeStressTensor:Explicit"
            << " Matl = " << matl 
            << " DWI = " << matl->getDWIndex() 
            << " patch = " << (patches->get(0))->getID();
  }

  //*********Start - Used for testing purposes - CG *******
  // int timestep = d_materialManager->getCurrentTopLevelTimeStep();
  //*********End   - Used for testing purposes - CG *******

  // General stuff
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  Matrix3 tensorD(0.0);                   // Rate of deformation
  Matrix3 tensorW(0.0);                   // Spin 
  Matrix3 tensorF; tensorF.Identity();    // Deformation gradient
  Matrix3 tensorU; tensorU.Identity();    // Right Cauchy-Green stretch
  Matrix3 tensorR; tensorR.Identity();    // Rotation 
  Matrix3 sigma(0.0);                     // The Cauchy stress
  Matrix3 tensorEta(0.0);                 // Deviatoric part of tensor D
  Matrix3 tensorS(0.0);                   // Devaitoric part of tensor Sig
  Matrix3 tensorF_new; tensorF_new.Identity(); // Deformation gradient

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double rho_0 = matl->getInitialDensity();
  double Tm    = matl->getMeltTemperature();
  double sqrtThreeTwo = sqrt(1.5);
  //double sqrtTwoThird = 1.0/sqrtThreeTwo;

  //********** Concentration Component****************************
  double vol_exp_coeff = d_initialData.vol_exp_coeff;
  double concentration = 0.0;
  double concentration_pn = 0.0;
  double conc_rate = 0.0;
  //********** Concentration Component****************************
  
  double totalStrainEnergy = 0.0;
//  double include_AV_heating=0.0;
//  if (flag->d_artificial_viscosity_heating) {
//    include_AV_heating=1.0;
//  }

  // Loop thru patches
  for(int patchIndex=0; patchIndex<patches->size(); patchIndex++){
    const Patch* patch = patches->get(patchIndex);

    // Get grid size
    Vector dx = patch->dCell();

    // Get the set of particles
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the particle location,  particle mass, particle volume, etc.
    constParticleVariable<double> pMass;
    constParticleVariable<double> pVolume;
    constParticleVariable<double> pTemperature;
    constParticleVariable<Vector> pVelocity;
    constParticleVariable<Matrix3> pDeformGrad;
    constParticleVariable<Matrix3> pStress;

    //********** Concentration Component****************************
    constParticleVariable<double> pConcentration;
    constParticleVariable<double> pConc_prenew;
    //********** Concentration Component****************************

    old_dw->get(pMass,        lb->pMassLabel,               pset);
    old_dw->get(pVolume,      lb->pVolumeLabel,             pset);
    old_dw->get(pTemperature, lb->pTemperatureLabel,        pset);
    old_dw->get(pVelocity,    lb->pVelocityLabel,           pset);
    old_dw->get(pStress,      lb->pStressLabel,             pset);
    old_dw->get(pDeformGrad,  lb->pDeformationMeasureLabel, pset);

    //********** Concentration Component****************************
    if(flag->d_doScalarDiffusion){
      old_dw->get(pConcentration, lb->diffusion->pConcentration, pset);
      old_dw->get(pConc_prenew,   lb->diffusion->pConcPrevious,  pset);
    }
    //********** Concentration Component****************************

    constParticleVariable<double> pPlasticStrain;
    constParticleVariable<double> pPlasticStrainRate, pEnergy;
    constParticleVariable<int> pLocalized;

    old_dw->get(pPlasticStrain,     pPlasticStrainLabel,     pset);
    old_dw->get(pPlasticStrainRate, pPlasticStrainRateLabel, pset);
    old_dw->get(pEnergy,            pEnergyLabel,            pset);
    old_dw->get(pLocalized,         lb->pLocalizedMPMLabel,  pset);

    // Get the particle IDs, useful in case a simulation goes belly up
    constParticleVariable<long64> pParticleID; 
    old_dw->get(pParticleID, lb->pParticleIDLabel, pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    constParticleVariable<Matrix3> pDeformGrad_new, velGrad;
    constParticleVariable<double> pVolume_deformed;
    new_dw->get(pDeformGrad_new,  lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,            pset);
    new_dw->get(pVolume_deformed, lb->pVolumeLabel_preReloc,             pset);

    // Create and allocate arrays for storing the updated information
    ParticleVariable<double>  pPlasticStrain_new; 
    ParticleVariable<double>  pPlasticStrainRate_new;
    ParticleVariable<int>     pLocalized_new;
    ParticleVariable<double>  pdTdt, p_q, pEnergy_new;
    ParticleVariable<Matrix3> pStress_new;

    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pPlasticStrainRate_new,      
                           pPlasticStrainRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           lb->pLocalizedMPMLabel_preReloc,       pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);

    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel,                 pset);
    new_dw->allocateAndPut(p_q,   lb->p_qLabel_preReloc,          pset);
    new_dw->allocateAndPut(pEnergy_new, pEnergyLabel_preReloc,    pset);

    d_flow     ->getInternalVars(pset, old_dw);
    d_devStress->getInternalVars(pset, old_dw);
    
    d_flow     ->allocateAndPutInternalVars(pset, new_dw);
    d_devStress->allocateAndPutInternalVars(pset, new_dw);

    //______________________________________________________________________
    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero int. heating by default, modify with appropriate sources
      // This has units (in MKS) of K/s  (i.e. temperature/time)
      pdTdt[idx] = 0.0;

      Matrix3 tensorL=velGrad[idx];

      // Carry forward the pLocalized tag for now, alter below
      pLocalized_new[idx] = pLocalized[idx];

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      double J = pDeformGrad_new[idx].Determinant();
      tensorF_new=pDeformGrad_new[idx];

      if(!(J > 0.) || J > 1.e5){
          cerr << "**ERROR** Negative (or huge) Jacobian of deformation gradient."
               << "  Deleting particle " << pParticleID[idx] << endl;
          cerr << "l = " << tensorL << endl;
          cerr << "F_old = " << pDeformGrad[idx] << endl;
          cerr << "J_old = " << pDeformGrad[idx].Determinant() << endl;
          cerr << "F_new = " << tensorF_new << endl;
          cerr << "J = " << J << endl;
          cerr << "Temp = " << pTemperature[idx] << endl;
          cerr << "Tm = " << Tm << endl;
          cerr << "DWI = " << matl->getDWIndex() << endl;
          cerr << "L.norm()*dt = " << tensorL.Norm()*delT << endl;
          pLocalized_new[idx]=-999;

          tensorL=zero;
          tensorF_new.Identity();
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;

      //********** Concentration Component****************************
      // Compute rate of concentration
      if(flag->d_doScalarDiffusion){
        concentration = pConcentration[idx];
        concentration_pn = pConc_prenew[idx];
        conc_rate = (concentration - concentration_pn)/delT;
      }
      //********** Concentration Component****************************

      // Calculate rate of deformation tensor (D)
      tensorD = (tensorL + tensorL.Transpose())*0.5;

      // Compute polar decomposition of F (F = RU)
      pDeformGrad[idx].polarDecompositionRMB(tensorU, tensorR);

      // Rotate the total rate of deformation tensor back to the 
      // material configuration
      tensorD = (tensorR.Transpose())*(tensorD*tensorR);

      //********** Concentration Component****************************
      // Remove concentration dependent portion of rate of deformation 
      // if(timestep < 20000)
      //   conc_rate = 1.0;
      // else
      //   conc_rate = -1.0;

      if(flag->d_doScalarDiffusion){
        tensorD = tensorD - one * vol_exp_coeff * (conc_rate);
        //cout << "Concentration Rate: " << conc_rate << ", delT: " << delT << endl;
      }
      //********** Concentration Component****************************
      // Calculate the deviatoric part of the non-concentration part
      // of the rate of deformation tensor
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);
      
      // Rotate the Cauchy stress back to the 
      // material configuration and calculate the deviatoric part
      sigma = pStress[idx];
      sigma = (tensorR.Transpose())*(sigma*tensorR);
      double pressure = sigma.Trace()/3.0; 
      tensorS = sigma - one * pressure;

      // Rotate internal Cauchy stresses back to the 
      // material configuration (only for viscoelasticity)

      d_devStress->rotateInternalStresses(idx, tensorR);

      //double temperature = pTemperature[idx];

      // Set up the PlasticityState (for t_n+1)
      PlasticityState* state = scinew PlasticityState();
      //state->plasticStrain     = pPlasticStrain[idx];
      //state->plasticStrainRate = sqrtTwoThird*tensorEta.Norm();
      state->plasticStrainRate   = pPlasticStrainRate[idx];
      state->plasticStrain       = pPlasticStrain[idx] 
                                 + state->plasticStrainRate*delT;
      state->pressure            = pressure;
      //********** Concentration Component****************************
      // Used to cancel out temperature component of JohnsonCook
      // flowstress model
      state->temperature         = 298.0;
      //********** Concentration Component****************************
      state->initialTemperature  = d_initialMaterialTemperature;
      state->density             = rho_cur;
      state->initialDensity      = rho_0;
      state->volume              = pVolume_deformed[idx];
      state->initialVolume       = pMass[idx]/rho_0;
      state->bulkModulus         = bulk ;
      state->initialBulkModulus  = bulk;
      state->shearModulus        = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp         = Tm ;
      state->initialMeltTemp     = Tm;
      state->specificHeat        = matl->getSpecificHeat();
      state->energy              = pEnergy[idx];
      
      // Get or compute the specific heat
      if (d_computeSpecificHeat) {
        double C_p = d_Cp->computeSpecificHeat(state);
        state->specificHeat = C_p;
      }
    
      //********** Concentration Component****************************
      // --For the time being temperature dependent shear is not being used
      //
      // // Calculate the shear modulus and the melting temperature at the
      // // start of the time step and update the plasticity state
      // double Tm_cur = d_melt->computeMeltingTemp(state);
      // state->meltingTemp = Tm_cur ;
      //********** Concentration Component****************************
      
      double mu_cur = d_shear->computeShearModulus(state);
      state->shearModulus = mu_cur ;

      // compute the local sound wave speed
      double c_dil = sqrt((bulk + 4.0*mu_cur/3.0)/rho_cur);

      //-----------------------------------------------------------------------
      // Stage 2:
      //-----------------------------------------------------------------------
      // Assume elastic deformation to get a trial deviatoric stress
      // This is simply the previous timestep deviatoric stress plus a
      // deviatoric elastic increment based on the shear modulus supplied by
      // the strength routine in use.
      DeformationState* defState = scinew DeformationState();
      defState->tensorD    = tensorD;
      defState->tensorEta  = tensorEta;
      defState->viscoElasticWorkRate = 0.0;
      
      d_devStress->computeDeviatoricStressInc(idx, state, defState, delT);

      Matrix3 trialS = tensorS + defState->devStressInc;

      // Calculate the equivalent stress
      // this will be removed next, 
      // it should be computed in the flow stress routine
      // the flow stress routines should be passed
      // the entire stress (not just deviatoric)
      double equivStress = sqrtThreeTwo*trialS.Norm();

      // Calculate flow stress
      double flowStress = d_flow->computeFlowStress(state, delT, d_tol, 
                                                    matl, idx);
      state->yieldStress = flowStress;

      bool plastic = false;
      //********** Concentration Component****************************
      // // Material has melted if flowStress <= 0.0
      // bool melted  = false;
      // bool plastic = false;
      // if (temperature > Tm_cur || flowStress <= 0.0) {
      //
      //   melted = true;
      //   // Set the deviatoric stress to zero
      //   if (d_doMelting){
      //      tensorS = 0.0;
      //   } else {
      //      cerr << "The material has exceed the melt temperature, but you haven't turned \n";
      //      cerr << "melting on.  RFElasticPlastic does nonsensical things here.  You should \n";
      //      cerr << "probably either set <do_melting>true</do_melting>, or increase the material\n";
      //      cerr << "<melt_temp> to a level that won't be exceeded.\n";
      //   }
      //
      //   d_flow->updateElastic(idx);
      //
      // } else {
      //********** Concentration Component****************************

        // Get the current porosity 
        double porosity = 0.0;

        // Evaluate yield condition
        double traceOfTrialStress = 3.0*pressure + 
                                        tensorD.Trace()*(2.0*mu_cur*delT);

        double flow_rule = d_yield->evalYieldCondition(equivStress, flowStress,
                                                       traceOfTrialStress, 
                                                       porosity, state->yieldStress);
        // Compute the deviatoric stress
        /*
        cout << "flow_rule = " << flow_rule << " s_eq = " << equivStress
             << " s_flow = " << flowStress << endl;
        */

        //if (timestep == 10000)
                                //  cout << "Index: " << idx << ", Equiv: " << equivStress << endl;

        if (flow_rule < 0.0) {
          // Set the deviatoric stress to the trial stress
          tensorS = trialS;

          // Update the internal variables
          d_flow->updateElastic(idx);

          // Update internal Cauchy stresses (only for viscoelasticity)
          Matrix3 dp = zero;
          d_devStress->updateInternalStresses(idx, dp, defState, delT);

        } else {
          plastic = true;
          double delGamma = 0.0;

          // If the material goes plastic in the first step, or
          // gammadotplus < 0 or delGamma < 0 use the Simo algorithm
          // with Newton iterations.

           //  Here set to true, if all conditionals are met (immediately above) then set to false.
          bool doRadialReturn = true;
          Matrix3 tensorEtaPlasticInc = zero;
          //__________________________________
          //
          if (doRadialReturn) {
            // Compute Stilde using Newton iterations a la Simo
            state->plasticStrain     = pPlasticStrain[idx];
            Matrix3 nn(0.0);
            computePlasticStateViaRadialReturn(trialS, delT, matl, idx, state, nn, delGamma);

            tensorEtaPlasticInc = nn * delGamma;
            tensorS = trialS - tensorEtaPlasticInc *(2.0 * state->shearModulus);
          }

          // Update internal variables
          d_flow->updatePlastic(idx, delGamma);
          
          // Update internal Cauchy stresses (only for viscoelasticity)
          Matrix3 dp = tensorEtaPlasticInc/delT;
          d_devStress->updateInternalStresses(idx, dp, defState, delT);

        } // end of flow_rule if
      //********** Concentration Component****************************
      // } // end of temperature if
      //********** Concentration Component****************************

      // Calculate the updated hydrostatic stress
      double p = d_eos->computePressure(matl, state, tensorF_new, tensorD,delT);

      //********** Concentration Component****************************
      // -- not used currently in model
      // double Dkk = tensorD.Trace();
      // double dTdt_isentropic = d_eos->computeIsentropicTemperatureRate(
      //                                           temperature,rho_0,rho_cur,Dkk);
      // pdTdt[idx] += dTdt_isentropic;
      //
      // // Calculate Tdot from viscoelasticity
      // double taylorQuinney = d_initialData.Chi;
      // double fac = taylorQuinney/(rho_cur*state->specificHeat);
      // double Tdot_VW = defState->viscoElasticWorkRate*fac;
      //
      // pdTdt[idx] += Tdot_VW;
      //
      // double de_s=0.;
      // if (flag->d_artificial_viscosity) {
      //   double c_bulk = sqrt(bulk/rho_cur);
      //   double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
      //   p_q[idx] = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
      //   de_s     = -p_q[idx]*Dkk/rho_cur;
      // } else {
      //  p_q[idx] = 0.;
      //  de_s     = 0.;
      // }
      //
      // // Calculate Tdot due to artificial viscosity
      // double Tdot_AV = de_s/state->specificHeat;
      // pdTdt[idx] += Tdot_AV*include_AV_heating;
      //********** Concentration Component****************************

      Matrix3 tensorHy = one*p;
   
      // Calculate the total stress
      sigma = tensorS + tensorHy;


      //-----------------------------------------------------------------------
      // Stage 3:
      //-----------------------------------------------------------------------

      // // Compute porosity/temperature change
      if (!plastic) {
      
        // Save the updated data
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pPlasticStrainRate_new[idx] = 0.0;
      } else {
        // Update the plastic strain
        pPlasticStrain_new[idx]     = state->plasticStrain;
        pPlasticStrainRate_new[idx] = state->plasticStrainRate;
      }

      /**
      //-----------------------------------------------------------------------
      // Stage 4:
      //-----------------------------------------------------------------------
      // Find if the particle has failed/localized
      bool isLocalized = false;
      double tepla = 0.0;
      if (flag->d_doErosion) {

        // Check 1: Look at the temperature
        if (melted) isLocalized = true;

        // Check 2 and 3: Look at TEPLA and stability
        else if (plastic) {

          // Check 3: Stability criterion (only if material is plastic)
          if (d_stable->doIt() && !isLocalized) {

            // Calculate values needed for tangent modulus calculation
            state->temperature = temperature;
            Tm_cur = d_melt->computeMeltingTemp(state);
            state->meltingTemp = Tm_cur ;
            mu_cur = d_shear->computeShearModulus(state);
            state->shearModulus = mu_cur ;
            double sigY = d_flow->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);
            if (!(sigY > 0.0)) isLocalized = true;
            else {
              double dsigYdep = 
                d_flow->evalDerivativeWRTPlasticStrain(state, idx);
              double A = voidNucleationFactor(state->plasticStrain);

              // Calculate the elastic tangent modulus
              TangentModulusTensor Ce;
              computeElasticTangentModulus(bulk, mu_cur, Ce);
  
              // Calculate the elastic-plastic tangent modulus
              TangentModulusTensor Cep;
              d_yield->computeElasPlasTangentModulus(Ce, sigma, sigY, 
                                                     dsigYdep, 
                                                     pPorosity_new[idx],
                                                     A, Cep);
          
              // Initialize localization direction
              Vector direction(0.0,0.0,0.0);
              isLocalized = d_stable->checkStability(sigma, tensorD, Cep, 
                                                     direction);
            }
          }
        } 

        // Check 4: Look at maximum stress
        if (d_checkStressTriax) {

          // Compute eigenvalues of the stress tensor
          SymmMatrix3 stress(sigma);          
          Vector eigVal(0.0, 0.0, 0.0);
          Matrix3 eigVec;
          stress.eigen(eigVal, eigVec);
          
          double max_stress = Max(Max(eigVal[0],eigVal[1]), eigVal[2]);
          if (max_stress > d_initialData.sigma_crit) {
            isLocalized = true;
          }
        }

        // Use erosion algorithms to treat newly localized particles
        if (isLocalized) {

          // If the localized particles fail again then set their stress to zero
          if (pLocalized[idx]) {

          } else {
            // set the particle localization flag to true  
            pLocalized_new[idx] = 1;

            // Apply various erosion algorithms
            if (d_allowNoTension){
              if (p > 0.0){
                sigma = zero;
              }
              else{
                sigma = tensorHy;
              }
            }
            else if (d_allowNoShear){
              sigma = tensorHy;
            }
            else if (d_setStressToZero){
              sigma = zero;
            }
          }
        }
      }
      **/

      //-----------------------------------------------------------------------
      // Stage 5:
      //-----------------------------------------------------------------------

      // Rotate the stress back to the laboratory coordinates using new R
      // Compute polar decomposition of new F (F = RU)
      tensorF_new.polarDecompositionRMB(tensorU, tensorR);

      sigma = (tensorR*sigma)*(tensorR.Transpose());

      // Rotate internal Cauchy stresses back to laboratory
      // coordinates (only for viscoelasticity)

      d_devStress->rotateInternalStresses(idx, tensorR);

      // Save the new data
      pStress_new[idx] = sigma;
        
      // Rotate the deformation rate back to the laboratory coordinates
      tensorD = (tensorR*tensorD)*(tensorR.Transpose());

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        Matrix3 avgStress = (pStress_new[idx] + pStress[idx])*0.5;
        double avgVolume  = (pVolume_deformed[idx]+pVolume[idx])*0.5;
        
        double pSpecificStrainEnergy = (tensorD(0,0)*avgStress(0,0) +
                                        tensorD(1,1)*avgStress(1,1) +
                                        tensorD(2,2)*avgStress(2,2) +
                                   2.0*(tensorD(0,1)*avgStress(0,1) + 
                                        tensorD(0,2)*avgStress(0,2) +
                                        tensorD(1,2)*avgStress(1,2)))*
                                        avgVolume*delT/pMass[idx];

        // Compute rate of change of specific volume
//      double Vdot = (pVolume_deformed[idx] - pVolume[idx])/(pMass[idx]*delT);

        pEnergy_new[idx] = pEnergy[idx] + pSpecificStrainEnergy;
//                                      - p_q[idx]*Vdot*delT*include_AV_heating;

        totalStrainEnergy += pSpecificStrainEnergy*pMass[idx];
      }else{
        pEnergy_new[idx] = pEnergy[idx];
      }

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
      
      delete defState;
      delete state;
    }  // end particle loop

    //__________________________________
    //
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }
  }

  if (cout_EP.active()) 
    cout_EP << getpid() << "... End." << endl;

}

////////////////////////////////////////////////////////////////////////
/*! \brief Compute Stilde, epdot, ep, and delGamma using 
  Simo's approach */
////////////////////////////////////////////////////////////////////////
void 
RFElasticPlastic::computePlasticStateViaRadialReturn(const Matrix3& trialS,
                                                     const double& delT,
                                                     const MPMMaterial* matl,
                                                     const particleIndex idx,
                                                     PlasticityState* state,
                                                     Matrix3& nn,
                                                     double& delGamma)
{
  double normTrialS = trialS.Norm();
  
  // Do Newton iteration to compute delGamma and updated 
  // plastic strain, plastic strain rate, and yield stress
  double tolerance = min(delT, 1.0e-6);
  delGamma = computeDeltaGamma(delT, tolerance, normTrialS, matl, idx, state);
                               
  nn  = trialS/normTrialS;
}

////////////////////////////////////////////////////////////////////////
// Compute the quantity 
//             \f$d(\gamma)/dt * \Delta T = \Delta \gamma \f$ 
//             using Newton iterative root finder */
////////////////////////////////////////////////////////////////////////
double 
RFElasticPlastic::computeDeltaGamma(const double& delT,
                                    const double& tolerance,
                                    const double& normTrialS,
                                    const MPMMaterial* matl,
                                    const particleIndex idx,
                                    PlasticityState* state)
{
  // Initialize constants
  double twothird  = 2.0/3.0;
  double stwothird = sqrt(twothird);
  double sthreetwo = 1.0/stwothird;
  double twomu     = 2.0*state->shearModulus;

  // Initialize variables
  double ep         = state->plasticStrain;
  double sigma_y    = state->yieldStress;
  double deltaGamma = state->plasticStrainRate * delT * sthreetwo;
  double deltaGammaOld = deltaGamma;
  double g  = 0.0;
  double Dg = 1.0;

  //__________________________________
  // iterate
  int count = 0;
  do {

    ++count;

    // Compute the yield stress
    sigma_y = d_flow->computeFlowStress(state, delT, tolerance, 
                                           matl, idx);

    // Compute g
    g = normTrialS - stwothird*sigma_y - twomu*deltaGamma;

    // Compute d(sigma_y)/d(epdot)
    double dsigy_depdot = d_flow->evalDerivativeWRTStrainRate(state,
                                                                 idx);

    // Compute d(sigma_y)/d(ep)
    double dsigy_dep = d_flow->evalDerivativeWRTPlasticStrain(state,
                                                                 idx);

    // Compute d(g)/d(deltaGamma)
    Dg = -twothird*(dsigy_depdot/delT + dsigy_dep) - twomu;

    // Update deltaGamma
    deltaGammaOld = deltaGamma;
    deltaGamma -= g/Dg;

    if (std::isnan(g) || std::isnan(deltaGamma)) {
      cout << "idx = " << idx << " iter = " << count 
           << " g = " << g << " Dg = " << Dg << " deltaGamma = " << deltaGamma 
           << " sigy = " << sigma_y 
           << " dsigy/depdot = " << dsigy_depdot << " dsigy/dep= " << dsigy_dep 
           << " epdot = " << state->plasticStrainRate 
           << " ep = " << state->plasticStrain
           << " normTrialS = " << normTrialS << endl;
      throw InternalError("nans in computation",__FILE__,__LINE__);
    }

    // Update local plastic strain rate
    double stt_deltaGamma    = max(stwothird*deltaGamma, 0.0);
    state->plasticStrainRate = stt_deltaGamma/delT;

    // Update local plastic strain 
    state->plasticStrain = ep + stt_deltaGamma;

    if (fabs(deltaGamma-deltaGammaOld) < tolerance || count > 100) break;

  } while (fabs(g) > sigma_y/1000.);

  // Compute the yield stress
  state->yieldStress = d_flow->computeFlowStress(state, delT, tolerance, 
                                                    matl, idx);

  if (std::isnan(state->yieldStress)) {
    cout << "idx = " << idx << " iter = " << count 
         << " sig_y = " << state->yieldStress
         << " epdot = " << state->plasticStrainRate
         << " ep = " << state->plasticStrain
         << " T = " << state->temperature 
         << " Tm = " << state->meltingTemp << endl;
  }

  return deltaGamma;
}
//______________________________________________________________________
//
void 
RFElasticPlastic::carryForward(const PatchSubset* patches,
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
    constParticleVariable<double>  pPlasticStrain; 
    constParticleVariable<double>  pPlasticStrainRate;

    old_dw->get(pPlasticStrain,  pPlasticStrainLabel,  pset);
    old_dw->get(pPlasticStrainRate,  pPlasticStrainRateLabel,  pset);

    ParticleVariable<double>       pPlasticStrain_new;
    ParticleVariable<double>       pPlasticStrainRate_new;

    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pPlasticStrainRate_new,      
                           pPlasticStrainRateLabel_preReloc,      pset);

    // Get the plastic strain
    d_flow->getInternalVars(pset, old_dw);
    d_flow->allocateAndPutRigid(pset, new_dw);
    
    d_flow->getInternalVars(pset, old_dw);
    d_flow->allocateAndPutRigid(pset, new_dw);    
    

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pPlasticStrain_new[idx] = pPlasticStrain[idx];
      pPlasticStrainRate_new[idx] = pPlasticStrainRate[idx];
    }

    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);
    }
  }
}
         
//______________________________________________________________________
//
double RFElasticPlastic::computeRhoMicroCM(double pressure,
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
    double p_g_over_bulk = p_gauge/bulk;
    rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
  }
  return rho_cur;
}
//______________________________________________________________________
//
void RFElasticPlastic::computePressEOSCM(double rho_cur,double& pressure,
                                       double p_ref,  
                                       double& dp_drho, double& tmp,
                                       const MPMMaterial* matl, 
                                       double temperature)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();
  double inv_rho_orig = 1./rho_orig;

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = bulk/p_ref;
    double rho_rat_to_the_n = pow(rho_cur/rho_orig,n);
    pressure = A*rho_rat_to_the_n;
    dp_drho  = (bulk/rho_cur)*rho_rat_to_the_n;
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS
    double p_g = .5*bulk*(rho_cur*inv_rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + inv_rho_orig);
    tmp        = bulk/rho_cur;  // speed of sound squared
  }
}
//__________________________________
//
double RFElasticPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

//______________________________________________________________________
//
void 
RFElasticPlastic::addSplitParticlesComputesAndRequires(Task* task,
                                                       const MPMMaterial* matl,
                                                       const PatchSet* patches) 
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->modifies(pPlasticStrainLabel_preReloc,      matlset);
  task->modifies(pPlasticStrainRateLabel_preReloc,  matlset);
  task->modifies(pEnergyLabel_preReloc,             matlset);
}

void 
RFElasticPlastic::splitCMSpecificParticleData(const Patch* patch,
                                              const int dwi,
                                              const int fourOrEight,
                                              ParticleVariable<int> &prefOld,
                                              ParticleVariable<int> &prefNew,
                                              const unsigned int oldNumPar,
                                              const unsigned int numNewPartNeeded,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  ParticleVariable<double>  PlasStrain, PlasStrainRate,Energy;
  ParticleVariable<int> pLocalized;

  new_dw->getModifiable(PlasStrain,    pPlasticStrainLabel_preReloc,     pset);
  new_dw->getModifiable(PlasStrainRate,pPlasticStrainRateLabel_preReloc, pset);
  new_dw->getModifiable(pLocalized,    lb->pLocalizedMPMLabel_preReloc,  pset);
  new_dw->getModifiable(Energy,        pEnergyLabel_preReloc,            pset);

  ParticleVariable<double> PlasStrainTmp, PlasStrainRateTmp, EnergyTmp;
  ParticleVariable<int> pLocalizedTmp;

  new_dw->allocateTemporary(PlasStrainTmp,        pset);
  new_dw->allocateTemporary(PlasStrainRateTmp,    pset);
  new_dw->allocateTemporary(pLocalizedTmp,        pset);
  new_dw->allocateTemporary(EnergyTmp,            pset);
  //new_dw->allocateTemporary(pEquivStressTmp,      pset);
  // copy data from old variables for particle IDs and the position vector
  for(unsigned int pp=0; pp<oldNumPar; ++pp ){
    PlasStrainTmp[pp]     = PlasStrain[pp];
    PlasStrainRateTmp[pp] = PlasStrainRate[pp];
    pLocalizedTmp[pp]     = pLocalized[pp];
    EnergyTmp[pp]         = Energy[pp];
  }

  int numRefPar=0;
  for(unsigned int idx=0; idx<oldNumPar; ++idx ){
    if(prefNew[idx]!=prefOld[idx]){  // do refinement!
      for(int i = 0;i<fourOrEight;i++){
        int new_index;
        if(i==0){
          new_index=idx;
        } else {
          new_index=oldNumPar+(fourOrEight-1)*numRefPar+i;
        }
        PlasStrainTmp[new_index]     = PlasStrain[idx];
        PlasStrainRateTmp[new_index] = PlasStrainRate[idx];
        pLocalizedTmp[new_index]     = pLocalized[idx];
        EnergyTmp[new_index]         = Energy[idx];
      }
      numRefPar++;
    }
  }

  new_dw->put(PlasStrainTmp,      pPlasticStrainLabel_preReloc,       true);
  new_dw->put(PlasStrainRateTmp,  pPlasticStrainRateLabel_preReloc,   true);
  new_dw->put(pLocalizedTmp,      lb->pLocalizedMPMLabel_preReloc,    true);
  new_dw->put(EnergyTmp,          pEnergyLabel_preReloc,              true);
}
