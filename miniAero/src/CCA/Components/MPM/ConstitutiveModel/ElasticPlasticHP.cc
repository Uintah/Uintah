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

#include <CCA/Components/MPM/ConstitutiveModel/ElasticPlasticHP.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/YieldConditionFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/StabilityCheckFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/FlowStressModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DamageModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/MPMEquationOfStateFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/ShearModulusModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/MeltingTempModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/SpecificHeatModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DevStressModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PlasticityState.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DeformationState.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Labels/MPMLabel.h>
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

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>

using namespace std;
using namespace Uintah;

static DebugStream cout_EP("EP",false);
static DebugStream cout_EP1("EP1",false);
static DebugStream CSTi("EPi",false);
static DebugStream CSTir("EPir",false);

ElasticPlasticHP::ElasticPlasticHP(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM()
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

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
  string tmp = "empty";
  
  ps->get("plastic_convergence_algo",tmp);
  
  if (tmp == "biswajit"){
    d_plasticConvergenceAlgo = "biswajit";
    usingRR = false;
  }
  if(tmp != "radialReturn" && tmp != "biswajit" && tmp != "empty"){
    ostringstream warn;
    warn << "ElasticPlasticHP:: Invalid plastic_convergence_algo option ("
         << tmp << ") Valid options are: biswajit, radialReturn" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  // 
  d_yield = YieldConditionFactory::create(ps, usingRR );
  if(!d_yield){
    ostringstream desc;
    desc << "An error occured in the YieldConditionFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_stable = StabilityCheckFactory::create(ps);
  if(!d_stable) cerr << "Stability check disabled\n";

  d_flow = FlowStressModelFactory::create(ps);
  if(!d_flow){
    ostringstream desc;
    desc << "An error occured in the FlowModelFactory that has \n"
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
         << " Jim.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_shear = ShearModulusModelFactory::create(ps);
  if (!d_shear) {
    ostringstream desc;
    desc << "ElasticPlasticHP::Error in shear modulus model factory" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  d_melt = MeltingTempModelFactory::create(ps);
  if (!d_melt) {
    ostringstream desc;
    desc << "ElasticPlasticHP::Error in melting temp model factory" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  d_devStress = DevStressModelFactory::create(ps);
  if (!d_devStress) {
    ostringstream desc;
    desc << "ElasticPlasticHP::Error creating deviatoric stress model" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_computeSpecificHeat = false;
  ps->get("compute_specific_heat",d_computeSpecificHeat);
  d_Cp = SpecificHeatModelFactory::create(ps);
  
  setErosionAlgorithm();
  getInitialPorosityData(ps);
  getInitialDamageData(ps);
  //getSpecificHeatData(ps);
  initializeLocalMPMLabels();
}

ElasticPlasticHP::ElasticPlasticHP(const ElasticPlasticHP* cm) :
  ConstitutiveModel(cm), ImplicitCM(cm)
{
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  d_initialData.alpha = cm->d_initialData.alpha;
  d_initialData.Chi = cm->d_initialData.Chi;
  d_initialData.sigma_crit = cm->d_initialData.sigma_crit;

  d_tol = cm->d_tol ;
  d_useModifiedEOS = cm->d_useModifiedEOS;

  d_initialMaterialTemperature = cm->d_initialMaterialTemperature ;
  d_checkTeplaFailureCriterion = cm->d_checkTeplaFailureCriterion;
  d_doMelting = cm->d_doMelting;
  d_checkStressTriax = cm->d_checkStressTriax;

  d_setStressToZero = cm->d_setStressToZero;
  d_allowNoTension = cm->d_allowNoTension;
  d_allowNoShear = cm->d_allowNoShear;

  d_evolvePorosity = cm->d_evolvePorosity;
  d_porosity.f0 = cm->d_porosity.f0 ;
  d_porosity.f0_std = cm->d_porosity.f0_std ;
  d_porosity.fc = cm->d_porosity.fc ;
  d_porosity.fn = cm->d_porosity.fn ;
  d_porosity.en = cm->d_porosity.en ;
  d_porosity.sn = cm->d_porosity.sn ;
  d_porosity.porosityDist = cm->d_porosity.porosityDist ;

  d_evolveDamage = cm->d_evolveDamage;
  d_scalarDam.D0 = cm->d_scalarDam.D0 ;
  d_scalarDam.D0_std = cm->d_scalarDam.D0_std ;
  d_scalarDam.Dc = cm->d_scalarDam.Dc ;
  d_scalarDam.scalarDamageDist = cm->d_scalarDam.scalarDamageDist ;

  d_computeSpecificHeat = cm->d_computeSpecificHeat;
  /*
  d_Cp.A = cm->d_Cp.A;
  d_Cp.B = cm->d_Cp.B;
  d_Cp.C = cm->d_Cp.C;
  d_Cp.n = cm->d_Cp.n;
  */
  d_Cp      = SpecificHeatModelFactory::createCopy(cm->d_Cp);
  d_yield   = YieldConditionFactory::createCopy(cm->d_yield);
  d_stable  = StabilityCheckFactory::createCopy(cm->d_stable);
  d_flow    = FlowStressModelFactory::createCopy(cm->d_flow);
  d_damage  = DamageModelFactory::createCopy(cm->d_damage);
  d_eos     = MPMEquationOfStateFactory::createCopy(cm->d_eos);
  d_eos->setBulkModulus(d_initialData.Bulk);
  d_shear   = ShearModulusModelFactory::createCopy(cm->d_shear);
  d_melt    = MeltingTempModelFactory::createCopy(cm->d_melt);
  d_devStress = 0;
  
  initializeLocalMPMLabels();
}

ElasticPlasticHP::~ElasticPlasticHP()
{
  // Destructor 
  VarLabel::destroy(pRotationLabel);
  VarLabel::destroy(pStrainRateLabel);
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainRateLabel);
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pPorosityLabel);
  VarLabel::destroy(pLocalizedLabel);
  VarLabel::destroy(pEnergyLabel);

  VarLabel::destroy(pRotationLabel_preReloc);
  VarLabel::destroy(pStrainRateLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pPlasticStrainRateLabel_preReloc);
  VarLabel::destroy(pDamageLabel_preReloc);
  VarLabel::destroy(pPorosityLabel_preReloc);
  VarLabel::destroy(pLocalizedLabel_preReloc);
  VarLabel::destroy(pEnergyLabel_preReloc);

  delete d_flow;
  delete d_yield;
  delete d_stable;
  delete d_damage;
  delete d_eos;
  delete d_shear;
  delete d_melt;
  delete d_Cp;
  delete d_devStress;
}

//______________________________________________________________________
//
void ElasticPlasticHP::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","elastic_plastic_hp");
  }
  
  cm_ps->appendElement("bulk_modulus",                  d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",                 d_initialData.Shear);
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
  d_damage     ->outputProblemSpec(cm_ps);
  d_eos        ->outputProblemSpec(cm_ps);
  d_shear      ->outputProblemSpec(cm_ps);
  d_melt       ->outputProblemSpec(cm_ps);
  d_Cp         ->outputProblemSpec(cm_ps);

  cm_ps->appendElement("evolve_porosity",           d_evolvePorosity);
  cm_ps->appendElement("initial_mean_porosity",     d_porosity.f0);
  cm_ps->appendElement("initial_std_porosity",      d_porosity.f0_std);
  cm_ps->appendElement("critical_porosity",         d_porosity.fc);
  cm_ps->appendElement("frac_nucleation",           d_porosity.fn);
  cm_ps->appendElement("meanstrain_nucleation",     d_porosity.en);
  cm_ps->appendElement("stddevstrain_nucleation",   d_porosity.sn);
  cm_ps->appendElement("initial_porosity_distrib",  d_porosity.porosityDist);

  cm_ps->appendElement("evolve_damage",             d_evolveDamage);
  cm_ps->appendElement("initial_mean_scalar_damage",d_scalarDam.D0);
  cm_ps->appendElement("initial_std_scalar_damage", d_scalarDam.D0_std);
  cm_ps->appendElement("critical_scalar_damage",    d_scalarDam.Dc);
  cm_ps->appendElement("initial_scalar_damage_distrib",
                       d_scalarDam.scalarDamageDist);

  /*
  cm_ps->appendElement("Cp_constA", d_Cp.A);
  cm_ps->appendElement("Cp_constB", d_Cp.B);
  cm_ps->appendElement("Cp_constC", d_Cp.C);
  */

}


ElasticPlasticHP* ElasticPlasticHP::clone()
{
  return scinew ElasticPlasticHP(*this);
}

//______________________________________________________________________
//
void
ElasticPlasticHP::initializeLocalMPMLabels()
{
  pRotationLabel = VarLabel::create("p.rotation",
    ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel = VarLabel::create("p.strainRate",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainRateLabel = VarLabel::create("p.plasticStrainRate",
    ParticleVariable<double>::getTypeDescription());
  pDamageLabel = VarLabel::create("p.damage",
    ParticleVariable<double>::getTypeDescription());
  pPorosityLabel = VarLabel::create("p.porosity",
    ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel = VarLabel::create("p.localized",
    ParticleVariable<int>::getTypeDescription());
  pEnergyLabel = VarLabel::create("p.energy",
    ParticleVariable<double>::getTypeDescription());

  pRotationLabel_preReloc = VarLabel::create("p.rotation+",
    ParticleVariable<Matrix3>::getTypeDescription());
  pStrainRateLabel_preReloc = VarLabel::create("p.strainRate+",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainRateLabel_preReloc = VarLabel::create("p.plasticStrainRate+",
    ParticleVariable<double>::getTypeDescription());
  pDamageLabel_preReloc = VarLabel::create("p.damage+",
    ParticleVariable<double>::getTypeDescription());
  pPorosityLabel_preReloc = VarLabel::create("p.porosity+",
    ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel_preReloc = VarLabel::create("p.localized+",
    ParticleVariable<int>::getTypeDescription());
  pEnergyLabel_preReloc = VarLabel::create("p.energy+",
    ParticleVariable<double>::getTypeDescription());
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::getInitialPorosityData(ProblemSpecP& ps)
{
  d_evolvePorosity = true;
  ps->get("evolve_porosity",d_evolvePorosity);
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
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::getInitialDamageData(ProblemSpecP& ps)
{
  d_evolveDamage = true;
  ps->get("evolve_damage",d_evolveDamage);
  d_scalarDam.D0 = 0.0; // Initial scalar damage
  d_scalarDam.D0_std = 0.0; // Initial STD scalar damage
  d_scalarDam.Dc = 1.0; // Critical scalar damage
  d_scalarDam.scalarDamageDist = "constant";
  ps->get("initial_mean_scalar_damage",        d_scalarDam.D0);
  ps->get("initial_std_scalar_damage",         d_scalarDam.D0_std);
  ps->get("critical_scalar_damage",            d_scalarDam.Dc);
  ps->get("initial_scalar_damage_distrib",     d_scalarDam.scalarDamageDist);
}

/*! Compute specific heat

    double T = temperature;
    C_p = 1.0e3*(A + B*T + C/T^2)
    ** For steel **
    C_p = 1.0e3*(0.09278 + 7.454e-4*T + 12404.0/(T*T));
*/
/*
void 
ElasticPlasticHP::getSpecificHeatData(ProblemSpecP& ps)
{
  d_Cp.A = 0.09278;  // Constant A (HY100)
  d_Cp.B = 7.454e-4; // Constant B (HY100)
  d_Cp.C = 12404.0;  // Constant C (HY100)
  d_Cp.n = 2.0;      // Constant n (HY100)
  ps->get("Cp_constA", d_Cp.A);
  ps->get("Cp_constB", d_Cp.B);
  ps->get("Cp_constC", d_Cp.C);
  ps->get("Cp_constn", d_Cp.n);
}
*/
//______________________________________________________________________
//
void 
ElasticPlasticHP::setErosionAlgorithm()
{
  d_setStressToZero = false;
  d_allowNoTension  = false;
  d_allowNoShear    = false;
  if (flag->d_doErosion) {
    if (flag->d_erosionAlgorithm == "AllowNoTension") 
      d_allowNoTension = true;
    else if (flag->d_erosionAlgorithm == "AllowNoShear") 
      d_allowNoShear = true;
    else if (flag->d_erosionAlgorithm == "ZeroStress") 
      d_setStressToZero = true;
  }
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pRotationLabel);
  from.push_back(pStrainRateLabel);
  from.push_back(pPlasticStrainLabel);
  from.push_back(pPlasticStrainRateLabel);
  from.push_back(pDamageLabel);
  from.push_back(pPorosityLabel);
  from.push_back(pLocalizedLabel);
  from.push_back(pEnergyLabel);

  to.push_back(pRotationLabel_preReloc);
  to.push_back(pStrainRateLabel_preReloc);
  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pPlasticStrainRateLabel_preReloc);
  to.push_back(pDamageLabel_preReloc);
  to.push_back(pPorosityLabel_preReloc);
  to.push_back(pLocalizedLabel_preReloc);
  to.push_back(pEnergyLabel_preReloc);

  // Add the particle state for the flow & deviatoric stress model
  d_flow     ->addParticleState(from, to);
  d_devStress->addParticleState(from, to);
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::addInitialComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(pRotationLabel,      matlset);
  task->computes(pStrainRateLabel,    matlset);
  task->computes(pPlasticStrainLabel, matlset);
  task->computes(pPlasticStrainRateLabel, matlset);
  task->computes(pDamageLabel,        matlset);
  task->computes(pPorosityLabel,      matlset);
  task->computes(pLocalizedLabel,     matlset);
  task->computes(pEnergyLabel,        matlset);
 
  // Add internal evolution variables computed by flow & deviatoric stress model
  d_flow     ->addInitialComputesAndRequires(task, matl, patch);
  d_devStress->addInitialComputesAndRequires(task, matl);
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::initializeCMData(const Patch* patch,
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
  //cout << "Initialize CM Data in ElasticPlasticHP" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pRotation;
  ParticleVariable<double>  pPlasticStrain, pDamage, pPorosity, 
                            pPlasticStrainRate, pStrainRate, pEnergy;
  ParticleVariable<int>     pLocalized;

  new_dw->allocateAndPut(pRotation,          pRotationLabel, pset);
  new_dw->allocateAndPut(pStrainRate,        pStrainRateLabel, pset);
  new_dw->allocateAndPut(pPlasticStrain,     pPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pPlasticStrainRate, pPlasticStrainRateLabel, pset);
  new_dw->allocateAndPut(pDamage,            pDamageLabel, pset);
  new_dw->allocateAndPut(pLocalized,         pLocalizedLabel, pset);
  new_dw->allocateAndPut(pPorosity,          pPorosityLabel, pset);
  new_dw->allocateAndPut(pEnergy,            pEnergyLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){

    pRotation[*iter] = one;
    pStrainRate[*iter] = 0.0;
    pPlasticStrain[*iter] = 0.0;
    pPlasticStrainRate[*iter] = 0.0;
    pDamage[*iter] = d_damage->initialize();
    pPorosity[*iter] = d_porosity.f0;
    pLocalized[*iter] = 0;
    pEnergy[*iter] = 0.;
  }

  // Do some extra things if the porosity or the damage distribution
  // is not uniform.  
  // ** WARNING ** Weibull distribution needs to be implemented.
  //               At present only Gaussian available.
  if (d_porosity.porosityDist != "constant") {

    SCIRun::Gaussian gaussGen(d_porosity.f0, d_porosity.f0_std, 0, 1, DBL_MAX);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){

      // Generate a Gaussian distributed random number given the mean
      // porosity and the std.
      pPorosity[*iter] = fabs(gaussGen.rand(1.0));
    }
  }

  if (d_scalarDam.scalarDamageDist != "constant") {

    SCIRun::Gaussian gaussGen(d_scalarDam.D0, d_scalarDam.D0_std, 0, 1,DBL_MAX);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){

      // Generate a Gaussian distributed random number given the mean
      // damage and the std.
      pDamage[*iter] = fabs(gaussGen.rand(1.0));
    }
  }

  // Initialize the data for the flow model
  d_flow->initializeInternalVars(pset, new_dw);
  
  // Deviatoric Stress Model
  d_devStress->initializeInternalVars(pset, new_dw);
  
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::computeStableTimestep(const Patch* patch,
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
ElasticPlasticHP::addComputesAndRequires(Task* task,
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
  task->requires(Task::OldDW, pRotationLabel,         matlset, gnone);
  task->requires(Task::OldDW, pStrainRateLabel,       matlset, gnone);
  task->requires(Task::OldDW, pPlasticStrainLabel,    matlset, gnone);
  task->requires(Task::OldDW, pPlasticStrainRateLabel,matlset, gnone);
  task->requires(Task::OldDW, pDamageLabel,           matlset, gnone);
  task->requires(Task::OldDW, pPorosityLabel,         matlset, gnone);
  task->requires(Task::OldDW, pLocalizedLabel,        matlset, gnone);
  task->requires(Task::OldDW, lb->pParticleIDLabel,   matlset, gnone);
  task->requires(Task::OldDW, pEnergyLabel,           matlset, gnone);

  task->computes(pRotationLabel_preReloc,       matlset);
  task->computes(pStrainRateLabel_preReloc,     matlset);
  task->computes(pPlasticStrainLabel_preReloc,  matlset);
  task->computes(pPlasticStrainRateLabel_preReloc,  matlset);
  task->computes(pDamageLabel_preReloc,         matlset);
  task->computes(pPorosityLabel_preReloc,       matlset);
  task->computes(pLocalizedLabel_preReloc,      matlset);
  task->computes(pEnergyLabel_preReloc,         matlset);

  // Add internal evolution variables computed by flow model
  d_flow->addComputesAndRequires(task, matl, patches);
  
  // Deviatoric stress model
  d_devStress->addComputesAndRequires(task, matl);
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  if (cout_EP.active()) {
    cout_EP << getpid() 
            << " ElasticPlasticHP:ComputeStressTensor:Explicit"
            << " Matl = " << matl 
            << " DWI = " << matl->getDWIndex() 
            << " patch = " << (patches->get(0))->getID();
  }

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
  double sqrtTwoThird = 1.0/sqrtThreeTwo;
  
  double totalStrainEnergy = 0.0;
  double include_AV_heating=0.0;
  if (flag->d_artificial_viscosity_heating) {
    include_AV_heating=1.0;
  }

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

    old_dw->get(pMass,        lb->pMassLabel,               pset);
    old_dw->get(pVolume,      lb->pVolumeLabel,             pset);
    old_dw->get(pTemperature, lb->pTemperatureLabel,        pset);
    old_dw->get(pVelocity,    lb->pVelocityLabel,           pset);
    old_dw->get(pStress,      lb->pStressLabel,             pset);
    old_dw->get(pDeformGrad,  lb->pDeformationMeasureLabel, pset);

    constParticleVariable<double> pPlasticStrain, pDamage, pPorosity;
    constParticleVariable<double> pStrainRate, pPlasticStrainRate, pEnergy;
    constParticleVariable<int> pLocalized;
    constParticleVariable<Matrix3> pRotation;

    old_dw->get(pPlasticStrain,     pPlasticStrainLabel,     pset);
    old_dw->get(pDamage,            pDamageLabel,            pset);
    old_dw->get(pStrainRate,        pStrainRateLabel,        pset);
    old_dw->get(pPlasticStrainRate, pPlasticStrainRateLabel, pset);
    old_dw->get(pPorosity,          pPorosityLabel,          pset);
    old_dw->get(pEnergy,            pEnergyLabel,            pset);
    old_dw->get(pLocalized,         pLocalizedLabel,         pset);
    old_dw->get(pRotation,    pRotationLabel,               pset);

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
    ParticleVariable<Matrix3> pRotation_new;
    ParticleVariable<double>  pPlasticStrain_new, pDamage_new, pPorosity_new; 
    ParticleVariable<double>  pStrainRate_new, pPlasticStrainRate_new;
    ParticleVariable<int>     pLocalized_new;
    ParticleVariable<double>  pdTdt, p_q, pEnergy_new;
    ParticleVariable<Matrix3> pStress_new;
    
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pPlasticStrainRate_new,      
                           pPlasticStrainRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pPorosity_new,      
                           pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
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

      // Calculate rate of deformation tensor (D)
      tensorD = (tensorL + tensorL.Transpose())*0.5;

      // Compute polar decomposition of F (F = RU)
      pDeformGrad[idx].polarDecompositionRMB(tensorU, tensorR);

      // Rotate the total rate of deformation tensor back to the 
      // material configuration
      tensorD = (tensorR.Transpose())*(tensorD*tensorR);

      // Calculate the deviatoric part of the non-thermal part
      // of the rate of deformation tensor
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);
      
      pStrainRate_new[idx] = sqrtTwoThird*tensorD.Norm();

      // Rotate the Cauchy stress back to the 
      // material configuration and calculate the deviatoric part
      sigma = pStress[idx];
      sigma = (tensorR.Transpose())*(sigma*tensorR);
      double pressure = sigma.Trace()/3.0; 
      tensorS = sigma - one * pressure;

      // Rotate internal Cauchy stresses back to the 
      // material configuration (only for viscoelasticity)

      d_devStress->rotateInternalStresses(idx, tensorR);

      double temperature = pTemperature[idx];

      // Set up the PlasticityState (for t_n+1)
      PlasticityState* state = scinew PlasticityState();
      //state->plasticStrainRate = pStrainRate_new[idx];
      //state->plasticStrain     = pPlasticStrain[idx];
      //state->plasticStrainRate = sqrtTwoThird*tensorEta.Norm();
      state->strainRate          = pStrainRate_new[idx];
      state->plasticStrainRate   = pPlasticStrainRate[idx];
      state->plasticStrain       = pPlasticStrain[idx] 
                                 + state->plasticStrainRate*delT;
      state->pressure            = pressure;
      state->temperature         = temperature;
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
    
      // Calculate the shear modulus and the melting temperature at the
      // start of the time step and update the plasticity state
      double Tm_cur = d_melt->computeMeltingTemp(state);
      state->meltingTemp = Tm_cur ;
      
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
      //  the entire stress (not just deviatoric)
      double equivStress = sqrtThreeTwo*trialS.Norm();

      // Calculate flow stress
      double flowStress = d_flow->computeFlowStress(state, delT, d_tol, 
                                                    matl, idx);
      state->yieldStress = flowStress;

      // Material has melted if flowStress <= 0.0
      bool melted  = false;
      bool plastic = false;
      if (temperature > Tm_cur || flowStress <= 0.0) {

        melted = true;
        // Set the deviatoric stress to zero
        if (d_doMelting){
           tensorS = 0.0;
        } else {
           cerr << "The material has exceed the melt temperature, but you haven't turned \n";
           cerr << "melting on.  ElasticPlasticHP does nonsensical things here.  You should \n";
           cerr << "probably either set <do_melting>true</do_melting>, or increase the material\n";
           cerr << "<melt_temp> to a level that won't be exceeded.\n";
        }

        d_flow->updateElastic(idx);

      } else {

        // Get the current porosity 
        double porosity = pPorosity[idx];

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
          double normS  = tensorS.Norm();

          // If the material goes plastic in the first step, or
          // gammadotplus < 0 or delGamma < 0 use the Simo algorithm
          // with Newton iterations.

           //  Here set to true, if all conditionals are met (immediately above) then set to false.
          bool doRadialReturn = true;
          Matrix3 tensorEtaPlasticInc = zero;
          //__________________________________
          //
          if (normS > 0.0 && d_plasticConvergenceAlgo == "biswajit") {
            doRadialReturn = computePlasticStateBiswajit(state, pPlasticStrain, pStrainRate, 
                                                         sigma, trialS, tensorEta, tensorS,
                                                         delGamma, flowStress, porosity, mu_cur, delT, matl, idx);
          }
          
          //__________________________________
          //
          if (doRadialReturn) {

            // Compute Stilde using Newton iterations a la Simo
            state->plasticStrainRate = pStrainRate_new[idx];
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
      } // end of temperature if

      // Calculate the updated hydrostatic stress
      double p = d_eos->computePressure(matl, state, tensorF_new, tensorD,delT);

      double Dkk = tensorD.Trace();
      double dTdt_isentropic = d_eos->computeIsentropicTemperatureRate(
                                                 temperature,rho_0,rho_cur,Dkk);
      pdTdt[idx] += dTdt_isentropic;

      // Calculate Tdot from viscoelasticity
      double taylorQuinney = d_initialData.Chi;
      double fac = taylorQuinney/(rho_cur*state->specificHeat);
      double Tdot_VW = defState->viscoElasticWorkRate*fac;

      pdTdt[idx] += Tdot_VW;

      double de_s=0.;
      if (flag->d_artificial_viscosity) {
        double c_bulk = sqrt(bulk/rho_cur);
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        p_q[idx] = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
        de_s     = -p_q[idx]*Dkk/rho_cur;
      } else {
        p_q[idx] = 0.;
        de_s     = 0.;
      }

      // Calculate Tdot due to artificial viscosity
      double Tdot_AV = de_s/state->specificHeat;
      pdTdt[idx] += Tdot_AV*include_AV_heating;

      Matrix3 tensorHy = one*p;
   
      // Calculate the total stress
      sigma = tensorS + tensorHy;

      // If the particle has already failed, apply various erosion algorithms
      if (flag->d_doErosion) {
        if (pLocalized[idx]) {
          if (d_allowNoTension) {
            if (p > 0.0){
              sigma = zero;
            } else{
              sigma = tensorHy;
            }
          }
          if(d_allowNoShear){
            sigma = tensorHy;
          } else if (d_setStressToZero){
            sigma = zero;
          }
        }
      }
      
      //-----------------------------------------------------------------------
      // Stage 3:
      //-----------------------------------------------------------------------
      // Compute porosity/damage/temperature change
      if (!plastic) {

        // Save the updated data
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pPlasticStrainRate_new[idx] = 0.0;
        pDamage_new[idx]   = pDamage[idx];
        pPorosity_new[idx] = pPorosity[idx];
        
      } else {

        // Update the plastic strain
        pPlasticStrain_new[idx]     = state->plasticStrain;
        pPlasticStrainRate_new[idx] = state->plasticStrainRate;

        /*
        if (pPlasticStrainRate_new[idx] > pStrainRate_new[idx]) {
          cout << "Patch = " << patch->getID() << " particle = " << idx
               << " edot = " << pStrainRate_new[idx] 
               << " epdot = " << pPlasticStrainRate_new[idx] << endl;
        }
        */

        // Update the porosity
        if (d_evolvePorosity) {
          pPorosity_new[idx] = updatePorosity(tensorD, delT, pPorosity[idx], 
                                              state->plasticStrain);
        } else {
          pPorosity_new[idx] = pPorosity[idx];
        }
        
        // Calculate the updated scalar damage parameter
        if (d_evolveDamage) { 
          pDamage_new[idx] = 
            d_damage->computeScalarDamage(state->plasticStrainRate, sigma, 
                                          temperature, delT, matl, d_tol, 
                                          pDamage[idx]);
        } else {
          pDamage_new[idx] = pDamage[idx];
        }
        // Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = d_initialData.Chi;
        double fac = taylorQuinney/(rho_cur*state->specificHeat);

        // Calculate Tdot (internal plastic heating rate)
        double Tdot_PW = state->yieldStress*state->plasticStrainRate*fac;

        pdTdt[idx] += Tdot_PW;
      }

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

          // Check 2: Modified Tepla rule
          if (d_checkTeplaFailureCriterion) {
            tepla = (pPorosity_new[idx]*pPorosity_new[idx])/
                    (d_porosity.fc*d_porosity.fc);
            if (tepla > 1.0) isLocalized = true;
          } 

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
            pDamage_new[idx]   = 0.0;
            pPorosity_new[idx] = 0.0;
          } else {
            // set the particle localization flag to true  
            pLocalized_new[idx] = 1;
            pDamage_new[idx]    = 0.0;
            pPorosity_new[idx]  = 0.0;

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

      // Update the kinematic variables
      pRotation_new[idx] = tensorR;

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
        double Vdot = (pVolume_deformed[idx] - pVolume[idx])/(pMass[idx]*delT);

        pEnergy_new[idx] = pEnergy[idx] + pSpecificStrainEnergy 
                                        - p_q[idx]*Vdot*delT*include_AV_heating;

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

//______________________________________________________________________
//
bool ElasticPlasticHP::computePlasticStateBiswajit(PlasticityState* state, 
                                                   constParticleVariable<double>& pPlasticStrain,
                                                   constParticleVariable<double>& pStrainRate,
                                                   const Matrix3& sigma,
                                                   const Matrix3& trialS,
                                                   const Matrix3& tensorEta,
                                                   Matrix3& tensorS,
                                                   double& delGamma,
                                                   double& flowStress,
                                                   double& porosity,
                                                   double& mu_cur,
                                                   const double delT,
                                                   const MPMMaterial* matl,
                                                   const int idx)
{
  // Using the algorithm from Zocher, Maudlin, Chen, Flower-Maudlin
  // European Congress on Computational Methods in Applied Sciences 
  // and Engineering,  September 11-14, 2000.
  // Basic assumption is that all strain rate is plastic strain rate

  // Calculate the derivative of the yield function (using the 
  // previous time step (n) values)
  Matrix3 q(0.0);
  Matrix3 Stilde(0.0);
  
  double sqrtTwo      = sqrt(2.0);
  double sqrtThreeTwo = sqrt(1.5);
  double sqrtTwoThird = 1.0/sqrtThreeTwo;
  
  d_yield->evalDevDerivOfYieldFunction(sigma, flowStress, porosity, q);

  // Calculate the tensor u (at start of time interval) This is the normal to the yield surface.
  double sqrtqs = sqrt(q.Contract(tensorS));
  Matrix3 u = q/sqrtqs;

  // Calculate u_q and u_eta
  double etaeta = sqrt(tensorEta.NormSquared());
  Matrix3 u_eta = tensorEta/etaeta;
  double sqrtqq = sqrt(q.NormSquared());
  Matrix3 u_q   = q/sqrtqq;

  // Calculate c and d at the beginning of time step
  double cplus = u.NormSquared();
  double dplus = u.Contract(tensorEta);
  double gammadotplus = dplus/cplus;

  // Alternative calculation of gammadotplus
  //double gammadotplus = 
  // sqrtThreeTwo*sqrtqs/sqrtqq*state->plasticStrainRate;
  //gammadotplus = (gammadotplus < 0.0) ? 0.0 : gammadotplus;

  //__________________________________
  //
  bool doRadialReturn = true;
  if (gammadotplus > 0.0) {

    // Calculate dStar/cstar 
    double u_eta_eta = u_eta.Contract(tensorEta);
    double u_q_eta   = u_q.Contract(tensorEta);
    double AA        = 2.0/sqrt(cplus);
    double BB        = - (u_eta_eta + u_q_eta);
    double CC        = - gammadotplus*cplus*(u_eta_eta - u_q_eta);
    double term1     = BB*BB - 4.0*AA*CC;
    term1 = (term1 < 0.0) ? 0.0 : term1;

    double dStar = (-BB + sqrt(term1))/(2.0*AA);

    // Calculate delGammaEr
    //state->plasticStrainRate = 
    //  (sqrtTwoThird*sqrtqq*gammadotplus)/sqrtqs;
    //state->yieldStress = d_flow->computeFlowStress(state, delT, 
    //                                                  d_tol, matl, 
    //                                                  idx);
    double delGammaEr =  (sqrtTwo*state->yieldStress - sqrtqs)/
                         (2.0*mu_cur*cplus);

    // Calculate delGamma
    delGamma = dStar/cplus*delT - delGammaEr;
    if (delGamma > 0.0) {

      // Compute the actual epdot, ep, yieldStress
      double epdot = (sqrtTwoThird * sqrtqq * delGamma)/(sqrtqs * delT);
      if (epdot <= pStrainRate[idx]) {

        state->plasticStrainRate = epdot;
        state->plasticStrain = pPlasticStrain[idx] + 
                               state->plasticStrainRate * delT;

        state->yieldStress = d_flow->computeFlowStress(state, delT,
                                                          d_tol, matl,
                                                          idx);

        // Calculate Stilde
        // The exact form of denom will be different for 
        // different yield conditions ** WARNING ***
        ASSERT(state->yieldStress != 0.0);
        double denom = 1.0 + (3.0 * sqrtTwo * mu_cur * delGamma)/state->yieldStress; 
        ASSERT(denom != 0.0);
        Stilde = trialS/denom;

        /*
        double delLambda = sqrtqq*delGamma/sqrtqs;
        cout << "idx = " << idx << " delGamma = " << delLambda 
             << " sigy = " << state->yieldStress 
             << " epdot = " << state->plasticStrainRate 
             << " ep = " << state->plasticStrain << endl;
        */

        // We have found Stilde. Turn off Newton Iterations.
        doRadialReturn = false;

      } // end of epdot <= edot if
    } // end of delGamma > 0 if
  } // end of gammdotplus > 0 if

  // Do radial return adjustment
  double stst = sqrtThreeTwo*Stilde.Norm();

  Stilde = Stilde*(state->yieldStress/stst);
  tensorS = Stilde;
  
  return doRadialReturn;
}

////////////////////////////////////////////////////////////////////////
/*! \brief Compute Stilde, epdot, ep, and delGamma using 
  Simo's approach */
////////////////////////////////////////////////////////////////////////
void 
ElasticPlasticHP::computePlasticStateViaRadialReturn(const Matrix3& trialS,
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
ElasticPlasticHP::computeDeltaGamma(const double& delT,
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

    if (isnan(g) || isnan(deltaGamma)) {
      cout << "idx = " << idx << " iter = " << count 
           << " g = " << g << " Dg = " << Dg << " deltaGamma = " << deltaGamma 
           << " sigy = " << sigma_y 
           << " dsigy/depdot = " << dsigy_depdot << " dsigy/dep= " << dsigy_dep 
           << " epdot = " << state->plasticStrainRate 
           << " ep = " << state->plasticStrain << endl;
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

  if (isnan(state->yieldStress)) {
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
ElasticPlasticHP::computeStressTensorImplicit(const PatchSubset* patches,
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
                                 pPlasticStrain, pDamage, pPorosity, 
                                 pStrainRate, pPlasticStrainRate,pEnergy;

  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pDeformGrad, pStress, pRotation;
  constNCVariable<Vector>        gDisp;

  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<Matrix3>      pDeformGrad_new, pStress_new, pRotation_new;
  ParticleVariable<double>       pVolume_deformed, pPlasticStrain_new, 
                                 pDamage_new, pPorosity_new, pStrainRate_new,
                                 pPlasticStrainRate_new, pdTdt,pEnergy_new;

  // Local variables
  Matrix3 DispGrad(0.0); // Displacement gradient
  Matrix3 DefGrad, incDefGrad, incFFt, incFFtInv, Rotation, RightStretch; 
  Matrix3 incTotalStrain(0.0), incThermalStrain(0.0), incStrain(0.0);
  Matrix3 sigma(0.0), trialStress(0.0), trialS(0.0);
  DefGrad.Identity(); incDefGrad.Identity(); incFFt.Identity(); 
  incFFtInv.Identity(); Rotation.Identity(), RightStretch.Identity();

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
    old_dw->get(pRotation,           pRotationLabel,           pset);
    old_dw->get(pPlasticStrain,      pPlasticStrainLabel,      pset);
    old_dw->get(pPlasticStrainRate,  pPlasticStrainRateLabel,  pset);
    old_dw->get(pDamage,             pDamageLabel,             pset);
    old_dw->get(pStrainRate,         pStrainRateLabel,         pset);
    old_dw->get(pPorosity,           pPorosityLabel,           pset);
    old_dw->get(pLocalized,          pLocalizedLabel,          pset);
    old_dw->get(pEnergy,             pEnergyLabel,             pset);

    // Create and allocate arrays for storing the updated information
    // GLOBAL
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolume_deformed, 
                           lb->pVolumeDeformedLabel,              pset);
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel,            pset);

    // LOCAL
    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pPlasticStrainRate_new,      
                           pPlasticStrainRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pPorosity_new,      
                           pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pEnergy_new,      
                           pEnergyLabel_preReloc,                 pset);

    // Get the plastic strain
    d_flow     ->getInternalVars(pset, old_dw);
    d_devStress->getInternalVars(pset, old_dw);
    d_flow     ->allocateAndPutInternalVars(pset, new_dw);
    d_devStress->allocateAndPutInternalVars(pset, new_dw);

    //__________________________________
    // Special case for rigid materials
    double totalStrainEnergy = 0.0;
    if (matl->getIsRigid()) {
      ParticleSubset::iterator iter = pset->begin(); 
      for( ; iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pRotation_new[idx]      = pRotation[idx];
        pStrainRate_new[idx]    = pStrainRate[idx];
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pPlasticStrainRate_new[idx] = 0.0;
        pDamage_new[idx]        = pDamage[idx];
        pPorosity_new[idx]      = pPorosity[idx];
        pLocalized_new[idx]     = pLocalized[idx];

        pStress_new[idx]      = Zero;
        pDeformGrad_new[idx]  = One; 
        pVolume_deformed[idx] = pVolume[idx];
        pdTdt[idx] = 0.0;
      }
      
      if (flag->d_reductionVars->accStrainEnergy ||
          flag->d_reductionVars->strainEnergy) {
        new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
      }
      delete interpolator;
      continue;
    }

    //__________________________________
    // Standard case for deformable materials
    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;
      pEnergy_new[idx] = pEnergy[idx];

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
        throw InternalError("Negative Jacobian",__FILE__,__LINE__);
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
      DefGrad.polarDecompositionRMB(RightStretch, Rotation);
      pRotation_new[idx] = Rotation;

      // Compute the current strain and strain rate
      incFFt = incDefGrad*incDefGrad.Transpose(); 
      incFFtInv = incFFt.Inverse();
      incTotalStrain = (One - incFFtInv)*0.5;
      
      // Try the small strain option (BB - 06/22/05)
      // Calculate the strain 
      //incTotalStrain = (DispGrad + DispGrad.Transpose())*0.5;
      
      pStrainRate_new[idx] = incTotalStrain.Norm()*sqrtTwoThird/delT;
      
      // Compute thermal strain
      double temperature = pTemperature[idx];
      double incT = temperature - pTempPrev[idx];
      incThermalStrain = One*(alpha*incT);
      incStrain = incTotalStrain - incThermalStrain;
      
      // Compute pressure and deviatoric stress at t_n and
      // the volumetric strain and deviatoric strain increments at t_n+1
      sigma = pStress[idx];
      double pressure = sigma.Trace()/3.0;
      Matrix3 tensorS = sigma - One*pressure;
      
      // Set up the PlasticityState
      PlasticityState* state   = scinew PlasticityState();
      state->strainRate        = pStrainRate_new[idx];
      state->plasticStrainRate = pPlasticStrainRate[idx];
      state->plasticStrain     = pPlasticStrain[idx];
      state->pressure          = pressure;
      state->temperature       = temperature;
      state->initialTemperature = d_initialMaterialTemperature;
      state->density            = rho_cur;
      state->initialDensity     = rho_0;
      state->volume             = pVolume_deformed[idx];
      state->initialVolume      = volold;
      state->bulkModulus        = bulk ;
      state->initialBulkModulus = bulk;
      state->shearModulus       = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp        = Tm ;
      state->initialMeltTemp    = Tm;
      state->specificHeat       = matl->getSpecificHeat();
      state->energy             = pEnergy[idx];

      // Get or compute the specific heat
      if (d_computeSpecificHeat) {
        double C_p = d_Cp->computeSpecificHeat(state);
        state->specificHeat = C_p;
      }
    
      // Calculate the shear modulus and the melting temperature at the
      // start of the time step and update the plasticity state
      double Tm_cur = d_melt->computeMeltingTemp(state);
      state->meltingTemp = Tm_cur ;
      double mu_cur = d_shear->computeShearModulus(state);
      state->shearModulus = mu_cur ;

      // The calculation of tensorD and tensorEta are so that the deviatoric
      // stress calculation will work.  

      Matrix3 tensorD = incStrain/delT;
      
      // Calculate the deviatoric part of the non-thermal part
      // of the rate of deformation tensor
      Matrix3 tensorEta = tensorD - One*(tensorD.Trace()/3.0);

      DeformationState* defState = scinew DeformationState();
      defState->tensorD   = tensorD;
      defState->tensorEta = tensorEta;
      
      // Assume elastic deformation to get a trial deviatoric stress
      // This is simply the previous timestep deviatoric stress plus a
      // deviatoric elastic increment based on the shear modulus supplied by
      // the strength routine in use.
      d_devStress->computeDeviatoricStressInc(idx, state, defState, delT);
      trialS = tensorS + defState->devStressInc;
      trialStress    = trialS + One*(bulk*incStrain.Trace());
      
      // Calculate the equivalent stress
      // this will be removed next, it should be computed in the flow stress routine
      // the flow stress routines should be passed the entire stress (not just deviatoric)
      double equivStress = sqrt((trialS.NormSquared())*1.5);

      // Calculate flow stress (strain driven problem)
      double flowStress = d_flow->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);

      // Get the current porosity 
      double porosity = pPorosity[idx];

      // Evaluate yield condition
      double traceOfTrialStress = trialStress.Trace();
      double sig = flowStress;
      double flow_rule = d_yield->evalYieldCondition(equivStress, flowStress,
                                                     traceOfTrialStress, 
                                                     porosity, sig);
      
      // Compute the deviatoric stress
      if (flow_rule < 0.0) {

        // Save the updated data
        pStress_new[idx]        = trialStress;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pPlasticStrainRate_new[idx] = 0.0;
        pDamage_new[idx]        = pDamage[idx];
        pPorosity_new[idx]      = pPorosity[idx];
        
        // Update the internal variables
        d_flow->updateElastic(idx);

        // Update internal Cauchy stresses (only for viscoelasticity)
        Matrix3 dp = Zero;
        d_devStress->updateInternalStresses(idx, dp, defState, delT);

        // Compute stability criterion
        pLocalized_new[idx] = pLocalized[idx];

      } else {
        //__________________________________
        // Radial Return
        // Do Newton iteration to compute delGamma and updated 
        // plastic strain, plastic strain rate, and yield stress
        double delGamma = 0.0;
        Matrix3 nn = (0.);
        
        computePlasticStateViaRadialReturn(trialS, delT, matl, idx, state, nn, delGamma); 
        
        Matrix3 tensorEtaPlasticInc = nn * delGamma;
        pStress_new[idx]            = trialStress - tensorEtaPlasticInc*(2.0*state->shearModulus);
        pPlasticStrain_new[idx]     = state->plasticStrain;
        pPlasticStrainRate_new[idx] = state->plasticStrainRate;

        // Update internal Cauchy stresses (only for viscoelasticity)
        Matrix3 dp = tensorEtaPlasticInc/delT;
        d_devStress->updateInternalStresses(idx, dp, defState, delT);

        // Update the porosity
        
        pPorosity_new[idx] = pPorosity[idx];
        
        if (d_evolvePorosity) {
          Matrix3 tensorD = incStrain/delT;
          double ep = state->plasticStrain;
          pPorosity_new[idx] = updatePorosity(tensorD, delT, porosity, ep);
        }

        // Calculate the updated scalar damage parameter
        if (d_evolveDamage) 
          pDamage_new[idx] = 
            d_damage->computeScalarDamage(state->plasticStrainRate, 
                                          pStress_new[idx],
                                          temperature,
                                          delT, matl, d_tol, 
                                          pDamage[idx]);
        else
          pDamage_new[idx] = pDamage[idx];

        // Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = d_initialData.Chi;
        double fac = taylorQuinney/(rho_cur*state->specificHeat);

        // Calculate Tdot (internal plastic heating rate)
        double Tdot = state->yieldStress*state->plasticStrainRate*fac;
        pdTdt[idx] = Tdot;

        // No failure implemented for implcit time integration
        pLocalized_new[idx] = pLocalized[idx];

        // Update internal variables in the flow model
        d_flow->updatePlastic(idx, delGamma);
      }

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        Matrix3 avgStress = (pStress_new[idx] + pStress[idx])*0.5;
        double pStrainEnergy = (incStrain(0,0)*avgStress(0,0) +
                                incStrain(1,1)*avgStress(1,1) +
                                incStrain(2,2)*avgStress(2,2) +
                           2.0*(incStrain(0,1)*avgStress(0,1) + 
                                incStrain(0,2)*avgStress(0,2) +
                                incStrain(1,2)*avgStress(1,2)))
                               *pVolume_deformed[idx]*delT;
        totalStrainEnergy += pStrainEnergy;
      }
      delete state;
      delete defState;
    }
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::addComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patches,
                                       const bool recurse,
                                       const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForImplicitHypo(task, matlset, true, recurse, SchedParent);

  Ghost::GhostType  gnone = Ghost::None;
  if(SchedParent){
    // For subscheduler
    task->requires(Task::ParentOldDW, lb->pTempPreviousLabel,  matlset, gnone); 
    task->requires(Task::ParentOldDW, lb->pTemperatureLabel,   matlset, gnone);
    task->requires(Task::ParentOldDW, pPlasticStrainLabel,     matlset, gnone);
    task->requires(Task::ParentOldDW, pPlasticStrainRateLabel, matlset, gnone);
    task->requires(Task::ParentOldDW, pPorosityLabel,          matlset, gnone);

    task->computes(pPlasticStrainLabel_preReloc,               matlset);
    task->computes(pPlasticStrainRateLabel_preReloc,           matlset);
    task->computes(pPorosityLabel_preReloc,                    matlset);
  }else{
    // For scheduleIterate
    task->requires(Task::OldDW, lb->pTempPreviousLabel,  matlset, gnone); 
    task->requires(Task::OldDW, lb->pTemperatureLabel,   matlset, gnone);
    task->requires(Task::OldDW, pPlasticStrainLabel,     matlset, gnone);
    task->requires(Task::OldDW, pPlasticStrainRateLabel, matlset, gnone);
    task->requires(Task::OldDW, pPorosityLabel,          matlset, gnone);
  }

  // Add internal evolution variables computed by flow model
  d_flow->addComputesAndRequires(task, matl, patches, recurse, SchedParent);
  
  // Deviatoric Stress Model
  d_devStress->addComputesAndRequires(task, matl, SchedParent);
}
//______________________________________________________________________
//
void 
ElasticPlasticHP::computeStressTensorImplicit(const PatchSubset* patches,
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
                                 pPlasticStrain, pPlasticStrainRate,
                                 pPorosity;

  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pDeformGrad, pStress;
  constNCVariable<Vector>        gDisp;

  ParticleVariable<Matrix3>      pDeformGrad_new, pStress_new;
  ParticleVariable<double>       pVolume_deformed, pPlasticStrain_new,
                                 pPlasticStrainRate_new; 

  // Local variables
  Matrix3 DispGrad(0.0); // Displacement gradient
  Matrix3 DefGrad, incDefGrad, incFFt, incFFtInv;
  Matrix3 incTotalStrain(0.0), incThermalStrain(0.0), incStrain(0.0);
  Matrix3 sigma(0.0), trialStress(0.0), trialS(0.0);
  DefGrad.Identity(); incDefGrad.Identity(); incFFt.Identity(); 
  incFFtInv.Identity(); 

  // For B matrices
  double D[6][6];
  double B[6][24];
  double Bnl[3][24];
  double Kmatrix[24][24];
  int dof[24];
  double v[576];

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

    parent_old_dw->get(delT,         lb->delTLabel, getLevel(patches));
    parent_old_dw->get(pTempPrev,    lb->pTempPreviousLabel,       pset); 
    parent_old_dw->get(pTemperature, lb->pTemperatureLabel,        pset);
    parent_old_dw->get(px,           lb->pXLabel,                  pset);
    parent_old_dw->get(psize,        lb->pSizeLabel,               pset);
    parent_old_dw->get(pMass,        lb->pMassLabel,               pset);
    parent_old_dw->get(pDeformGrad,  lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pStress,      lb->pStressLabel,             pset);

    // GET LOCAL DATA 
    parent_old_dw->get(pPlasticStrain,      pPlasticStrainLabel,     pset);
    parent_old_dw->get(pPlasticStrainRate,  pPlasticStrainRateLabel, pset);
    parent_old_dw->get(pPorosity,           pPorosityLabel,          pset);

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
    new_dw->allocateAndPut(pPlasticStrainRate_new,      
                           pPlasticStrainRateLabel_preReloc,      pset);

    // Special case for rigid materials
    if (matl->getIsRigid()) {
      ParticleSubset::iterator iter = pset->begin(); 
      for( ; iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pPlasticStrainRate_new[idx] = 0.0;

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

      // Calculate the displacement gradient
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDeformGrad[idx]);
      computeGradAndBmats(DispGrad,ni,d_S, oodx, gDisp, l2g,B, Bnl, dof);

      // Compute the deformation gradient increment
      incDefGrad = DispGrad + One;

      // Update the deformation gradient
      DefGrad = incDefGrad*pDeformGrad[idx];
      pDeformGrad_new[idx] = DefGrad;
      double J = DefGrad.Determinant();

      // Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw InternalError("Negative Jacobian",__FILE__,__LINE__);
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;
      double volold = (pMass[idx]/rho_0);
      pVolume_deformed[idx] = volold*J;

      // Compute the current strain and strain rate
      incFFt = incDefGrad*incDefGrad.Transpose(); 
      incFFtInv = incFFt.Inverse();
      incTotalStrain = (One - incFFtInv)*0.5;

      double pStrainRate_new = incTotalStrain.Norm()*sqrt(2.0/3.0)/delT;

      // Compute thermal strain increment
      double temperature = pTemperature[idx];
      double incT = temperature - pTempPrev[idx];
      incThermalStrain = One*(alpha*incT);
      incStrain = incTotalStrain - incThermalStrain;
      
      // Compute pressure and deviatoric stress at t_n and
      // the volumetric strain and deviatoric strain increments at t_n+1
      sigma = pStress[idx];
      double pressure = sigma.Trace()/3.0;
      Matrix3 tensorS = sigma - One*pressure;
      
      // Set up the PlasticityState
      PlasticityState* state    = scinew PlasticityState();
      state->strainRate         = pStrainRate_new;
      state->plasticStrainRate  = pPlasticStrainRate[idx];
      state->plasticStrain      = pPlasticStrain[idx];
      state->pressure           = pressure;
      state->temperature        = temperature;
      state->initialTemperature = d_initialMaterialTemperature;
      state->density            = rho_cur;
      state->initialDensity     = rho_0;
      state->volume             = pVolume_deformed[idx];
      state->initialVolume      = volold;
      state->bulkModulus        = bulk ;
      state->initialBulkModulus = bulk;
      state->shearModulus       = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp        = Tm ;
      state->initialMeltTemp = Tm;
      state->specificHeat = matl->getSpecificHeat();

      // Get or compute the specific heat
      if (d_computeSpecificHeat) {
        double C_p = d_Cp->computeSpecificHeat(state);
        state->specificHeat = C_p;
      }
    
      // Calculate the shear modulus and the melting temperature at the
      // start of the time step and update the plasticity state
      double Tm_cur = d_melt->computeMeltingTemp(state);
      state->meltingTemp = Tm_cur ;
      double mu_cur = d_shear->computeShearModulus(state);
      state->shearModulus = mu_cur ;

      // The calculation of tensorD and tensorEta are so that the deviatoric
      // stress calculation will work.  

      Matrix3 tensorD = incStrain/delT;
      
      // Calculate the deviatoric part of the non-thermal part
      // of the rate of deformation tensor
      Matrix3 tensorEta = tensorD - One*(tensorD.Trace()/3.0);

      DeformationState* defState = scinew DeformationState();
      defState->tensorD   = tensorD;
      defState->tensorEta = tensorEta;
      
      // Assume elastic deformation to get a trial deviatoric stress
      // This is simply the previous timestep deviatoric stress plus a
      // deviatoric elastic increment based on the shear modulus supplied by
      // the strength routine in use.
      d_devStress->computeDeviatoricStressInc( idx, state, defState, delT);
      trialS = tensorS + defState->devStressInc;
      trialStress    = trialS + One*(bulk*incStrain.Trace());
      
      delete defState;
      
      // Calculate the equivalent stress
      // this will be removed next, it should be computed in the flow stress routine
      // the flow stress routines should be passed the entire stress (not just deviatoric)
      double equivStress = sqrt((trialS.NormSquared())*1.5);

      // Calculate flow stress (strain driven problem)
      double flowStress = d_flow->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);

      // Get the current porosity 
      double porosity = pPorosity[idx];

      // Evaluate yield condition
      double traceOfTrialStress = trialStress.Trace();
      double sig = flowStress;
      double flow_rule = d_yield->evalYieldCondition(equivStress, flowStress,
                                                     traceOfTrialStress, 
                                                     porosity, sig);
      
      // Compute the deviatoric stress
      if (flow_rule < 0.0) {

        // Save the updated data
        pStress_new[idx] = trialStress;
        pPlasticStrain_new[idx]     = pPlasticStrain[idx];
        pPlasticStrainRate_new[idx] = 0.0;
        
        // Update internal Cauchy stresses (only for viscoelasticity)
        Matrix3 dp = Zero;
        d_devStress->updateInternalStresses(idx, dp, defState, delT);

        computeElasticTangentModulus(bulk, shear, D);

      } else {

        //__________________________________
        // Radial Return
        // Do Newton iteration to compute delGamma and updated 
        // plastic strain, plastic strain rate, and yield stress
      
        double delGamma = 0.;
        Matrix3 nn=(0.);
        
        computePlasticStateViaRadialReturn(trialS, delT, matl, idx, state, nn, delGamma);
        
        Matrix3 tensorEtaPlasticInc = nn * delGamma;
        pStress_new[idx]            = trialStress - tensorEtaPlasticInc*(2.0*state->shearModulus);
        pPlasticStrain_new[idx]     = state->plasticStrain;
        pPlasticStrainRate_new[idx] = state->plasticStrainRate;

        // Update internal Cauchy stresses (only for viscoelasticity)
        Matrix3 dp = tensorEtaPlasticInc/delT;
        d_devStress->updateInternalStresses(idx, dp, defState, delT);

        computeEPlasticTangentModulus(bulk, shear, delGamma, trialS,
                                      idx, state, D, true);
      }

      // Compute K matrix = Kmat + Kgeo
      computeStiffnessMatrix(B, Bnl, D, pStress[idx], volold,
                             pVolume_deformed[idx], Kmatrix);

      // Assemble into global K matrix
      for (int ii = 0; ii < 24; ii++){
        for (int jj = 0; jj < 24; jj++){
          v[24*ii+jj] = Kmatrix[ii][jj];
        }
      }
      solver->fillMatrix(24,dof,24,dof,v);

      delete state;
    }
    delete interpolator;
  }
}

//______________________________________________________________________
/*! Compute the elastic tangent modulus tensor for isotropic
    materials
    Assume: [stress] = [s11 s22 s33 s12 s23 s31]
            [strain] = [e11 e22 e33 2e12 2e23 2e31] 
*/
//______________________________________________________________________
//
void 
ElasticPlasticHP::computeElasticTangentModulus(const double& K,
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
//______________________________________________________________________
//
/*! Compute the elastic-plastic tangent modulus tensor for isotropic
  materials for use in the implicit stress update
  Assume: [stress] = [s11 s22 s33 s12 s23 s31]
  [strain] = [e11 e22 e33 2e12 2e23 2e31] 
  Uses alogorithm for small strain plasticity (Simo 1998, p.124)
*/
void 
ElasticPlasticHP::computeEPlasticTangentModulus(const double& K,
                                              const double& mu,
                                              const double& delGamma,
                                              const Matrix3& trialStress,
                                              const particleIndex idx,
                                              PlasticityState* state,
                                              double Cep[6][6],
                                              bool consistent)
{

  double normTrialS = trialStress.Norm();
  Matrix3 n         = trialStress/normTrialS;
  
  // Compute theta and theta_bar
  double twomu = 2.0*mu;
  double dsigYdep = d_flow->evalDerivativeWRTPlasticStrain(state, idx);

  double theta = 1.0;
  if (consistent) {
    theta = 1.0 - (twomu*delGamma)/normTrialS;
  } 
  double  thetabar = 1.0/(1.0 + dsigYdep/(3.0*mu)) - (1.0 - theta);

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
  Cep[0][3] = -0.5*twomuthetabarn11*n(0,1);
  Cep[0][4] = -0.5*twomuthetabarn11*n(1,2);
  Cep[0][5] = -0.5*twomuthetabarn11*n(2,0);
  
  Cep[1][0] = Cep[0][1];
  Cep[1][1] = kfourmu3theta - twomuthetabarn22*n(1,1); 
  Cep[1][2] = ktwomu3theta - twomuthetabarn22*n(2,2);
  Cep[1][3] = -0.5*twomuthetabarn22*n(0,1);
  Cep[1][4] = -0.5*twomuthetabarn22*n(1,2);
  Cep[1][5] = -0.5*twomuthetabarn22*n(2,0);

  Cep[2][0] = Cep[0][2];
  Cep[2][1] = Cep[1][2];
  Cep[2][2] = kfourmu3theta - twomuthetabarn33*n(2,2); 
  Cep[2][3] = -0.5*twomuthetabarn33*n(0,1);
  Cep[2][4] = -0.5*twomuthetabarn33*n(1,2);
  Cep[2][5] = -0.5*twomuthetabarn33*n(2,0);

  Cep[3][0] = Cep[0][3];
  Cep[3][1] = Cep[1][3];
  Cep[3][2] = Cep[2][3];
  Cep[3][3] =  0.5*(twomutheta - twomuthetabarn12*n(0,1)); 
  Cep[3][4] = -0.5*twomuthetabarn12*n(1,2);
  Cep[3][5] = -0.5*twomuthetabarn12*n(2,0);

  Cep[4][0] = Cep[0][4];
  Cep[4][1] = Cep[1][4];
  Cep[4][2] = Cep[2][4];
  Cep[4][3] = Cep[3][4];
  Cep[4][4] =  0.5*(twomutheta - twomuthetabarn23*n(1,2)); 
  Cep[4][5] = -0.5*twomuthetabarn23*n(2,0);

  Cep[5][0] = Cep[0][5];
  Cep[5][1] = Cep[1][5];
  Cep[5][2] = Cep[2][5];
  Cep[5][3] = Cep[3][5];
  Cep[5][4] = Cep[4][5];
  Cep[5][5] =  0.5*(twomutheta - twomuthetabarn31*n(2,0)); 
}
//______________________________________________________________________
//
/*! Compute K matrix */
void 
ElasticPlasticHP::computeStiffnessMatrix(const double B[6][24],
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
    for(int jj = ii;jj<24;jj++){
      Kmatrix[ii][jj] =  Kmat[ii][jj]*vol_old + Kgeo[ii][jj]*vol_new;
    }
  }
  for(int ii = 0;ii<24;ii++){
    for(int jj = 0;jj<ii;jj++){
      Kmatrix[ii][jj] =  Kmatrix[jj][ii];
    }
  }
}

void 
ElasticPlasticHP::BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
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
//______________________________________________________________________
//
void 
ElasticPlasticHP::carryForward(const PatchSubset* patches,
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
    constParticleVariable<Matrix3> pRotation;
    constParticleVariable<double>  pPlasticStrain, pDamage, pPorosity; 
    constParticleVariable<double>  pStrainRate, pPlasticStrainRate;
    constParticleVariable<int>     pLocalized;

    old_dw->get(pRotation,       pRotationLabel,       pset);
    old_dw->get(pStrainRate,     pStrainRateLabel,     pset);
    old_dw->get(pPlasticStrain,  pPlasticStrainLabel,  pset);
    old_dw->get(pPlasticStrainRate,  pPlasticStrainRateLabel,  pset);
    old_dw->get(pDamage,         pDamageLabel,         pset);
    old_dw->get(pPorosity,       pPorosityLabel,       pset);
    old_dw->get(pLocalized,      pLocalizedLabel,      pset);

    ParticleVariable<Matrix3>      pRotation_new;
    ParticleVariable<double>       pPlasticStrain_new, pDamage_new;
    ParticleVariable<double>       pPorosity_new, pStrainRate_new, pPlasticStrainRate_new;
    ParticleVariable<int>          pLocalized_new;

    new_dw->allocateAndPut(pRotation_new,    
                           pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pPlasticStrainRate_new,      
                           pPlasticStrainRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pPorosity_new,      
                           pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);

    // Get the plastic strain
    d_flow->getInternalVars(pset, old_dw);
    d_flow->allocateAndPutRigid(pset, new_dw);
    
    d_flow->getInternalVars(pset, old_dw);
    d_flow->allocateAndPutRigid(pset, new_dw);    
    

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pRotation_new[idx] = pRotation[idx];
      pStrainRate_new[idx] = pStrainRate[idx];
      pPlasticStrain_new[idx] = pPlasticStrain[idx];
      pPlasticStrainRate_new[idx] = pPlasticStrainRate[idx];
      pDamage_new[idx] = pDamage[idx];
      pPorosity_new[idx] = pPorosity[idx];
      pLocalized_new[idx] = pLocalized[idx];
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
void 
ElasticPlasticHP::addRequiresDamageParameter(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}
//__________________________________
//
void 
ElasticPlasticHP::getDamageParameter(const Patch* patch,
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
         
// Compute the elastic tangent modulus tensor for isotropic
// materials (**NOTE** can get rid of one copy operation if needed)
void 
ElasticPlasticHP::computeElasticTangentModulus(double bulk,
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
ElasticPlasticHP::updatePorosity(const Matrix3& D,
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
//______________________________________________________________________
//
// Calculate the void nucleation factor
inline double 
ElasticPlasticHP::voidNucleationFactor(double ep)
{
  double temp = (ep - d_porosity.en)/d_porosity.sn;
  double A = d_porosity.fn/(d_porosity.sn*sqrt(2.0*M_PI))*
    exp(-0.5*temp*temp);
  return A;
}

/* Hardcoded specific heat computation for 4340 steel */
/*
double 
ElasticPlasticHP::computeSpecificHeat(double T)
{
  // Specific heat model for 4340 steel (SI units)
  double Tc = 1040.0;
  if (T == Tc) {
    T = T - 1.0;
  }
  double Cp = 500.0;
  if (T < Tc) {
    double t = 1 - T/Tc;
    d_Cp.A = 190.14;
    d_Cp.B = 273.75;
    d_Cp.C = 418.30;
    d_Cp.n = 0.2;
    Cp = d_Cp.A - d_Cp.B*t + d_Cp.C/pow(t, d_Cp.n);
  } else {
    double t = T/Tc - 1.0;
    d_Cp.A = 465.21;
    d_Cp.B = 267.52;
    d_Cp.C = 58.16;
    d_Cp.n = 0.35;
    Cp = d_Cp.A + d_Cp.B*t + d_Cp.C/pow(t, d_Cp.n);
  }
  return Cp;

  // Specific heat model for HY-100 steel
  //return 1.0e3*(d_Cp.A + d_Cp.B*T + d_Cp.C/(T*T));
}
*/
//______________________________________________________________________
//
double ElasticPlasticHP::computeRhoMicroCM(double pressure,
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
void ElasticPlasticHP::computePressEOSCM(double rho_cur,double& pressure,
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
double ElasticPlasticHP::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}
