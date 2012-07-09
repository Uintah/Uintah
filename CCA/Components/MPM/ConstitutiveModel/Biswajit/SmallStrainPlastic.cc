/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/SmallStrainPlastic.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/StabilityCheckFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/FlowStressModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DamageModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/MeltingTempModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/SpecificHeatModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/KinematicHardeningModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/YieldConditionFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/PressureModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/ShearModulusModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/ModelState.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/LinearInterpolator.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
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
#include <Core/Exceptions/InvalidValue.h>


using namespace std;
using namespace Uintah;

static DebugStream cout_EP("SSEP",false);
static DebugStream cout_EP1("SSEP1",false);
static DebugStream CSTi("SSEPi",false);
static DebugStream CSTir("SSEPir",false);

SmallStrainPlastic::SmallStrainPlastic(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM()
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  d_initialData.CTE = 1.0e-5; // default is per K
  ps->get("coeff_thermal_expansion", d_initialData.CTE);
  d_initialData.Chi = 0.9;
  ps->get("taylor_quinney_coeff",d_initialData.Chi);
  d_initialData.sigma_crit = 2.0e9; // default is Pa
  ps->get("critical_stress", d_initialData.sigma_crit);

  d_doIsothermal = false;
  d_isothermal = 1.0;
  ps->get("isothermal", d_doIsothermal);
  if (d_doIsothermal) d_isothermal = 0.0;

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

  d_eos = UintahBB::PressureModelFactory::create(ps);
  d_eos->setBulkModulus(d_initialData.Bulk);
  if(!d_eos){
    ostringstream desc;
    desc << "An error occured in the PressureModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_shear = UintahBB::ShearModulusModelFactory::create(ps);
  if (!d_shear) {
    ostringstream desc;
    desc << "SmallStrainPlastic::Error in shear modulus model factory" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  d_melt = MeltingTempModelFactory::create(ps);
  if (!d_melt) {
    ostringstream desc;
    desc << "SmallStrainPlastic::Error in melting temp model factory" << endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_computeSpecificHeat = false;
  ps->get("compute_specific_heat",d_computeSpecificHeat);
  d_Cp = SpecificHeatModelFactory::create(ps);

  d_yield = UintahBB::YieldConditionFactory::create(ps);
  if(!d_yield){
    ostringstream desc;
    desc << "An error occured in the YieldConditionFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_plastic = FlowStressModelFactory::create(ps);
  if(!d_plastic){
    ostringstream desc;
    desc << "An error occured in the FlowStressModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  d_kinematic = UintahBB::KinematicHardeningModelFactory::create(ps);
  if(!d_kinematic){
    ostringstream desc;
    desc << "An error occured in the KinematicHardeningModelFactory that has \n"
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
  
  d_stable = StabilityCheckFactory::create(ps);
  if(!d_stable) cerr << "Stability check disabled\n";

  setErosionAlgorithm();
  getInitialPorosityData(ps);
  getInitialDamageData(ps);
  initializeLocalMPMLabels();

}

SmallStrainPlastic::SmallStrainPlastic(const SmallStrainPlastic* cm) :
  ConstitutiveModel(cm), ImplicitCM(cm)
{
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  d_initialData.CTE = cm->d_initialData.CTE;
  d_initialData.Chi = cm->d_initialData.Chi;
  d_initialData.sigma_crit = cm->d_initialData.sigma_crit;

  d_tol = cm->d_tol ;
  d_useModifiedEOS = cm->d_useModifiedEOS;
  d_isothermal = cm->d_isothermal;

  d_initialMaterialTemperature = cm->d_initialMaterialTemperature ;
  d_checkTeplaFailureCriterion = cm->d_checkTeplaFailureCriterion;
  d_doMelting = cm->d_doMelting;
  d_checkStressTriax = cm->d_checkStressTriax;

  d_setStressToZero = cm->d_setStressToZero;
  d_allowNoTension = cm->d_allowNoTension;

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

  d_eos = UintahBB::PressureModelFactory::createCopy(cm->d_eos);
  d_eos->setBulkModulus(d_initialData.Bulk);
  d_shear = UintahBB::ShearModulusModelFactory::createCopy(cm->d_shear);
  d_melt = MeltingTempModelFactory::createCopy(cm->d_melt);
  d_computeSpecificHeat = cm->d_computeSpecificHeat;
  d_Cp = SpecificHeatModelFactory::createCopy(cm->d_Cp);

  d_yield = UintahBB::YieldConditionFactory::createCopy(cm->d_yield);
  d_plastic = FlowStressModelFactory::createCopy(cm->d_plastic);
  d_kinematic = UintahBB::KinematicHardeningModelFactory::createCopy(cm->d_kinematic);
  d_damage = DamageModelFactory::createCopy(cm->d_damage);
  d_stable = StabilityCheckFactory::createCopy(cm->d_stable);
  
  initializeLocalMPMLabels();
}

SmallStrainPlastic::~SmallStrainPlastic()
{
  // Destructor 
  VarLabel::destroy(pStrainRateLabel);
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainRateLabel);
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pPorosityLabel);
  VarLabel::destroy(pLocalizedLabel);

  VarLabel::destroy(pStrainRateLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pPlasticStrainRateLabel_preReloc);
  VarLabel::destroy(pDamageLabel_preReloc);
  VarLabel::destroy(pPorosityLabel_preReloc);
  VarLabel::destroy(pLocalizedLabel_preReloc);

  delete d_eos;
  delete d_shear;
  delete d_melt;
  delete d_Cp;
  delete d_yield;
  delete d_plastic;
  delete d_kinematic;
  delete d_damage;
  delete d_stable;
}


void SmallStrainPlastic::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","small_strain_plastic");
  }
  
  cm_ps->appendElement("bulk_modulus",d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",d_initialData.Shear);
  cm_ps->appendElement("coeff_thermal_expansion", d_initialData.CTE);
  cm_ps->appendElement("taylor_quinney_coeff",d_initialData.Chi);
  cm_ps->appendElement("critical_stress", d_initialData.sigma_crit);
  cm_ps->appendElement("isothermal", d_doIsothermal);
  cm_ps->appendElement("tolerance",d_tol);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
  cm_ps->appendElement("initial_material_temperature",
                       d_initialMaterialTemperature);
  cm_ps->appendElement("check_TEPLA_failure_criterion",
                       d_checkTeplaFailureCriterion);
  cm_ps->appendElement("do_melting",d_doMelting);
  cm_ps->appendElement("check_max_stress_failure",d_checkStressTriax);
  cm_ps->appendElement("compute_specific_heat",d_computeSpecificHeat);

  d_eos->outputProblemSpec(cm_ps);
  d_shear->outputProblemSpec(cm_ps);
  d_melt->outputProblemSpec(cm_ps);
  d_Cp->outputProblemSpec(cm_ps);
  d_yield->outputProblemSpec(cm_ps);
  d_plastic->outputProblemSpec(cm_ps);
  d_kinematic->outputProblemSpec(cm_ps);
  d_damage->outputProblemSpec(cm_ps);
  d_stable->outputProblemSpec(cm_ps);

  cm_ps->appendElement("evolve_porosity",d_evolvePorosity);
  cm_ps->appendElement("initial_mean_porosity",d_porosity.f0);
  cm_ps->appendElement("initial_std_porosity",d_porosity.f0_std);
  cm_ps->appendElement("critical_porosity",d_porosity.fc);
  cm_ps->appendElement("frac_nucleation",d_porosity.fn);
  cm_ps->appendElement("meanstrain_nucleation",d_porosity.en);
  cm_ps->appendElement("stddevstrain_nucleation",d_porosity.sn);
  cm_ps->appendElement("initial_porosity_distrib",d_porosity.porosityDist);

  cm_ps->appendElement("evolve_damage",d_evolveDamage);
  cm_ps->appendElement("initial_mean_scalar_damage",d_scalarDam.D0);
  cm_ps->appendElement("initial_std_scalar_damage",d_scalarDam.D0_std);
  cm_ps->appendElement("critical_scalar_damage",d_scalarDam.Dc);
  cm_ps->appendElement("initial_scalar_damage_distrib",
                       d_scalarDam.scalarDamageDist);

}


SmallStrainPlastic* SmallStrainPlastic::clone()
{
  return scinew SmallStrainPlastic(*this);
}


void
SmallStrainPlastic::initializeLocalMPMLabels()
{
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
}

void 
SmallStrainPlastic::getInitialPorosityData(ProblemSpecP& ps)
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

void 
SmallStrainPlastic::getInitialDamageData(ProblemSpecP& ps)
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

void 
SmallStrainPlastic::setErosionAlgorithm()
{
  d_setStressToZero = false;
  d_allowNoTension = false;
  if (flag->d_doErosion) {
    if (flag->d_erosionAlgorithm == "AllowNoTension") 
      d_allowNoTension = true;
    else if (flag->d_erosionAlgorithm == "ZeroStress") 
      d_setStressToZero = true;
  }
}

void 
SmallStrainPlastic::addParticleState(std::vector<const VarLabel*>& from,
                                 std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pStrainRateLabel);
  from.push_back(pPlasticStrainLabel);
  from.push_back(pPlasticStrainRateLabel);
  from.push_back(pDamageLabel);
  from.push_back(pPorosityLabel);
  from.push_back(pLocalizedLabel);

  to.push_back(pStrainRateLabel_preReloc);
  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pPlasticStrainRateLabel_preReloc);
  to.push_back(pDamageLabel_preReloc);
  to.push_back(pPorosityLabel_preReloc);
  to.push_back(pLocalizedLabel_preReloc);

  // Add the particle state for the plasticity models
  d_plastic->addParticleState(from, to);
  d_kinematic->addParticleState(from, to);
}

void 
SmallStrainPlastic::addInitialComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(pStrainRateLabel,    matlset);
  task->computes(pPlasticStrainLabel, matlset);
  task->computes(pPlasticStrainRateLabel, matlset);
  task->computes(pDamageLabel,        matlset);
  task->computes(pPorosityLabel,      matlset);
  task->computes(pLocalizedLabel,     matlset);
 
  // Add internal evolution variables computed by plasticity model
  d_plastic->addInitialComputesAndRequires(task, matl, patch);
  d_kinematic->addInitialComputesAndRequires(task, matl, patch);
}

void 
SmallStrainPlastic::initializeCMData(const Patch* patch,
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
  //cout << "Initialize CM Data in SmallStrainPlastic" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pPlasticStrain, pDamage, pPorosity, 
                            pPlasticStrainRate, pStrainRate;
  ParticleVariable<int>     pLocalized;

  new_dw->allocateAndPut(pStrainRate, pStrainRateLabel, pset);
  new_dw->allocateAndPut(pPlasticStrain, pPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pPlasticStrainRate, pPlasticStrainRateLabel, pset);
  new_dw->allocateAndPut(pDamage, pDamageLabel, pset);
  new_dw->allocateAndPut(pLocalized, pLocalizedLabel, pset);
  new_dw->allocateAndPut(pPorosity, pPorosityLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){

    pStrainRate[*iter] = 0.0;
    pPlasticStrain[*iter] = 0.0;
    pPlasticStrainRate[*iter] = 0.0;
    pDamage[*iter] = d_damage->initialize();
    pPorosity[*iter] = d_porosity.f0;
    pLocalized[*iter] = 0;
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

  // Initialize the data for the plasticity model
  d_plastic->initializeInternalVars(pset, new_dw);
  d_kinematic->initializeBackStress(pset, new_dw);
}

void 
SmallStrainPlastic::computeStableTimestep(const Patch* patch,
                                      const MPMMaterial* matl,
                                      DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);

  constParticleVariable<double> pMass, pVol_new;
  constParticleVariable<Vector> pVelocity;

  new_dw->get(pMass,     lb->pMassLabel,     pset);
  new_dw->get(pVol_new,  lb->pVolumeLabel,   pset);
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector waveSpeed(1.e-12,1.e-12,1.e-12);

  double shear = d_initialData.Shear;
  double bulk = d_initialData.Bulk;

  ParticleSubset::iterator iter = pset->begin(); 
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    Vector pvelocity_idx = pVelocity[idx];
    if(pMass[idx] > 0){
      c_dil = sqrt((bulk + 4.0*shear/3.0)*pVol_new[idx]/pMass[idx]);
    } else {
      c_dil = 0.0;
      pvelocity_idx = Vector(0.0,0.0,0.0);
    }
    waveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),waveSpeed.x()),
                     Max(c_dil+fabs(pvelocity_idx.y()),waveSpeed.y()),
                     Max(c_dil+fabs(pvelocity_idx.z()),waveSpeed.z()));
  }

  waveSpeed = dx/waveSpeed;
  double delT_new = waveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void 
SmallStrainPlastic::addComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  if (flag->d_integrator == MPMFlags::Implicit) {
    addSharedCRForImplicit(task, matlset, patches);
  } else {
    addSharedCRForHypoExplicit(task, matlset, patches);
  }

  // Other constitutive model and input dependent computes and requires
  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 

  task->requires(Task::OldDW, pStrainRateLabel,       matlset, gnone);
  task->requires(Task::OldDW, pPlasticStrainLabel,    matlset, gnone);
  task->requires(Task::OldDW, pPlasticStrainRateLabel,matlset, gnone);
  task->requires(Task::OldDW, pDamageLabel,           matlset, gnone);
  task->requires(Task::OldDW, pPorosityLabel,         matlset, gnone);
  task->requires(Task::OldDW, pLocalizedLabel,        matlset, gnone);

  task->computes(pStrainRateLabel_preReloc,     matlset);
  task->computes(pPlasticStrainLabel_preReloc,  matlset);
  task->computes(pPlasticStrainRateLabel_preReloc,  matlset);
  task->computes(pDamageLabel_preReloc,         matlset);
  task->computes(pPorosityLabel_preReloc,       matlset);
  task->computes(pLocalizedLabel_preReloc,      matlset);

  // Add internal evolution variables computed by plasticity model
  d_plastic->addComputesAndRequires(task, matl, patches);
  d_kinematic->addComputesAndRequires(task, matl, patches);
}

void 
SmallStrainPlastic::computeStressTensor(const PatchSubset* patches,
                                    const MPMMaterial* matl,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  if (flag->d_integrator == MPMFlags::Implicit) {
    computeStressTensorImplicit(patches, matl, old_dw, new_dw);
  } else {
    computeStressTensorExplicit(patches, matl, old_dw, new_dw);
  }
}

void 
SmallStrainPlastic::computeStressTensorExplicit(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  // General stuff
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  Matrix3 defGrad_new; defGrad_new.Identity(); 
  Matrix3 rightStretch; rightStretch.Identity(); 
  Matrix3 rotation; rotation.Identity(); 
  Matrix3 rateOfDef_new(0.0); 
  Matrix3 rateOfDef_dev_new(0.0); 
  Matrix3 sigma_old(0.0); 
  Matrix3 sigma_dev_old(0.0); 
  Matrix3 sigma_dev_new(0.0);
  Matrix3 backStress_old(0.0); 
  Matrix3 backStress_dev_old(0.0); 
  Matrix3 backStress_new(0.0); 

  Vector waveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double CTE = d_initialData.CTE;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();
  double sqrtThreeTwo = sqrt(1.5);
  double sqrtTwoThird = 1.0/sqrtThreeTwo;
  double totalStrainEnergy = 0.0;

  Ghost::GhostType  gac = Ghost::AroundCells;

  // Do thermal expansion?
  if (!flag->d_doThermalExpansion) {
    CTE = 0.0;
  }

  // Loop thru patches
  for(int patchIndex=0; patchIndex<patches->size(); patchIndex++){
    const Patch* patch = patches->get(patchIndex);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector>    d_S(interpolator->size());
    vector<double>    S(interpolator->size());
    
    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get the set of particles
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // GET GLOBAL DATA 

    // Get the deformation gradient (F)
    constParticleVariable<Matrix3>  pDefGrad;
    old_dw->get(pDefGrad, lb->pDeformationMeasureLabel, pset);

    // Get the particle location, particle size, particle mass, particle volume
    constParticleVariable<Point>  px;
    constParticleVariable<Matrix3> psize;
    constParticleVariable<double> pMass, pVol_old;
    old_dw->get(px,       lb->pXLabel,      pset);
    old_dw->get(psize,    lb->pSizeLabel,   pset);
    old_dw->get(pMass,    lb->pMassLabel,   pset);
    old_dw->get(pVol_old, lb->pVolumeLabel, pset);

    // Get the velocity from the grid and particle velocity
    constParticleVariable<Vector> pVelocity;
    constNCVariable<Vector>       gVelocity;
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);
    new_dw->get(gVelocity, lb->gVelocityStarLabel, dwi, patch, gac, NGN);

    // Get the particle stress and temperature
    constParticleVariable<Matrix3> pStress_old;
    constParticleVariable<double>  pTempPrev, pTemp_old;
    old_dw->get(pStress_old, lb->pStressLabel,       pset);
    old_dw->get(pTempPrev,   lb->pTempPreviousLabel, pset); 
    old_dw->get(pTemp_old,   lb->pTemperatureLabel,  pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // GET LOCAL DATA 
    constParticleVariable<double> pPlasticStrain_old, pDamage_old, pPorosity_old, 
                                  pStrainRate_old, pPlasticStrainRate_old;
    old_dw->get(pPlasticStrain_old,     pPlasticStrainLabel,     pset);
    old_dw->get(pDamage_old,            pDamageLabel,            pset);
    old_dw->get(pStrainRate_old,        pStrainRateLabel,        pset);
    old_dw->get(pPlasticStrainRate_old, pPlasticStrainRateLabel, pset);
    old_dw->get(pPorosity_old,          pPorosityLabel,          pset);

    constParticleVariable<int> pLocalized_old;
    old_dw->get(pLocalized_old, pLocalizedLabel, pset);

    // Create and allocate arrays for storing the updated information
    // GLOBAL
    ParticleVariable<Matrix3> pDefGrad_new, pStress_new;
    ParticleVariable<double>  pVol_new;
    ParticleVariable<double> pdTdt, p_q;

    new_dw->allocateAndPut(pDefGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVol_new, 
                           lb->pVolumeLabel_preReloc,             pset);
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel_preReloc,        pset);
    new_dw->allocateAndPut(p_q,   lb->p_qLabel_preReloc,          pset);

    // LOCAL
    ParticleVariable<double>  pPlasticStrain_new, pDamage_new, pPorosity_new, 
                              pStrainRate_new, pPlasticStrainRate_new;
    ParticleVariable<int>     pLocalized_new;
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

    // Get the plastic strain and back stress and allocate
    // space for the updated internal variables and back stress
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);
    d_kinematic->getBackStress(pset, old_dw);
    d_kinematic->allocateAndPutBackStress(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      //-----------------------------------------------------------------------
      // Stage 1:
      //-----------------------------------------------------------------------
      // Calculate the velocity gradient (L) from the grid velocity
      Matrix3 velGrad(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDefGrad[idx]);

        computeVelocityGradient(velGrad,ni,d_S, oodx, gVelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                   psize[idx],pDefGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gVelocity,px[idx]);
      }

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      Matrix3 defGradInc = velGrad*delT + one;
      defGrad_new = defGradInc*pDefGrad[idx];
      pDefGrad_new[idx] = defGrad_new;
      double J_new = defGrad_new.Determinant();

      // If the erosion algorithm sets the stress to zero then don't allow
      // any deformation.
      if(d_setStressToZero && pLocalized_old[idx]){
        pDefGrad_new[idx] = pDefGrad[idx];
        J_new = pDefGrad[idx].Determinant();
      }

      // Check 1: Check for negative Jacobian (determinant of deformation gradient)
      if (!(J_new > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" 
             << " in particle " << idx << endl;
        cerr << "l = " << velGrad << endl;
        cerr << "F_old = " << pDefGrad[idx] << endl;
        cerr << "F_inc = " << defGradInc << endl;
        cerr << "F_new = " << defGrad_new << endl;
        cerr << "J_old = " << pDefGrad[idx].Determinant() << endl;
        cerr << "J_new = " << J_new << endl;
        throw ParameterNotFound("**ERROR**:InvalidValue: J < 0.0", __FILE__, __LINE__);
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J_new;
      pVol_new[idx]=pMass[idx]/rho_cur;

      // Compute polar decomposition of F (F = RU)
      pDefGrad[idx].polarDecompositionRMB(rightStretch, rotation);

      // Calculate rate of deformation tensor (D)
      rateOfDef_new = (velGrad + velGrad.Transpose())*0.5;

      // Rotate the total rate of deformation tensor back to the 
      // material configuration
      rateOfDef_new = (rotation.Transpose())*(rateOfDef_new*rotation);
      pStrainRate_new[idx] = sqrtTwoThird*rateOfDef_new.Norm();

      // Calculate the deviatoric part of the non-thermal part
      // of the rate of deformation tensor
      rateOfDef_dev_new = rateOfDef_new - one*(rateOfDef_new.Trace()/3.0);

      // Rotate the Cauchy stress back to the 
      // material configuration and calculate the deviatoric part
      sigma_old = pStress_old[idx];
      sigma_old = (rotation.Transpose())*(sigma_old*rotation);
      double pressure_old = sigma_old.Trace()/3.0;
      sigma_dev_old = sigma_old - one*pressure_old;

      // Get the back stress from the kinematic hardening model and rotate
      d_kinematic->getBackStress(idx, backStress_old);
      backStress_old = (rotation.Transpose())*(backStress_old*rotation);
      backStress_dev_old = backStress_old - one*(backStress_old.Trace()/3.0);
      backStress_new = backStress_old;
      
      // Set up the ModelState (for t_n)
      UintahBB::ModelState* state = scinew UintahBB::ModelState();
      state->strainRate          = pStrainRate_new[idx];
      state->plasticStrainRate   = pPlasticStrainRate_old[idx];
      state->plasticStrain       = pPlasticStrain_old[idx];
      state->pressure            = pressure_old;
      state->temperature         = pTemp_old[idx];
      state->initialTemperature  = d_initialMaterialTemperature;
      state->density             = rho_cur;
      state->initialDensity      = rho_0;
      state->volume              = pVol_new[idx];
      state->initialVolume       = pMass[idx]/rho_0;
      state->bulkModulus         = bulk;
      state->initialBulkModulus  = bulk;
      state->shearModulus        = shear;
      state->initialShearModulus = shear;
      state->meltingTemp         = Tm;
      state->initialMeltTemp     = Tm;
      state->specificHeat        = matl->getSpecificHeat();
      state->porosity            = pPorosity_old[idx];
      state->backStress          = backStress_old;

      // Compute the pressure
      double pressure_new = d_eos->computePressure(matl, state, defGrad_new, 
                                                   rateOfDef_new, delT);
      state->pressure = pressure_new;

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
      // Stage 2: Elastic-plastic stress update
      //-----------------------------------------------------------------------
      // Keep the temperature constant over the time step
      double T_new = state->temperature;

      // Calculate flow stress 
      double flowStress = d_plastic->computeFlowStress(state, delT, d_tol, 
                                                       matl, idx);
      state->yieldStress = flowStress;

      // Material has melted if flowStress <= 0.0
      bool melted = false;
      bool plastic = false;
      if (T_new > Tm_cur || flowStress <= 0.0) {

        melted = true;

        // Set the deviatoric stress to zero
        if (d_doMelting) sigma_dev_old = 0.0;

        d_plastic->updateElastic(idx);
        d_kinematic->updateBackStress(idx, Matrix3(0.0));

      } else {

        plastic = true;

        // Integrate the stress rate equation to get a trial deviatoric stress
        Matrix3 sigma_dev_trial = sigma_dev_old + rateOfDef_dev_new*(2.0*mu_cur*delT);

        // Compute xi_trial = s_trial - beta_n
        Matrix3 xi_trial = sigma_dev_trial - backStress_dev_old;

        // Check whether the step is elastic or plastic
        double f_0 = d_yield->evalYieldCondition(xi_trial, state);
        if (isnan(f_0)) {
          cout << "idx = " << idx 
               << " epdot = " << state->plasticStrainRate 
               << " ep = " << state->plasticStrain 
               << " T = " << state->temperature 
               << " p = " << state->pressure 
               << " sigy = " << state->yieldStress << endl;
          throw InvalidValue("**ERROR**:SmallStrainPlastic: f_0 = nan.", __FILE__, __LINE__);
        }

        if (f_0 < 0.0) {

          // Set the elastic stress to the trial stress
          sigma_dev_new = sigma_dev_trial;

          // Update the internal variables
          d_plastic->updateElastic(idx);
          d_kinematic->updateBackStress(idx, backStress_old);

        } else {

          Matrix3 xi_n = sigma_dev_old - backStress_dev_old;
          if (!(xi_n.NormSquared() > 0.0)) {
            cout << "Particle idx = " << idx
                 << " has zero deviatoric stress.  Reduce initial time step size"
                 << " and restart. " << endl;
            throw InvalidValue("**ERROR**:SmallStrainPlastic: Lower time step", __FILE__, __LINE__);
          }
 
          // Get ep_n, phi_n
          double ep_n = state->plasticStrain;
          double phi_n = state->porosity;

          // Compute r_k, h_k
          Matrix3 xi_k = xi_trial;
          Matrix3 r_k(0.0);
          d_yield->eval_df_dsigma(xi_k, state, r_k);
          double h_alpha_k = d_yield->eval_h_alpha(xi_k, state);
          double A_k = voidNucleationFactor(state->plasticStrain);
          double h_phi_k = d_yield->eval_h_phi(xi_k, A_k, state);
          Matrix3 h_beta_k(0.0);
          d_kinematic->eval_h_beta(r_k, state, h_beta_k);
          Matrix3 r_k_dev = r_k - one*(r_k.Trace()/3.0);
          Matrix3 h_beta_k_dev = h_beta_k - one*(h_beta_k.Trace()/3.0);
          Matrix3 term1_k = r_k_dev*(2.0*mu_cur) + h_beta_k_dev;

          // Iterate to find DeltaGamma
          int count = 0;
          double Delta_gamma = 0.0;
          double f_k = f_0;

          while (f_k > d_tol) {

            // Get the derivatives of the yield function
            Matrix3 df_dxi_k(0.0);
            d_yield->eval_df_dxi(xi_k, state, df_dxi_k);
            double dsigy_dep_k = d_plastic->evalDerivativeWRTPlasticStrain(state, idx);
            double df_dep_k = d_yield->eval_df_dep(xi_k, dsigy_dep_k, state);
            double df_dphi_k = d_yield->eval_df_dphi(xi_k, state);
            
            // compute delta gamma (k)
            double denom = df_dxi_k.Contract(term1_k) - h_alpha_k*df_dep_k - h_phi_k*df_dphi_k;
            double delta_gamma_k = f_k/denom;
            if (isnan(f_k) || isnan(delta_gamma_k)) {
              cout << "idx = " << idx << " iter = " << count 
                   << " f_k = " << f_k << " delta_gamma_k = " << delta_gamma_k 
                   << " sigy = " << state->yieldStress
                   << " dsigy_dep_k = " << dsigy_dep_k << " df_dep_k = " << df_dep_k
                   << " epdot = " << state->plasticStrainRate 
                   << " ep = " << state->plasticStrain << endl;
              cout << "df_dxi = \n" << df_dxi_k << "\n term1 = " << term1_k
                   << "\n h_alpha = " << h_alpha_k << " df_dep = " << df_dep_k
                   << "\n h_phi = " << h_phi_k << " df_dphi = " << df_dphi_k 
                   << " denom = " << denom << endl;
              throw InvalidValue("**ERROR**:SmallStrainPlastic: Found nan.", __FILE__, __LINE__);
            }

            // Update Delta_gamma
            double Delta_gamma_old = Delta_gamma;
            Delta_gamma += delta_gamma_k;

            if (Delta_gamma < 0.0) {
              cout << "Delta_gamma = " << Delta_gamma << endl;
              cout << "h_alpha = " << h_alpha_k << " delta_gamma = " << delta_gamma_k
                   << " ep = " << state->plasticStrain << endl;
              cout << "idx = " << idx << " iter = " << count 
                   << " f_k = " << f_k << " delta_gamma_k = " << delta_gamma_k 
                   << " sigy = " << state->yieldStress
                   << " dsigy_dep_k = " << dsigy_dep_k << " df_dep_k = " << df_dep_k
                   << " epdot = " << state->plasticStrainRate 
                   << " ep = " << state->plasticStrain << endl;
              cout << "xi = \n" << xi_k << "\n df_dxi:term1 = "<< df_dxi_k.Contract(term1_k)
                   << "\n df_dxi = \n" << df_dxi_k << "\n term1 = " << term1_k 
                   << "\n h_alpha = " << h_alpha_k << " df_dep = " << df_dep_k
                   << "\n h_phi = " << h_phi_k << " df_dphi = " << df_dphi_k 
                   << " denom = " << denom << endl;
              cout << "r_n_dev = \n" <<  r_k_dev
                   << "\n mu_cur = " << mu_cur 
                   << "\n h_bet_n_dev = \n" << h_beta_k_dev << endl;
            }

            /* Updated algorithm - use value of xi_k */
            // Compute r_k, h_k
            d_yield->eval_df_dsigma(xi_k, state, r_k);
            h_alpha_k = d_yield->eval_h_alpha(xi_k, state);
            A_k = voidNucleationFactor(state->plasticStrain);
            h_phi_k = d_yield->eval_h_phi(xi_k, A_k, state);
            d_kinematic->eval_h_beta(r_k, state, h_beta_k);
            r_k_dev = r_k - one*(r_k.Trace()/3.0);
            h_beta_k_dev = h_beta_k - one*(h_beta_k.Trace()/3.0);
            term1_k = r_k_dev*(2.0*mu_cur) + h_beta_k_dev;
 
            // Update ep, phi, xi
            state->plasticStrain = ep_n + Delta_gamma*h_alpha_k;
            state->porosity = phi_n + Delta_gamma*h_phi_k;
            xi_k = xi_trial - term1_k*Delta_gamma;
       
            /*  Original Algorithm */
            /*
            // compute increments of xi, ep, phi
            Matrix3 Delta_xi_k = term1*(-delta_gamma_k);
            double Delta_ep_k = h_alpha_n*delta_gamma_k;
            double Delta_phi_k = h_phi_n*delta_gamma_k;

            // Update ep, phi, xi, Delta_gamma
            state->plasticStrain += Delta_ep_k;
            state->porosity += Delta_phi_k;
            xi_k += Delta_xi_k;
            */

            if (fabs(Delta_gamma-Delta_gamma_old) < d_tol || count > 100) break;

            // Update the flow stress 
            state->yieldStress = d_plastic->computeFlowStress(state, delT, d_tol, 
                                                              matl, idx);

            // Check yield condition.  The state variable contains
            // ep_k, phi_k, beta_k
            f_k = d_yield->evalYieldCondition(xi_k, state);
       
            ++count;
          }

          // Update the back stress and deviatoric stress
          Matrix3 r_new(0.0);
          d_yield->eval_df_dsigma(xi_k, state, r_new);
          Matrix3 h_beta_new(0.0);
          d_kinematic->eval_h_beta(r_new, state, h_beta_new);
          backStress_new = backStress_old + h_beta_new*Delta_gamma;
          state->backStress = backStress_new;
          Matrix3 backStress_dev_new = backStress_new - one*(backStress_new.Trace()/3.0);
          sigma_dev_new = xi_k + backStress_dev_new;

          // Update the plastic strain rate
          double h_alpha_new = d_yield->eval_h_alpha(xi_k, state);
          state->plasticStrainRate = Delta_gamma/delT*h_alpha_new;

          // Update internal variables
          d_plastic->updatePlastic(idx, Delta_gamma);

          // Calculate rate of temperature increase due to plastic strain
          double taylorQuinney = d_initialData.Chi;
          double fac = taylorQuinney/(rho_cur*state->specificHeat);

          // Calculate Tdot (internal plastic heating rate).  This
          // is used during the solution of the heat equation.
          double Tdot = state->yieldStress*state->plasticStrainRate*fac;
          pdTdt[idx] = Tdot*d_isothermal;

          // Calculate a local change in temperature due to adiabatic
          // heating for the purpose of thermal expansion corrections.
          // If isothermal conditions exist then d_isothermal = 0.
          T_new += (Tdot*delT*d_isothermal);

        } // end of Phi if

      } // end of temperature if

      // Calculate the total stress
      double T_0 = state->initialTemperature;
      double kappa_new = d_eos->eval_dp_dJ(matl, J_new, state);
      kappa_new *= J_new;
      Matrix3 sigma_new = sigma_dev_new + one*(pressure_new - 3.0*kappa_new*CTE*(T_new - T_0));

      // If the particle has already failed, apply various erosion algorithms
      if (flag->d_doErosion) {
        if (pLocalized_old[idx]) {
          if (d_allowNoTension) {
            if (pressure_new > 0.0) sigma_new = zero;
            else sigma_new = one*pressure_new;
          }
          else if (d_setStressToZero) sigma_new = zero;
        }
      }

      //-----------------------------------------------------------------------
      // Stage 3: Compute damage 
      //-----------------------------------------------------------------------
      if (!plastic) {

        // Save the updated data
        pPlasticStrain_new[idx] = pPlasticStrain_old[idx];
        pPlasticStrainRate_new[idx] = 0.0;
        pPorosity_new[idx] = pPorosity_old[idx];
        pDamage_new[idx] = pDamage_old[idx];
        
      } else {

        // Update the plastic strain, plastic strain rate, porosity
        pPlasticStrain_new[idx] = state->plasticStrain;
        pPlasticStrainRate_new[idx] = state->plasticStrainRate;
        if (d_evolvePorosity) {
          pPorosity_new[idx] = state->porosity;
        } else {
          pPorosity_new[idx] = pPorosity_old[idx];
        }

        // Calculate the updated scalar damage parameter
        if (d_evolveDamage) {
          pDamage_new[idx] = 
            d_damage->computeScalarDamage(state->plasticStrainRate, sigma_new, 
                                          T_new, delT, matl, d_tol, 
                                          pDamage_old[idx]);
        } else {
          pDamage_new[idx] = pDamage_old[idx];
        }

      }

      //-----------------------------------------------------------------------
      // Stage 4:
      //-----------------------------------------------------------------------
      // Find if the particle has failed/localized
      pLocalized_new[idx] = pLocalized_old[idx];
      bool isLocalized = false;
      double tepla = 0.0;

      if (flag->d_doErosion) {

        // Check 1: Look at the temperature
        if (melted) isLocalized = true;

        // Check 2 and 3: Look at TEPLA and stability
        else if (plastic) {

          // Check 2: Modified Tepla rule
          if (d_checkTeplaFailureCriterion) {
            tepla = pow(pPorosity_new[idx]/d_porosity.fc,2.0) + 
                    pow(pDamage_new[idx],2.0);
            if (tepla > 1.0) isLocalized = true;
          } 

          // Check 3: Stability criterion (only if material is plastic)
          if (d_stable->doIt() && !isLocalized) {

            // Calculate values needed for tangent modulus calculation
            state->temperature = T_new;
            Tm_cur = d_melt->computeMeltingTemp(state);
            state->meltingTemp = Tm_cur ;
            mu_cur = d_shear->computeShearModulus(state);
            state->shearModulus = mu_cur ;

            state->yieldStress = d_plastic->computeFlowStress(state, delT, d_tol, 
                                                              matl, idx);
            if (!(state->yieldStress > 0.0)) isLocalized = true;
            else {

              // Get the derivatives of the yield function
              Matrix3 xi_hat = sigma_new - backStress_new;
              Matrix3 xi = xi_hat - one*(xi_hat.Trace()/3.0);
              Matrix3 rr(0.0);
              d_yield->eval_df_dsigma(xi, state, rr);
              Matrix3 df_ds(0.0), df_dbeta(0.0);
              d_yield->eval_df_ds_df_dbeta(xi, state, df_ds, df_dbeta);
              Matrix3 h_beta(0.0);
              d_kinematic->eval_h_beta(rr, state, h_beta);
              Matrix3 r_dev = rr - one*(rr.Trace()/3.0);
              Matrix3 h_beta_dev = h_beta - one*(h_beta.Trace()/3.0);
              double dsigy_dep = d_plastic->evalDerivativeWRTPlasticStrain(state, idx);
              double df_dep = d_yield->eval_df_dep(xi, dsigy_dep, state);
              double h_alpha = d_yield->eval_h_alpha(xi, state);
              double df_dphi = d_yield->eval_df_dphi(xi, state);
              double A = voidNucleationFactor(state->plasticStrain);
              double h_phi = d_yield->eval_h_phi(xi, A, state);
              double dp_dJ = d_eos->eval_dp_dJ(matl, J_new, state);
              
              // Calculate the elastic-plastic tangent modulus
              TangentModulusTensor Cep;
              d_yield->computeElasPlasTangentModulus(rr, df_ds, h_beta, df_dbeta,
                                                     h_alpha, df_dep, h_phi, df_dphi,
                                                     J_new, dp_dJ, state, Cep);
          
              // Initialize localization direction
              Vector direction(0.0,0.0,0.0);
              isLocalized = d_stable->checkStability(sigma_new, rateOfDef_new, Cep, 
                                                     direction);
            }
          }
        } 

        // Check 4: Look at maximum stress
        if (d_checkStressTriax) {

          // Compute eigenvalues of the stress tensor
          SymmMatrix3 stress(sigma_new);          
          Vector eigVal(0.0, 0.0, 0.0);
          Matrix3 eigVec;
          stress.eigen(eigVal, eigVec);
          double max_stress = Max(Max(eigVal[0],eigVal[1]), eigVal[2]);
          if (max_stress > d_initialData.sigma_crit) {
            isLocalized = true;
          }
        }

        // Use erosion algorithms to treat localized particles
        if (isLocalized) {

          // If the localized particles fail again then set their stress to zero
          if (pLocalized_old[idx]) {
            pDamage_new[idx] = 0.0;
            pPorosity_new[idx] = 0.0;
            sigma_new = zero;
          } else {

            // set the particle localization flag to true  
            pLocalized_new[idx] = 1;
            pDamage_new[idx] = 0.0;
            pPorosity_new[idx] = 0.0;

            // Apply various erosion algorithms
            if (d_allowNoTension) {
              if (pressure_new > 0.0) sigma_new = zero;
              else sigma_new = one*pressure_new;
            }
            else if (d_setStressToZero) sigma_new = zero;
          }

        }
      }

      //-----------------------------------------------------------------------
      // Stage 5:
      //-----------------------------------------------------------------------
      // Rotate the stress/backStress back to the laboratory coordinates
      // Update the stress/back stress

      // Use new rotation
      defGrad_new.polarDecompositionRMB(rightStretch, rotation);

      backStress_new = (rotation*backStress_new)*(rotation.Transpose());
      sigma_new = (rotation*sigma_new)*(rotation.Transpose());
      d_kinematic->updateBackStress(idx, backStress_new);
      pStress_new[idx] = sigma_new;
        
      // Rotate the deformation rate back to the laboratory coordinates
      rateOfDef_new = (rotation*rateOfDef_new)*(rotation.Transpose());

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        Matrix3 avgStress = (pStress_new[idx] + pStress_old[idx])*0.5;
        double pStrainEnergy = (rateOfDef_new(0,0)*avgStress(0,0) +
                                rateOfDef_new(1,1)*avgStress(1,1) +
                                rateOfDef_new(2,2)*avgStress(2,2) +
                                2.0*(rateOfDef_new(0,1)*avgStress(0,1) + 
                                     rateOfDef_new(0,2)*avgStress(0,2) +
                                     rateOfDef_new(1,2)*avgStress(1,2)))*
          pVol_new[idx]*delT;
        totalStrainEnergy += pStrainEnergy;
      }         

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      waveSpeed=Vector(Max(c_dil+fabs(pVel.x()),waveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),waveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),waveSpeed.z()));

      // Compute artificial viscosity term
      double de_s=0.;
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        Matrix3 D=(velGrad + velGrad.Transpose())*0.5;
        double Dkk=D.Trace();
        p_q[idx] = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
        de_s = -p_q[idx]*Dkk/rho_cur;
      } else {
        p_q[idx] = 0.;
        de_s = 0.;
      }
      pdTdt[idx] += de_s/state->specificHeat;

      delete state;
    }  // end loop over particles

    waveSpeed = dx/waveSpeed;
    double delT_new = waveSpeed.minComponent();

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }
    delete interpolator;
  }

  if (cout_EP.active()) 
    cout_EP << getpid() << "... End." << endl;

}

void 
SmallStrainPlastic::computeStressTensorImplicit(const PatchSubset* patches,
                                                const MPMMaterial* matl,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  throw InvalidValue("**ERROR**:SmallStrainPlastic: No implicit stress update available", 
                     __FILE__, __LINE__);
}

void 
SmallStrainPlastic::addComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patches,
                                       const bool recurse,
                                       const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForImplicit(task, matlset, patches, recurse, SchedParent);

  // Local stuff
  Ghost::GhostType  gnone = Ghost::None;
  if(SchedParent){
    task->requires(Task::ParentOldDW, lb->pTempPreviousLabel,  matlset, gnone); 
    task->requires(Task::ParentOldDW, lb->pTemperatureLabel,   matlset, gnone);
    task->requires(Task::ParentOldDW, pPlasticStrainLabel,     matlset, gnone);
    task->requires(Task::ParentOldDW, pPlasticStrainRateLabel, matlset, gnone);
    task->requires(Task::ParentOldDW, pPorosityLabel,          matlset, gnone);
  }else{
    task->requires(Task::OldDW, lb->pTempPreviousLabel,  matlset, gnone); 
    task->requires(Task::OldDW, lb->pTemperatureLabel,   matlset, gnone);
    task->requires(Task::OldDW, pPlasticStrainLabel,     matlset, gnone);
    task->requires(Task::OldDW, pPlasticStrainRateLabel, matlset, gnone);
    task->requires(Task::OldDW, pPorosityLabel,          matlset, gnone);
  }

  // Add internal evolution variables computed by plasticity model
  d_plastic->addComputesAndRequires(task, matl, patches, recurse,SchedParent);
  d_kinematic->addComputesAndRequires(task, matl, patches, recurse);
}

void 
SmallStrainPlastic::computeStressTensor(const PatchSubset* patches,
                                    const MPMMaterial* matl,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    Solver* solver,
                                    const bool )
{
  throw InvalidValue("**ERROR**:SmallStrainPlastic: No implicit stress update available", 
                     __FILE__, __LINE__);
}

/*! Compute the elastic-plastic tangent modulus tensor for isotropic
  materials for use in the implicit stress update
  Assume: [stress] = [s11 s22 s33 s12 s23 s31]
  [strain] = [e11 e22 e33 2e12 2e23 2e31] 
  Uses alogorithm for small strain plasticity (Simo 1998, p.124)
*/
void 
SmallStrainPlastic::computeEPlasticTangentModulus(const double& K,
                                              const double& mu,
                                              const double& delGamma,
                                              const double& normTrialS,
                                              const particleIndex idx,
                                              const Matrix3& n,
                                              UintahBB::ModelState* state,
                                              double Cep[6][6],
                                              bool consistent)
{
  throw InvalidValue("**ERROR**:SmallStrainPlastic: No elastic-plastic tangent modulus available", 
                     __FILE__, __LINE__);
}

/*! Compute K matrix */
void 
SmallStrainPlastic::computeStiffnessMatrix(const double B[6][24],
                                       const double Bnl[3][24],
                                       const double D[6][6],
                                       const Matrix3& sig,
                                       const double& vol_old,
                                       const double& vol_new,
                                       double Kmatrix[24][24])
{
  throw InvalidValue("**ERROR**:SmallStrainPlastic: No stiffness matrix available", 
                     __FILE__, __LINE__);
}

void 
SmallStrainPlastic::BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                           double Kgeo[24][24]) const
{
  throw InvalidValue("**ERROR**:SmallStrainPlastic: No geometric stiffness matrix available", 
                     __FILE__, __LINE__);
}

void 
SmallStrainPlastic::carryForward(const PatchSubset* patches,
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
    constParticleVariable<double>  pPlasticStrain, pDamage, pPorosity, 
      pStrainRate, pPlasticStrainRate;
    constParticleVariable<int>     pLocalized;

    old_dw->get(pStrainRate,     pStrainRateLabel,     pset);
    old_dw->get(pPlasticStrain,  pPlasticStrainLabel,  pset);
    old_dw->get(pPlasticStrainRate,  pPlasticStrainRateLabel,  pset);
    old_dw->get(pDamage,         pDamageLabel,         pset);
    old_dw->get(pPorosity,       pPorosityLabel,       pset);
    old_dw->get(pLocalized,      pLocalizedLabel,      pset);

    ParticleVariable<double>       pPlasticStrain_new, pDamage_new, 
      pPorosity_new, pStrainRate_new, pPlasticStrainRate_new;
    ParticleVariable<int>          pLocalized_new;

    new_dw->allocateAndPut(pStrainRate_new,      
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pPlasticStrainRate_new,      
                           pPlasticStrainRateLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pPorosity_new,      
                           pPorosityLabel_preReloc,               pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,              pset);

    // Get the plastic strain
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutRigid(pset, new_dw);
    d_kinematic->getBackStress(pset, old_dw);
    d_kinematic->allocateAndPutRigid(pset, new_dw);

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
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
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

void 
SmallStrainPlastic::allocateCMDataAddRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patch,
                                          MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patch);

  // Add requires local to this model
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::NewDW, pStrainRateLabel_preReloc,     matlset, gnone);
  task->requires(Task::NewDW, pPlasticStrainLabel_preReloc,  matlset, gnone);
  task->requires(Task::NewDW, pPlasticStrainRateLabel_preReloc, matlset, gnone);
  task->requires(Task::NewDW, pDamageLabel_preReloc,         matlset, gnone);
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,      matlset, gnone);
  task->requires(Task::NewDW, pPorosityLabel_preReloc,       matlset, gnone);
  d_plastic->allocateCMDataAddRequires(task,matl,patch,lb);
  d_kinematic->allocateCMDataAddRequires(task,matl,patch,lb);
}

void 
SmallStrainPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
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

  ParticleVariable<double>  pPlasticStrain, pDamage,pPorosity, pStrainRate,
                            pPlasticStrainRate;
  ParticleVariable<int>     pLocalized;

  constParticleVariable<double>  o_PlasticStrain, o_Damage,o_Porosity, 
                                 o_StrainRate, o_PlasticStrainRate;
  constParticleVariable<int>     o_Localized;

  new_dw->allocateTemporary(pPlasticStrain,addset);
  new_dw->allocateTemporary(pPlasticStrainRate,addset);
  new_dw->allocateTemporary(pDamage,addset);
  new_dw->allocateTemporary(pStrainRate,addset);
  new_dw->allocateTemporary(pLocalized,addset);
  new_dw->allocateTemporary(pPorosity,addset);

  new_dw->get(o_StrainRate,pStrainRateLabel_preReloc,delset);
  new_dw->get(o_PlasticStrain,pPlasticStrainLabel_preReloc,delset);
  new_dw->get(o_PlasticStrainRate,pPlasticStrainRateLabel_preReloc,delset);
  new_dw->get(o_Damage,pDamageLabel_preReloc,delset);
  new_dw->get(o_Localized,pLocalizedLabel_preReloc,delset);
  new_dw->get(o_Porosity,pPorosityLabel_preReloc,delset);

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pStrainRate[*n] = o_StrainRate[*o];
    pPlasticStrain[*n] = o_PlasticStrain[*o];
    pPlasticStrainRate[*n] = o_PlasticStrainRate[*o];
    pDamage[*n] = o_Damage[*o];
    pLocalized[*n] = o_Localized[*o];
    pPorosity[*n] = o_Porosity[*o];
  }

  (*newState)[pStrainRateLabel]=pStrainRate.clone();
  (*newState)[pPlasticStrainLabel]=pPlasticStrain.clone();
  (*newState)[pPlasticStrainRateLabel]=pPlasticStrainRate.clone();
  (*newState)[pDamageLabel]=pDamage.clone();
  (*newState)[pLocalizedLabel]=pLocalized.clone();
  (*newState)[pPorosityLabel]=pPorosity.clone();
  
  // Initialize the data for the plasticity model
  d_plastic->allocateCMDataAdd(new_dw,addset, newState, delset, old_dw);
  d_kinematic->allocateCMDataAdd(new_dw,addset, newState, delset, old_dw);
}


void 
SmallStrainPlastic::addRequiresDamageParameter(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}

void 
SmallStrainPlastic::getDamageParameter(const Patch* patch,
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
         
// Calculate the void nucleation factor
inline double 
SmallStrainPlastic::voidNucleationFactor(double ep)
{
  double temp = (ep - d_porosity.en)/d_porosity.sn;
  double A = d_porosity.fn/(d_porosity.sn*sqrt(2.0*M_PI))*
    exp(-0.5*temp*temp);
  return A;
}

double SmallStrainPlastic::computeRhoMicroCM(double pressure,
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

void SmallStrainPlastic::computePressEOSCM(double rho_cur,double& pressure,
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

double SmallStrainPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

void
SmallStrainPlastic::scheduleCheckNeedAddMPMMaterial(Task* task,
                                                const MPMMaterial* ,
                                                const PatchSet* ) const
{
  task->computes(lb->NeedAddMPMMaterialLabel);
}

void SmallStrainPlastic::checkNeedAddMPMMaterial(const PatchSubset* patches,
                                             const MPMMaterial* matl,
                                             DataWarehouse* ,
                                             DataWarehouse* new_dw)
{
  if (cout_EP.active()) {
    cout_EP << getpid() << "checkNeedAddMPMMaterial: In : Matl = " << matl
            << " id = " << matl->getDWIndex() <<  " patch = "
            << (patches->get(0))->getID();
  }

  double need_add=0.;

  new_dw->put(sum_vartype(need_add),     lb->NeedAddMPMMaterialLabel);
}
