#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "PlasticityModels/YieldConditionFactory.h"
#include "PlasticityModels/StabilityCheckFactory.h"
#include "PlasticityModels/PlasticityModelFactory.h"
#include "PlasticityModels/DamageModelFactory.h"
#include "PlasticityModels/MPMEquationOfStateFactory.h"
#include "PlasticityModels/PlasticityState.h"
#include <math.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

static DebugStream cout_CST("HEP",false);
static DebugStream cout_CST1("HEP1",false);


HypoElasticPlastic::HypoElasticPlastic(ProblemSpecP& ps, MPMLabel* Mlb, 
                                       MPMFlags* Mflag)
  : ConstitutiveModel(Mlb,Mflag)
{

  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

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
    throw ParameterNotFound(desc.str());
  }

  d_stable = StabilityCheckFactory::create(ps);
  if(!d_stable) cerr << "Stability check disabled\n";

  d_plastic = PlasticityModelFactory::create(ps);
  if(!d_plastic){
    ostringstream desc;
    desc << "An error occured in the PlasticityModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str());
  }

  d_damage = DamageModelFactory::create(ps);
  if(!d_damage){
    ostringstream desc;
    desc << "An error occured in the DamageModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str());
  }
  
  d_eos = MPMEquationOfStateFactory::create(ps);
  if(!d_eos){
    ostringstream desc;
    desc << "An error occured in the EquationOfStateFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str());
  }
  

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

HypoElasticPlastic::HypoElasticPlastic(const HypoElasticPlastic* cm)
{
  lb = cm->lb;
  flag = cm->flag;
  NGN = cm->NGN;
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
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

void HypoElasticPlastic::initializeCMData(const Patch* patch,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

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

    SCIRun::Gaussian gaussGen(d_porosity.f0, d_porosity.f0_std, 0);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){

      // Generate a Gaussian distributed random number given the mean
      // porosity and the std.
      pPorosity[*iter] = fabs(gaussGen.rand());
    }
  }

  if (d_scalarDam.scalarDamageDist != "constant") {

    SCIRun::Gaussian gaussGen(d_scalarDam.D0, d_scalarDam.D0_std, 0);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){

      // Generate a Gaussian distributed random number given the mean
      // damage and the std.
      pDamage[*iter] = fabs(gaussGen.rand());
    }
  }

  // Initialize the data for the plasticity model
  d_plastic->initializeInternalVars(pset, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void HypoElasticPlastic::allocateCMDataAddRequires(Task* task,
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

void HypoElasticPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
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


void HypoElasticPlastic::computeStableTimestep(const Patch* patch,
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
  new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
              lb->delTLabel);
}

void 
HypoElasticPlastic::computeStressTensor(const PatchSubset* patches,
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
  Matrix3 tensorL(0.0); // Velocity gradient
  Matrix3 tensorD(0.0); // Rate of deformation
  Matrix3 tensorW(0.0); // Spin 
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
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();
  double sqrtTwo = sqrt(2.0);
  double totalStrainEnergy = 0.0;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

    //cerr << getpid() << " patch = " << patch->getID() << endl;
    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    //double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

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
    constParticleVariable<double> pMass, pVolume;
    old_dw->get(px, lb->pXLabel, pset);
    old_dw->get(psize, lb->pSizeLabel, pset);
    old_dw->get(pMass, lb->pMassLabel, pset);
    old_dw->get(pVolume, lb->pVolumeLabel, pset);

    // Get the velocity from the grid and particle velocity
    constParticleVariable<Vector> pVelocity;
    constNCVariable<Vector> gVelocity;
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(gVelocity, lb->gVelocityLabel, dwi, patch, gac, NGN);

    // Get the particle stress and temperature
    constParticleVariable<Matrix3> pStress;
    constParticleVariable<double> pTemperature;
    old_dw->get(pStress, lb->pStressLabel, pset);
    old_dw->get(pTemperature, lb->pTemperatureLabel, pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    constParticleVariable<Short27> pgCode;
    constNCVariable<Vector> GVelocity;
    if (flag->d_fracture) {
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      new_dw->get(GVelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
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
                           lb->pVolumeDeformedLabel,              pset);

    // LOCAL
    ParticleVariable<Matrix3> pLeftStretch_new, pRotation_new;
    ParticleVariable<double>  pPlasticStrain_new, pDamage_new, pPorosity_new, 
      pStrainRate_new;
    ParticleVariable<double>  pPlasticTemperature_new, pPlasticTempInc_new;
    ParticleVariable<int>     pLocalized_new;
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

    // Allocate variable to store internal heating rate
    ParticleVariable<double> pIntHeatRate;
    new_dw->allocateAndPut(pIntHeatRate, lb->pInternalHeatRateLabel_preReloc, 
                           pset);

    // Get the plastic strain
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pIntHeatRate[idx] = 0.0;

      //cerr << getpid() << " idx = " << idx << endl;
      // Calculate the velocity gradient (L) from the grid velocity

      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

      Matrix3 tensorL(0.0);
      short pgFld[27];
      if (flag->d_fracture) {
        for(int k=0; k<27; k++) 
          pgFld[k]=pgCode[idx][k];

        computeVelocityGradient(tensorL,ni,d_S,oodx,pgFld,gVelocity,GVelocity);
      } else {
        computeVelocityGradient(tensorL,ni,d_S,oodx,gVelocity);
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
        throw ParameterNotFound("**ERROR**:HypoElasticPlastic");
      }

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J;
      pVolume_deformed[idx]=pMass[idx]/rho_cur;

      // Compute polar decomposition of F (F = VR)
      tensorF_new.polarDecomposition(tensorV, tensorR, d_tol, false);

      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorW = (tensorL - tensorL.Transpose())*0.5;
      for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
          tensorD(ii,jj)=(fabs(tensorD(ii,jj)) < d_tol) ? 0.0 : tensorD(ii,jj);
          tensorW(ii,jj)=(fabs(tensorW(ii,jj)) < d_tol) ? 0.0 : tensorW(ii,jj);
        }
      }

      /* COMMENT : Old incremental update of V and R

      // Calculate the incremental update of the left stretch (V) 
      // and the rotation (R)
      tensorV = pLeftStretch[idx];
      tensorR = pRotation[idx];
      if (idx == 954) {
      cerr << getpid() << "idx = " << idx <<  " V_old = " << tensorV << endl;
      //cerr << getpid() << " R_old = " << tensorR << endl;
      //cerr << getpid() << " D_in = " << tensorD << endl;
      //cerr << getpid() << " W_in = " << tensorW << endl;
      }
      computeUpdatedVR(delT, tensorD, tensorW, tensorV, tensorR);
      if (idx == 954) {
      cerr << getpid() <<  "idx = " << idx << " V_new = " << tensorV << endl;
      //cerr << getpid() << " R_new = " << tensorR << endl;
      }
      //tensorF_new = tensorV*tensorR;
      //double J = tensorF_new.Determinant();
      //pDeformGrad_new[idx] = tensorF_new;

      */


      // Update the kinematic variables
      pLeftStretch_new[idx] = tensorV;
      pRotation_new[idx] = tensorR;

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
        d_plastic->updateElastic(idx);
        continue;
      }

      // Rotate the total rate of deformation tensor,
      // the plastic rate of deformation tensor, and the Cauchy stress
      // back to the material configuration and calculate their
      // deviatoric parts
      tensorD = (tensorR.Transpose())*(tensorD*tensorR);
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);
      pStrainRate_new[idx] = sqrt(tensorD.NormSquared()/1.5);

      tensorSig = pStress[idx];
      tensorSig = (tensorR.Transpose())*(tensorSig*tensorR);
      double pressure = tensorSig.Trace()/3.0;
      Matrix3 tensorP = one*pressure;
      tensorS = tensorSig - tensorP;

      // Calculate the temperature at the start of the time step
      double temperature = flag->d_adiabaticHeating*pTemperature[idx] + 
        (1.0-flag->d_adiabaticHeating)*pPlasticTemperature[idx];

      // Calculate the plastic strain rate and plastic strain
      double epdot = sqrt(tensorEta.NormSquared()/1.5);
      double ep = pPlasticStrain[idx] + epdot*delT;

      // Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->plasticStrainRate = epdot;
      state->plasticStrain = ep;
      state->pressure = pressure;
      state->temperature = temperature;
      state->density = rho_cur;
      state->initialDensity = rho_0;
      state->bulkModulus = bulk ;
      state->initialBulkModulus = bulk;
      state->shearModulus = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp = Tm ;
      state->initialMeltTemp = Tm;
    
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
      /*
      double qVisco = 0.0;
      if (flag->d_artificial_viscosity) {
        double Dkk = tensorD.Trace();
        double c_bulk = sqrt(bulk/rho_cur);
        qVisco = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
      }
      */

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
          throw ParameterNotFound("**ERROR**");
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
            throw ParameterNotFound("**ERROR**");
          }
          Matrix3 u = q/sqrtqs;

          // Calculate c and d at the beginning of time step
          double cplus = u.NormSquared();
          double dplus = u.Contract(tensorEta);
         
          // Calculate gamma_dot at the beginning of the time step
          ASSERT(cplus != 0);
          if (cplus == 0) {
            cout << getpid() << " HEP:cplus = " << cplus << " u = " << u << endl;
            throw ParameterNotFound("**ERROR**");
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
            throw ParameterNotFound("**ERROR**");
          }
          Matrix3 u_eta = tensorEta/etaeta;
          double qq = sqrt(q.NormSquared());
          ASSERT(qq != 0);
          if (qq == 0) {
            cout << getpid() << " HEP:qq = " << qq << " q = " << q << endl;
            throw ParameterNotFound("**ERROR**");
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
              cout << getpid() << " HEP:dStar = " << dStar << " theta = " << theta
                   << " u_eta = " << u_eta 
                   << " Eta = " << tensorEta << " u_q = " << u_q
                   << " cplus = " << cplus << endl;
              throw ParameterNotFound("**ERROR**");
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
            throw ParameterNotFound("**ERROR**");
          }
          double denom = 1.0 + (3.0*sqrtTwo*mu_cur*delGamma)/sig; 
          ASSERT(denom != 0);
          if (denom == 0) {
            cout << getpid() << " HEP:denom = " << denom << " mu_cur = " << mu_cur
                 << " delGamma = " << delGamma << " sig = " << sig << endl;
            throw ParameterNotFound("**ERROR**");
          }
          Stilde = trialS/denom;
        }
        
        // Do radial return adjustment
        double stst = sqrt(1.5*Stilde.NormSquared());
        ASSERT(stst != 0);
        if (stst == 0) {
          cout << getpid() << " HEP:stst = " << stst << " Stilde = " << Stilde << endl;
          throw ParameterNotFound("**ERROR**");
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
        double C_p = matl->getSpecificHeat();

        // ** WARNING ** Special for steel (remove for other materials)
        //double T = temperature;
        //C_p = 1.0e3*(0.09278 + 7.454e-4*T + 12404.0/(T*T));

        // Alternative approach
        double Tdot = flowStress*epdot*taylorQuinney/(rho_cur*C_p);
        pIntHeatRate[idx] = Tdot;
        double dT = Tdot*delT;
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
          if (d_stable && !isLocalized) {

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
            d_yield->computeElasPlasTangentModulus(Ce, tensorSig, sigY, dsigYdep,
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

      // Compute the strain energy for the particles
      Matrix3 avgStress = (pStress_new[idx] + pStress[idx])*0.5;
      double pStrainEnergy = (tensorD(0,0)*avgStress(0,0) +
                              tensorD(1,1)*avgStress(1,1) +
                              tensorD(2,2)*avgStress(2,2) +
                              2.0*(tensorD(0,1)*avgStress(0,1) + 
                                   tensorD(0,2)*avgStress(0,2) +
                                   tensorD(1,2)*avgStress(1,2)))*
        pVolume_deformed[idx]*delT;
      totalStrainEnergy += pStrainEnergy;                  

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      delete state;
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    delete interpolator;
  }
  // cout_CST << getpid() << "... Out" << endl;
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

void HypoElasticPlastic::carryForward(const PatchSubset* patches,
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

    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.e10)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void HypoElasticPlastic::addRequiresDamageParameter(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}

void HypoElasticPlastic::getDamageParameter(const Patch* patch,
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
HypoElasticPlastic::addComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* patch) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patch);

  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;

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
  d_plastic->addComputesAndRequires(task, matl, patch);

}

void 
HypoElasticPlastic::addComputesAndRequires(Task* ,
                                           const MPMMaterial* ,
                                           const PatchSet* ,
                                           const bool ) const
{
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
inline double 
HypoElasticPlastic::voidNucleationFactor(double ep)
{
  double temp = (ep - d_porosity.en)/d_porosity.sn;
  double A = d_porosity.fn/(d_porosity.sn*sqrt(2.0*M_PI))*
    exp(-0.5*temp*temp);
  return A;
}

double HypoElasticPlastic::computeRhoMicroCM(double pressure,
                                             const double p_ref,
                                             const MPMMaterial* matl)
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

void HypoElasticPlastic::computePressEOSCM(double rho_cur,double& pressure,
                                           double p_ref,  
                                           double& dp_drho, double& tmp,
                                           const MPMMaterial* matl)
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

double HypoElasticPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

void
HypoElasticPlastic::scheduleCheckNeedAddMPMMaterial(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* patch) const
{
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pPlasticStrainLabel_preReloc,   matlset, gnone);

  task->computes(lb->NeedAddMPMMaterialLabel);
}

void HypoElasticPlastic::checkNeedAddMPMMaterial(const PatchSubset* patches,
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


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

