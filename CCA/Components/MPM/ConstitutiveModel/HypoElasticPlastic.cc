#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/YieldConditionFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/StabilityCheckFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMEquationOfStateFactory.h>
#include <math.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Util/NotFinished.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityState.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

HypoElasticPlastic::HypoElasticPlastic(ProblemSpecP& ps, 
                                       MPMLabel* Mlb, 
                                       int n8or27)
{
  lb = Mlb;
  d_erosionAlgorithm = "none";

  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  ps->get("useModifiedEOS",d_useModifiedEOS);
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
  
  d_8or27 = n8or27;
  switch(d_8or27) {
  case 8:
    NGN = 1; break;
  case 27:
    NGN = 2; break;
  default:
    NGN = 1; break;
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
  // Instead of using the deformation gradient and a direct polar
  // decomposition to get the left stretch and rotation, store 
  // these in the data warehouse and update them before updating the
  // stress (BB 11/14/02)
  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);

  from.push_back(pLeftStretchLabel);
  from.push_back(pRotationLabel);
  from.push_back(pStrainRateLabel);
  from.push_back(pPlasticStrainLabel);
  from.push_back(pDamageLabel);
  from.push_back(pPorosityLabel);
  from.push_back(pLocalizedLabel);
  from.push_back(pPlasticTempLabel);
  from.push_back(pPlasticTempIncLabel);

  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);

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
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  //cout << "Initialize CM Data in HypoElasticPlastic" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pDeformGrad, pStress;
  ParticleVariable<Matrix3> pLeftStretch, pRotation;
  ParticleVariable<double> pPlasticStrain, pDamage, pPorosity, pStrainRate;
  ParticleVariable<int> pLocalized;
  ParticleVariable<double> pPlasticTemperature, pPlasticTempInc;

  new_dw->allocateAndPut(pDeformGrad, lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress, lb->pStressLabel, pset);

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

    // To fix : For a material that is initially stressed we need to
    // modify the leftStretch and the stress tensors to comply with the
    // initial stress state
    pDeformGrad[*iter] = one;
    pStress[*iter] = zero;

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
  //const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel, Ghost::None);

  task->requires(Task::OldDW,pLeftStretchLabel, Ghost::None);
  task->requires(Task::OldDW,pRotationLabel, Ghost::None);
  task->requires(Task::OldDW,pStrainRateLabel, Ghost::None);
  task->requires(Task::OldDW,pPlasticStrainLabel, Ghost::None);
  task->requires(Task::OldDW,pDamageLabel, Ghost::None);
  task->requires(Task::OldDW,pLocalizedLabel, Ghost::None);
  task->requires(Task::OldDW,pPorosityLabel, Ghost::None);
  task->requires(Task::OldDW,pPlasticTempLabel, Ghost::None);
  task->requires(Task::OldDW,pPlasticTempIncLabel, Ghost::None);

  d_plastic->allocateCMDataAddRequires(task,matl,patch,lb);
}

void HypoElasticPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
					   ParticleSubset* addset,
					   map<const VarLabel*, 
                                           ParticleVariableBase*>* newState,
					   ParticleSubset* delset,
					   DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3  zero(0.);
  ParticleSubset::iterator n,o;

  ParticleVariable<Matrix3> pDeformGrad, pStress;
  ParticleVariable<Matrix3> pLeftStretch, pRotation;
  ParticleVariable<double> pPlasticStrain, pDamage,pPorosity, pStrainRate;
  ParticleVariable<int> pLocalized;
  ParticleVariable<double> pPlasticTemperature, pPlasticTempInc;

  constParticleVariable<Matrix3> o_DeformGrad, o_Stress;
  constParticleVariable<Matrix3> o_LeftStretch, o_Rotation;
  constParticleVariable<double> o_PlasticStrain, o_Damage,o_Porosity, 
                                o_StrainRate;
  constParticleVariable<int> o_Localized;
  constParticleVariable<double> o_PlasticTemperature, o_PlasticTempInc;

  new_dw->allocateTemporary(pDeformGrad,addset);
  new_dw->allocateTemporary(pStress,addset);

  new_dw->allocateTemporary(pLeftStretch,addset);
  new_dw->allocateTemporary(pRotation,addset);
  new_dw->allocateTemporary(pPlasticStrain,addset);
  new_dw->allocateTemporary(pDamage,addset);
  new_dw->allocateTemporary(pStrainRate,addset);
  new_dw->allocateTemporary(pLocalized,addset);
  new_dw->allocateTemporary(pPorosity,addset);
  new_dw->allocateTemporary(pPlasticTemperature,addset);
  new_dw->allocateTemporary(pPlasticTempInc,addset);

  old_dw->get(o_DeformGrad,lb->pDeformationMeasureLabel,delset);
  old_dw->get(o_Stress,lb->pStressLabel,delset);

  old_dw->get(o_LeftStretch,pLeftStretchLabel,delset);
  old_dw->get(o_Rotation,pRotationLabel,delset);
  old_dw->get(o_StrainRate,pStrainRateLabel,delset);
  old_dw->get(o_PlasticStrain,pPlasticStrainLabel,delset);
  old_dw->get(o_Damage,pDamageLabel,delset);
  old_dw->get(o_Localized,pLocalizedLabel,delset);
  old_dw->get(o_Porosity,pPorosityLabel,delset);
  old_dw->get(o_PlasticTemperature,pPlasticTempLabel,delset);
  old_dw->get(o_PlasticTempInc,pPlasticTempIncLabel,delset);

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pDeformGrad[*n] = o_DeformGrad[*o];
    pStress[*n] = zero;

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

  (*newState)[lb->pDeformationMeasureLabel]=pDeformGrad.clone();
  (*newState)[lb->pStressLabel]=pStress.clone();

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
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void 
HypoElasticPlastic::computeStressTensor(const PatchSubset* patches,
					const MPMMaterial* matl,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw)
{
  // General stuff
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  Matrix3 tensorL; // Velocity gradient
  Matrix3 tensorD; // Rate of deformation
  Matrix3 tensorW; // Spin 
  Matrix3 tensorF; // Deformation gradient
  Matrix3 tensorV; // Left Cauchy-Green stretch
  Matrix3 tensorR; // Rotation 
  Matrix3 tensorSig; // The Cauchy stress
  Matrix3 tensorEta; // Deviatoric part of tensor D
  Matrix3 tensorS; // Devaitoric part of tensor Sig
  Matrix3 tensorF_new; // Deformation gradient

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();
  double sqrtTwo = sqrt(2.0);
  //double sqrtThree = sqrt(3.0);
  double totalStrainEnergy = 0.0;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

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
    if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);
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
    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    constNCVariable<Vector> GVelocity;
    new_dw->get(GVelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
#endif

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

    // Get the plastic strain
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // If the particle has localized, do nothing
      if (pLocalized[idx]) {
        pDeformGrad_new[idx] = pDeformGrad[idx];;
        pStress_new[idx] = pStress[idx];
        pVolume_deformed[idx] = pVolume[idx];

        pLeftStretch_new[idx] = pLeftStretch[idx]; 
        pRotation_new[idx] = pRotation[idx]; 
        pStrainRate_new[idx] = pStrainRate[idx];
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pDamage_new[idx] = pDamage[idx];
        pPorosity_new[idx] = pPorosity[idx];
        pLocalized_new[idx] = pLocalized[idx];
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx];
        pPlasticTempInc_new[idx] = 0.0;
        d_plastic->updateElastic(idx);
        continue;
      }

      // Calculate the velocity gradient (L) from the grid velocity
#ifdef FRACTURE
      short pgFld[27];
      for(int k=0; k<27; k++) 
	pgFld[k]=pgCode[idx][k];
      if (d_8or27==27) 
        tensorL = computeVelocityGradient(patch, oodx, px[idx], psize[idx],
                                          pgFld, gVelocity, GVelocity);
      else 
        tensorL = computeVelocityGradient(patch, oodx, px[idx], 
                                          pgFld, gVelocity, GVelocity);
#else
      if (d_8or27==27)
        tensorL = computeVelocityGradient(patch, oodx, px[idx], psize[idx],
					  gVelocity);
      else
        tensorL = computeVelocityGradient(patch, oodx, px[idx], gVelocity);
#endif

      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorW = (tensorL - tensorL.Transpose())*0.5;
      for (int ii = 1; ii < 4; ++ii) {
	for (int jj = 1; jj < 4; ++jj) {
	  tensorD(ii,jj)=(fabs(tensorD(ii,jj)) < d_tol) ? 0.0 : tensorD(ii,jj);
	  tensorW(ii,jj)=(fabs(tensorW(ii,jj)) < d_tol) ? 0.0 : tensorW(ii,jj);
	}
      }

      // Calculate the incremental update of the left stretch (V) 
      // and the rotation (R)
      tensorV = pLeftStretch[idx];
      tensorR = pRotation[idx];
      computeUpdatedVR(delT, tensorD, tensorW, tensorV, tensorR);
      //tensorF_new = tensorV*tensorR;
      //double J = tensorF_new.Determinant();
      //pDeformGrad_new[idx] = tensorF_new;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      Matrix3 tensorFinc = tensorL*delT + one;
      pDeformGrad_new[idx] = tensorFinc*pDeformGrad[idx];

      // Update the kinematic variables
      pLeftStretch_new[idx] = tensorV;
      pRotation_new[idx] = tensorR;

      // Calculate the current density and deformed volume
      double J = pDeformGrad_new[idx].Determinant();
      double rho_cur = rho_0/J;
      pVolume_deformed[idx]=pMass[idx]/rho_cur;

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
      double temperature = d_adiabaticHeating*pTemperature[idx] + 
                           (1.0-d_adiabaticHeating)*pPlasticTemperature[idx];

      // Calculate the plastic strain rate and plastic strain
      double epdot = sqrt(tensorEta.NormSquared()/1.5);
      epdot = max(epdot, d_tol);
      double ep = pPlasticStrain[idx] + epdot*delT;

      // Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->plasticStrainRate = epdot;
      state->plasticStrain = ep;
      state->pressure = pressure;
      state->temperature = temperature;
      state->density = rho_cur;
      state->initialDensity = rho_0;
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

      // Calculate the updated hydrostatic stress
      Matrix3 tensorHy = d_eos->computePressure(matl, bulk, mu_cur, 
						tensorF_new, tensorD, 
						tensorP, temperature,
						rho_cur, delT);

      // Integrate the stress rate equation to get a trial deviatoric stress
      Matrix3 trialS = tensorS + tensorEta*(2.0*mu_cur*delT);
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
      //cout << "Phi = " << Phi << endl;
      //cout << "Equivalent stress = " << equivStress 
      //     << " Flow stress = " << flowStress << endl;
     
      double Tdot = 0.0;
      if (Phi <= 0.0) {

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
          Matrix3 u = q/sqrtqs;

          // Calculate c and d at the beginning of time step
          double cplus = u.NormSquared();
          double dplus = u.Contract(tensorEta);
         
          // Calculate gamma_dot at the beginning of the time step
          ASSERT(cplus != 0);
          double gammadotplus = dplus/cplus;

          // Set initial theta
          double theta = 0.0;

          // Calculate u_q and u_eta
          double etaeta = sqrt(tensorEta.NormSquared());
          ASSERT(etaeta != 0);
	  Matrix3 u_eta = tensorEta/etaeta;
          double qq = sqrt(q.NormSquared());
          ASSERT(qq != 0);
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
	  double denom = 1.0 + (3.0*sqrtTwo*mu_cur*delGamma)/sig; 
	  ASSERT(denom != 0);
	  Stilde = trialS/denom;
	}
        
	// Do radial return adjustment
        double stst = sqrt(1.5*Stilde.NormSquared());
        ASSERT(stst != 0);
	tensorS = Stilde*(sig/stst);
	equivStress = sqrt((tensorS.NormSquared())*1.5);

        // Calculate total stress
	Matrix3 tensorSig = tensorS + tensorHy;

        // Update the plastic strain
        pPlasticStrain_new[idx] = ep;

        // Update the porosity
        pPorosity_new[idx] = updatePorosity(tensorD, delT, porosity, ep);

	// Calculate the updated scalar damage parameter
	pDamage_new[idx] = d_damage->computeScalarDamage(epdot, tensorS, 
							 temperature,
							 delT, matl, d_tol, 
							 pDamage[idx]);

        // Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = 0.9;
        double C_p = matl->getSpecificHeat();

        // ** WARNING ** Special for steel (remove for other materials)
        double T = temperature;
        C_p = 1.0e3*(0.09278 + 7.454e-4*T + 12404.0/(T*T));

        // Calculate Tdot (do not allow negative Tdot)
        Tdot = tensorSig.Contract(tensorD)*(taylorQuinney/(rho_cur*C_p));
        Tdot = max(Tdot, 0.0);

        // Update the plastic temperature
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx] + Tdot*delT; 
        pPlasticTempInc_new[idx] = Tdot*delT; 

        // Find if the particle has localized
	pLocalized_new[idx] = pLocalized[idx];
        bool isLocalized = false;

        // Check 1: Look at the temperature
        // Check 2: Look at Jacobian
        // Check 3: Modified Tepla rule
        // Check 4: Stability criterion
	double temp_new = temperature + Tdot*delT;
        double tepla = pPorosity_new[idx]/d_porosity.fc + pDamage_new[idx];
        if (temp_new > Tm_cur) isLocalized = true;
        else if (J < 0.0) isLocalized = true;
        else if (tepla > 1.0) isLocalized = true;
	else if (d_stable) {

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
	}
	if (isLocalized) {
	  cerr << " Particle " << idx << " is localized " << endl;

	  // set the particle localization flag to true and set the 
	  // stress to zero
	  pLocalized_new[idx] = 1;
	  pStress_new[idx] = zero;

	  // Update internal variables
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
      double pStrainEnergy = (tensorD(1,1)*avgStress(1,1) +
			      tensorD(2,2)*avgStress(2,2) +
			      tensorD(3,3)*avgStress(3,3) +
			      2.0*(tensorD(1,2)*avgStress(1,2) + 
				   tensorD(1,3)*avgStress(1,3) +
				   tensorD(2,3)*avgStress(2,3)))*
	pVolume_deformed[idx]*delT;
      totalStrainEnergy += pStrainEnergy;		   

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.0*mu_cur/3.0)*
			  pVolume_deformed[idx]/pMass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      delete state;
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
  }
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

    constParticleVariable<Matrix3> pDeformGrad, pStress;
    constParticleVariable<double>  pMass; 

    old_dw->get(pDeformGrad, lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress,     lb->pStressLabel,             pset);
    old_dw->get(pMass,       lb->pMassLabel,               pset);

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

    ParticleVariable<Matrix3>      pDeformGrad_new, pStress_new;
    ParticleVariable<double>       pVolume_new;

    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pVolume_new, lb->pVolumeDeformedLabel, pset);

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

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pStress_new[idx] = pStress[idx];
      pDeformGrad_new[idx] = pDeformGrad[idx];
      pVolume_new[idx]=(pMass[idx]/rho_orig);

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

    new_dw->put(delt_vartype(1.e10), lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void HypoElasticPlastic::addRequiresDamageParameter(Task* task,
						    const MPMMaterial* matl,
						    const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pLocalizedLabel,matlset,Ghost::None);

}

void HypoElasticPlastic::getDamageParameter(const Patch* patch,
					    ParticleVariable<int>& damage,
					    int dwi,
					    DataWarehouse* dw)
{
  ParticleSubset* pset = dw->getParticleSubset(dwi,patch);
  constParticleVariable<int> pLocalized;
  dw->get(pLocalized, pLocalizedLabel, pset);

  ParticleSubset::iterator iter;
  for (iter = pset->begin(); iter != pset->end(); iter++) {
    damage[*iter] = pLocalized[*iter];
  }
   
}
	 
void 
HypoElasticPlastic::computeStressTensorWithErosion(const PatchSubset* patches,
						   const MPMMaterial* matl,
						   DataWarehouse* old_dw,
						   DataWarehouse* new_dw)
{
  // General stuff
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  Matrix3 tensorL; // Velocity gradient
  Matrix3 tensorD; // Rate of deformation
  Matrix3 tensorW; // Spin 
  Matrix3 tensorF; // Deformation gradient
  Matrix3 tensorV; // Left Cauchy-Green stretch
  Matrix3 tensorR; // Rotation 
  Matrix3 tensorSig; // The Cauchy stress
  Matrix3 tensorEta; // Deviatoric part of tensor D
  Matrix3 tensorS; // Devaitoric part of tensor Sig
  Matrix3 tensorF_new; // Deformation gradient

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double rho_0 = matl->getInitialDensity();
  double Tm = matl->getMeltTemperature();
  double sqrtTwo = sqrt(2.0);
  //double sqrtThree = sqrt(3.0);
  double totalStrainEnergy = 0.0;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

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
    if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);
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
    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    constNCVariable<Vector> GVelocity;
    new_dw->get(GVelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
#endif

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

    // Get the plastic strain
    d_plastic->getInternalVars(pset, old_dw);
    d_plastic->allocateAndPutInternalVars(pset, new_dw);

    // Get the erosion data
    constParticleVariable<double> pErosion;
    ParticleVariable<double> pErosion_new;
    old_dw->get(pErosion, lb->pErosionLabel, pset);
    new_dw->allocateAndPut(pErosion_new, lb->pErosionLabel_preReloc, pset);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Calculate the velocity gradient (L) from the grid velocity
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      if (d_8or27==27) 
	patch->findCellAndShapeDerivatives27(px[idx], ni, d_S, psize[idx]);
      else
	patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
      tensorL.set(0.0);
      for(int k = 0; k < d_8or27; k++) {
	const Vector& gvel = gVelocity[ni[k]];
	d_S[k] *= pErosion[idx];
	for (int j = 0; j<3; j++){
          double d_SXoodx = d_S[k][j] * oodx[j];
	  for (int i = 0; i<3; i++) {
	    tensorL(i+1,j+1) += gvel[i] * d_SXoodx;
	  }
	}
      }

      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorW = (tensorL - tensorL.Transpose())*0.5;
      for (int ii = 1; ii < 4; ++ii) {
	for (int jj = 1; jj < 4; ++jj) {
	  tensorD(ii,jj)=(fabs(tensorD(ii,jj)) < d_tol) ? 0.0 : tensorD(ii,jj);
	  tensorW(ii,jj)=(fabs(tensorW(ii,jj)) < d_tol) ? 0.0 : tensorW(ii,jj);
	}
      }

      // Calculate the incremental update of the left stretch (V) 
      // and the rotation (R)
      tensorV = pLeftStretch[idx];
      tensorR = pRotation[idx];
      computeUpdatedVR(delT, tensorD, tensorW, tensorV, tensorR);
      //tensorF_new = tensorV*tensorR;
      //double J = tensorF_new.Determinant();
      //pDeformGrad_new[idx] = tensorF_new;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      Matrix3 tensorFinc = tensorL*delT + one;
      pDeformGrad_new[idx] = tensorFinc*pDeformGrad[idx];

      // Update the kinematic variables
      pLeftStretch_new[idx] = tensorV;
      pRotation_new[idx] = tensorR;

      // Calculate the current density and deformed volume
      double J = pDeformGrad_new[idx].Determinant();
      double rho_cur = rho_0/J;
      pVolume_deformed[idx]=pMass[idx]/rho_cur;

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
      double temperature = d_adiabaticHeating*pTemperature[idx] + 
                           (1.0-d_adiabaticHeating)*pPlasticTemperature[idx];

      // Calculate the plastic strain rate and plastic strain
      double epdot = sqrt(tensorEta.NormSquared()/1.5);
      epdot = max(epdot, d_tol);
      double ep = pPlasticStrain[idx] + epdot*delT;

      // Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->plasticStrainRate = epdot;
      state->plasticStrain = ep;
      state->pressure = pressure;
      state->temperature = temperature;
      state->density = rho_cur;
      state->initialDensity = rho_0;
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

      // Calculate the updated hydrostatic stress
      Matrix3 tensorHy = d_eos->computePressure(matl, bulk, mu_cur, 
						tensorF_new, tensorD, 
						tensorP, temperature,
						rho_cur, delT);

      // Integrate the stress rate equation to get a trial deviatoric stress
      Matrix3 trialS = tensorS + tensorEta*(2.0*mu_cur*delT);
      double equivStress = sqrt((trialS.NormSquared())*1.5);

      // Keep the mass of the particle
      pErosion_new[idx] = pErosion[idx];

      // If the particle has localized
      if (pLocalized[idx]) {
	if (d_erosionAlgorithm == "RemoveMass") {

	  // Further Erode the particle
	  pErosion_new[idx] = 0.1*pErosion[idx];

          // Calculate the elastic stress and rotate
	  tensorSig = trialS + tensorHy;
	  tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

	  // Save the updated data
	  pStress_new[idx] = tensorSig;
	  pPlasticStrain_new[idx] = pPlasticStrain[idx];
	  pDamage_new[idx] = pDamage[idx];
	  pPorosity_new[idx] = pPorosity[idx];
          pLocalized_new[idx] = pLocalized[idx];
          pPlasticTemperature_new[idx] = pPlasticTemperature[idx];
          pPlasticTempInc_new[idx] = 0.0;
          d_plastic->updateElastic(idx);
          continue;
        
	} else if (d_erosionAlgorithm == "AllowNoTension") {

          // Do not allow any tensile or shear stresses
          for (int ii = 1; ii < 4; ++ii) {
	    if (tensorHy(ii,ii) > 0.0) tensorHy(ii,ii) = 0.0;
          }
	  tensorHy = (tensorR*tensorHy)*tensorR.Transpose();

	  pStress_new[idx] = tensorHy;
	  pPlasticStrain_new[idx] = pPlasticStrain[idx];
	  pDamage_new[idx] = pDamage[idx];
	  pPorosity_new[idx] = pPorosity[idx];
          pLocalized_new[idx] = pLocalized[idx];
          pPlasticTemperature_new[idx] = pPlasticTemperature[idx];
          pPlasticTempInc_new[idx] = 0.0;
	  d_plastic->updateElastic(idx);
          continue;

        } else {

	  pDeformGrad_new[idx] = pDeformGrad[idx];;
	  pStress_new[idx] = pStress[idx];
	  pVolume_deformed[idx] = pVolume[idx];

	  pLeftStretch_new[idx] = pLeftStretch[idx]; 
	  pRotation_new[idx] = pRotation[idx]; 
	  pStrainRate_new[idx] = pStrainRate[idx];
	  pPlasticStrain_new[idx] = pPlasticStrain[idx];
	  pDamage_new[idx] = pDamage[idx];
	  pPorosity_new[idx] = pPorosity[idx];
	  pLocalized_new[idx] = pLocalized[idx];
	  pPlasticTemperature_new[idx] = pPlasticTemperature[idx];
	  pPlasticTempInc_new[idx] = 0.0;
	  d_plastic->updateElastic(idx);
	  continue;
        }
      }

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
      //cout << "Phi = " << Phi << endl;
      //cout << "Equivalent stress = " << equivStress 
      //     << " Flow stress = " << flowStress << endl;
     
      double Tdot = 0.0;
      if (Phi <= 0.0) {

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
          Matrix3 u = q/sqrtqs;

          // Calculate c and d at the beginning of time step
          double cplus = u.NormSquared();
          double dplus = u.Contract(tensorEta);
         
          // Calculate gamma_dot at the beginning of the time step
          ASSERT(cplus != 0);
          double gammadotplus = dplus/cplus;

          // Set initial theta
          double theta = 0.0;

          // Calculate u_q and u_eta
          double etaeta = sqrt(tensorEta.NormSquared());
          ASSERT(etaeta != 0);
	  Matrix3 u_eta = tensorEta/etaeta;
          double qq = sqrt(q.NormSquared());
          ASSERT(qq != 0);
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
	  double denom = 1.0 + (3.0*sqrtTwo*mu_cur*delGamma)/sig; 
	  ASSERT(denom != 0);
	  Stilde = trialS/denom;
	}
        
	// Do radial return adjustment
        double stst = sqrt(1.5*Stilde.NormSquared());
        ASSERT(stst != 0);
	tensorS = Stilde*(sig/stst);
	equivStress = sqrt((tensorS.NormSquared())*1.5);

        // Calculate total stress
	Matrix3 tensorSig = tensorS + tensorHy;

        // Update the plastic strain
        pPlasticStrain_new[idx] = ep;

        // Update the porosity
        pPorosity_new[idx] = updatePorosity(tensorD, delT, porosity, ep);

	// Calculate the updated scalar damage parameter
	pDamage_new[idx] = d_damage->computeScalarDamage(epdot, tensorS, 
							 temperature,
							 delT, matl, d_tol, 
							 pDamage[idx]);

        // Calculate rate of temperature increase due to plastic strain
        double taylorQuinney = 0.9;
        double C_p = matl->getSpecificHeat();

        // ** WARNING ** Special for steel (remove for other materials)
        double T = temperature;
        C_p = 1.0e3*(0.09278 + 7.454e-4*T + 12404.0/(T*T));

        // Calculate Tdot (do not allow negative Tdot)
        Tdot = tensorSig.Contract(tensorD)*(taylorQuinney/(rho_cur*C_p));
        Tdot = max(Tdot, 0.0);

        // Update the plastic temperature
        pPlasticTemperature_new[idx] = pPlasticTemperature[idx] + Tdot*delT; 
        pPlasticTempInc_new[idx] = Tdot*delT; 

        // Find if the particle has localized
	pLocalized_new[idx] = pLocalized[idx];
        bool isLocalized = false;

        // Check 1: Look at the temperature
        // Check 2: Look at Jacobian
        // Check 3: Modified Tepla rule
        // Check 4: Stability criterion
	double temp_new = temperature + Tdot*delT;
        double tepla = pPorosity_new[idx]/d_porosity.fc + pDamage_new[idx];
        if (temp_new > Tm_cur) isLocalized = true;
        else if (J < 0.0) isLocalized = true;
        else if (tepla > 1.0) isLocalized = true;
	else if (d_stable) {

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
	}
	if (isLocalized) {
	  cerr << " Particle " << idx << " is localized " << endl;

	  // set the particle localization flag to true 
	  pLocalized_new[idx] = 1;

	  if (d_erosionAlgorithm == "RemoveMass") {
	    // Erode the particle
	    pErosion_new[idx] = 0.1*pErosion[idx];
	  } else if (d_erosionAlgorithm == "AllowNoTension") {
	    // Assuming almost all strain energy is plastic strain
	    // energy - no conversion into kinetic energy needed
	    // and all kinematic quantities remain the same
	    // Only stress goes to zero
	    pStress_new[idx] = zero;
	  } else {
	    // Reset everything and return
	    pLeftStretch_new[idx] = one;
	    pRotation_new[idx] = pRotation[idx];
	    pDeformGrad_new[idx] = one;
	    pVolume_deformed[idx]=pMass[idx]/rho_0;
	    pStress_new[idx] = zero;
	  }

	  // Update internal variables
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
      double pStrainEnergy = (tensorD(1,1)*avgStress(1,1) +
			      tensorD(2,2)*avgStress(2,2) +
			      tensorD(3,3)*avgStress(3,3) +
			      2.0*(tensorD(1,2)*avgStress(1,2) + 
				   tensorD(1,3)*avgStress(1,3) +
				   tensorD(2,3)*avgStress(2,3)))*
	pVolume_deformed[idx]*delT;
      totalStrainEnergy += pStrainEnergy;		   

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.0*mu_cur/3.0)*
			  pVolume_deformed[idx]/pMass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      delete state;
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
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
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  if(d_8or27==27)
    task->requires(Task::OldDW, lb->pSizeLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel,       matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel,          matlset,Ghost::None);
  task->requires(Task::NewDW, lb->gVelocityLabel,          matlset,gac, NGN);

  task->requires(Task::OldDW, lb->pStressLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);

#ifdef FRACTURE
  task->requires(Task::NewDW,  lb->pgCodeLabel,    matlset, Ghost::None); 
  task->requires(Task::NewDW,  lb->GVelocityLabel, matlset, gac, NGN);
#endif

  task->requires(Task::OldDW, pLeftStretchLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pRotationLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pStrainRateLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pPlasticStrainLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pDamageLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pPorosityLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pLocalizedLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pPlasticTempLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pPlasticTempIncLabel, matlset,Ghost::None);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

  task->computes(pLeftStretchLabel_preReloc,  matlset);
  task->computes(pRotationLabel_preReloc, matlset);
  task->computes(pStrainRateLabel_preReloc, matlset);
  task->computes(pPlasticStrainLabel_preReloc, matlset);
  task->computes(pDamageLabel_preReloc, matlset);
  task->computes(pPorosityLabel_preReloc, matlset);
  task->computes(pLocalizedLabel_preReloc, matlset);
  task->computes(pPlasticTempLabel_preReloc, matlset);
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

// Actually calculate erosion
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

  for (int ii = 1; ii < 4; ++ii) {
    for (int jj = 1; jj < 4; ++jj) {
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
  double w[4];
  w[0] = 0.0;
  w[1] = -0.5*(tensorW(2,3)-tensorW(3,2));
  w[2] = -0.5*(tensorW(3,1)-tensorW(1,3));
  w[3] = -0.5*(tensorW(1,2)-tensorW(2,1));

  // Calculate tensor Z
  Matrix3 tensorZ = (tensorD*tensorV) - (tensorV*tensorD);

  // Calculate vector z
  double z[4];
  z[0] = 0.0;
  z[1] = -0.5*(tensorZ(2,3)-tensorZ(3,2));
  z[2] = -0.5*(tensorZ(3,1)-tensorZ(1,3));
  z[3] = -0.5*(tensorZ(1,2)-tensorZ(2,1));

  // Calculate I Trace(V) - V
  Matrix3 one;   one.Identity();
  Matrix3 temp = one*(tensorV.Trace()) - tensorV;
  ASSERT(temp.Determinant() != 0.0);
  temp = temp.Inverse();

  // Calculate vector omega = w + temp*z
  double omega[4];
  omega[0] = 0.0;
  for (int ii = 1; ii < 4; ++ii) {
    double sum = 0.0;
    for (int jj = 1; jj < 4; ++jj) {
      sum += temp(ii,jj)*z[jj]; 
    }
    omega[ii] = w[ii] + sum;
  }

  // Calculate tensor Omega
  Matrix3 tensorOmega;
  tensorOmega(1,2) = -omega[3];  
  tensorOmega(1,3) = omega[2];  
  tensorOmega(2,1) = omega[3];  
  tensorOmega(2,3) = -omega[1];  
  tensorOmega(3,1) = -omega[2];  
  tensorOmega(3,2) = omega[1];  

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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

