
#include "HyperElasticPlastic.h"
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMEquationOfStateFactory.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

HyperElasticPlastic::HyperElasticPlastic(ProblemSpecP& ps, MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  d_useMPMICEModifiedEOS = false;
  ps->get("useModifiedEOS",d_useMPMICEModifiedEOS); 
  d_erosionAlgorithm = "none";
  
  d_plasticity = PlasticityModelFactory::create(ps);
  if(!d_plasticity){
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

  pBbarElasticLabel = VarLabel::create("p.bbarElastic",
	ParticleVariable<Matrix3>::getTypeDescription());
  pBbarElasticLabel_preReloc = VarLabel::create("p.bbarElastic+",
	ParticleVariable<Matrix3>::getTypeDescription());
  pDamageLabel = VarLabel::create("p.damage",
	ParticleVariable<double>::getTypeDescription());
  pDamageLabel_preReloc = VarLabel::create("p.damage+",
	ParticleVariable<double>::getTypeDescription());

}

HyperElasticPlastic::~HyperElasticPlastic()
{
  // Destructor 
  VarLabel::destroy(pBbarElasticLabel);
  VarLabel::destroy(pBbarElasticLabel_preReloc);
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pDamageLabel_preReloc);

  delete d_plasticity;
  delete d_damage;
  delete d_eos;
}

void 
HyperElasticPlastic::addParticleState(std::vector<const VarLabel*>& from,
				      std::vector<const VarLabel*>& to)
{
  from.push_back(pBbarElasticLabel);
  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);
  from.push_back(pDamageLabel);

  to.push_back(pBbarElasticLabel_preReloc);
  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);
  to.push_back(pDamageLabel_preReloc);

  // Erosion stuff
  if (d_erosionAlgorithm != "none") {
    from.push_back(lb->pErosionLabel);
    to.push_back(lb->pErosionLabel_preReloc);
  }

  // Add the particle state for the plasticity model
  d_plasticity->addParticleState(from, to);
}

void 
HyperElasticPlastic::initializeCMData(const Patch* patch,
				      const MPMMaterial* matl,
				      DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pDeformGrad, pStress;
  ParticleVariable<Matrix3> pBbarElastic;
  ParticleVariable<double> pDamage;

  new_dw->allocateAndPut(pDeformGrad, lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress, lb->pStressLabel, pset);
  new_dw->allocateAndPut(pDamage, pDamageLabel, pset);
  new_dw->allocateAndPut(pBbarElastic, pBbarElasticLabel, pset);

  for(ParticleSubset::iterator iter =pset->begin();iter != pset->end(); iter++){

    // To fix : For a material that is initially stressed we need to
    // modify the left Cauchy-Green and stress tensors to comply with the
    // initial stress state
    pBbarElastic[*iter] = one;
    pDeformGrad[*iter] = one;
    pStress[*iter] = zero;
    pDamage[*iter] = 0.0;
  }

  // Initialize the data for the plasticity model
  d_plasticity->initializeInternalVars(pset, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void HyperElasticPlastic::allocateCMDataAddRequires(Task* task,
						   const MPMMaterial* matl,
						   const PatchSet* patch,
						   MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel, Ghost::None);
  task->requires(Task::OldDW,pDamageLabel, Ghost::None);
  task->requires(Task::OldDW,pBbarElasticLabel, Ghost::None);
}


void 
HyperElasticPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
				       ParticleSubset* addset,
				       map<const VarLabel*, ParticleVariableBase*>* newState,
				       ParticleSubset* delset,
				       DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 zero(0.); 
  ParticleSubset::iterator n,o;

  ParticleVariable<Matrix3> pDeformGrad, pStress;
  ParticleVariable<Matrix3> pBbarElastic;
  ParticleVariable<double> pDamage;

  constParticleVariable<Matrix3> o_DeformGrad,o_Stress;
  constParticleVariable<Matrix3> o_BbarElastic;
  constParticleVariable<double> o_Damage;

  new_dw->allocateTemporary(pDeformGrad,addset);
  new_dw->allocateTemporary(pStress,addset);
  new_dw->allocateTemporary(pDamage,addset);
  new_dw->allocateTemporary(pBbarElastic,addset);

  old_dw->get(o_DeformGrad,lb->pDeformationMeasureLabel,delset);
  old_dw->get(o_Stress,lb->pStressLabel,delset);
  old_dw->get(o_BbarElastic,pBbarElasticLabel,delset);
  old_dw->get(o_Damage,pDamageLabel,delset);
  
  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pBbarElastic[*n] = o_BbarElastic[*o];
    pDeformGrad[*n] = o_DeformGrad[*o];
    pStress[*n] = zero;
    pDamage[*n] = o_Damage[*o];
  }

  (*newState)[pBbarElasticLabel] = pBbarElastic.clone();
  (*newState)[lb->pDeformationMeasureLabel] = pDeformGrad.clone();
  (*newState)[lb->pStressLabel] = pStress.clone();
  (*newState)[pDamageLabel] = pDamage.clone();



  // Initialize the data for the plasticity model
  d_plasticity->allocateCMDataAdd(new_dw,addset, newState,delset,old_dw);


}

void 
HyperElasticPlastic::computeStableTimestep(const Patch* patch,
					   const MPMMaterial* matl,
					   DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

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
    Vector pVel = pVelocity[idx];
    if(pMass[idx] > 0){
      c_dil = sqrt((bulk + 4.*shear/3.)*pVolume[idx]/pMass[idx]);
    }
    else{
      c_dil = 0.0;
      pVel = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
		     Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
		     Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void 
HyperElasticPlastic::computeStressTensor(const PatchSubset* patches,
					 const MPMMaterial* matl,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  // General stuff
  Matrix3 one, zero(0.0); one.Identity(); 

  Matrix3 tensorL; // Velocity gradient
  Matrix3 tensorD; // Rate of deformation tensor
  Matrix3 tensorEta; // Deviatoric part of rate of deformation tensor
  Matrix3 tensorF_new; // Deformation gradient
  Matrix3 trialBbarElastic; // Trial volume preserving elastic Cauchy Green tensor
  Matrix3 tensorFinc; // Increment of the deformation gradient tensor
  Matrix3 trialS; // Trial deviatoric stress tensor
  Matrix3 tensorS; // Actual deviatoric stress tensor
  Matrix3 normal; // Normal to yield surface
  Matrix3 tensorFbar; // Volume preserving part of relative deformation tensor

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double trialSNorm = 0.0;
  double flowStress = 0.0;
  double totalStrainEnergy = 0.0;
  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double rho_0 = matl->getInitialDensity();
  double oneThird = (1.0/3.0);
  double sqrtTwoThird = sqrt(2.0/3.0);

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get the set of particles
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the deformation gradient (F) and the left Cauchy Green
    // tensor (bBar)
    constParticleVariable<Matrix3> pDeformGrad, pBbarElastic;
    old_dw->get(pBbarElastic, pBbarElasticLabel, pset);
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

    // Get the particle damage state
    constParticleVariable<double> pDamage;
    old_dw->get(pDamage, pDamageLabel, pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    constNCVariable<Vector> GVelocity;
    new_dw->get(GVelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
#endif

    // Create and allocate arrays for storing the updated information
    ParticleVariable<Matrix3> pBbarElastic_new, pDeformGrad_new;
    ParticleVariable<Matrix3> pStress_new;
    ParticleVariable<double> pDamage_new;
    ParticleVariable<double> pVolume_new;
    new_dw->allocateAndPut(pBbarElastic_new, 
                           pBbarElasticLabel_preReloc,            pset);
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pVolume_new, 
                           lb->pVolumeDeformedLabel,              pset);

    // Get the plastic strain
    d_plasticity->getInternalVars(pset, old_dw);
    d_plasticity->allocateAndPutInternalVars(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Check if the damage is greater than the cut-off value
      // Then reset everything and return
      if (d_damage->hasFailed(pDamage[idx])) {
         pBbarElastic_new[idx] = one;
         pDeformGrad_new[idx] = one;
         pVolume_new[idx]=pMass[idx]/rho_0;
         pStress_new[idx] = zero;
         pDamage_new[idx] = pDamage[idx];
         d_plasticity->updateElastic(idx);
         Vector pVel = pVelocity[idx];
         double c_dil = sqrt((bulk + 4.0*shear/3.0)*
                              pVolume_new[idx]/pMass[idx]);
         WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
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
        tensorL = computeVelocityGradient(patch, oodx, px[idx], psize[idx], gVelocity);
      else
        tensorL = computeVelocityGradient(patch, oodx, px[idx], gVelocity);
#endif
      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      tensorFinc = tensorL*delT + one;
      double Jinc = tensorFinc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      pDeformGrad_new[idx] = tensorFinc*pDeformGrad[idx];
      double J = pDeformGrad_new[idx].Determinant();
      //cout << "J = " << J << "\n Updated deformation gradient =\n " 
      //     << pDeformGrad_new[idx] << endl;

      // get the volume preserving part of the deformation gradient increment
      tensorFbar = tensorFinc*pow(Jinc,-oneThird);
      //cout << "fbar = \n" << tensorFbar << endl;

      // predict the elastic part of the volume preserving part of the left
      // Cauchy-Green deformation tensor
      trialBbarElastic = tensorFbar*(pBbarElastic[idx]*tensorFbar.Transpose());
      double traceBbarElastic = oneThird*trialBbarElastic.Trace();
      //cout << "Tr(bbar^el) = " << traceBbarElastic 
      //     << "\n bbar^el_trial = " << trialBbarElastic << endl;

      // Compute the trial deviatoric stress
      // trialS is equal to the shear modulus times dev(bElBar)
      // and calculate the norm of the deviatoric stress 
      // (assuming isotropic yield surface)
      trialS = (trialBbarElastic - one*traceBbarElastic)*shear;
      trialSNorm = trialS.Norm();
      //cout << "Norm(s_trial) = " << trialSNorm 
      //     << "\n s_trial = " << trialS << endl;

      // Calculate the flow stress
      flowStress = d_plasticity->computeFlowStress(tensorEta, 
                                                   pTemperature[idx],
                                                   delT, d_tol, matl, idx);
      flowStress *= sqrtTwoThird;
      //cout << "Flow Stress = " << flowStress << endl;

      // Check for plastic loading
      if(trialSNorm > flowStress){

	// Plastic case
        // Calculate delGamma
        double Ielastic = oneThird*traceBbarElastic;
        double muBar = shear*Ielastic;
	double delGamma = (trialSNorm - flowStress)/(2.0*muBar);
        //cout << "Ie = " << Ielastic << " mubar = " << muBar 
        //     << " delgamma = " << delGamma << endl;

        // Calculate normal
	normal = trialS/trialSNorm;
        //cout << " Normal = \n" << normal << endl;

        // The actual deviatoric stress
	tensorS = trialS - normal*2.0*muBar*delGamma;

	// Update deviatoric part of elastic left Cauchy-Green tensor
	pBbarElastic_new[idx] = tensorS/shear + one*Ielastic;

        // Calculate the updated scalar damage parameter
        pDamage_new[idx] = d_damage->computeScalarDamage(tensorEta, tensorS, 
                                                         pTemperature[idx],
                                                         delT, matl, d_tol, 
                                                         pDamage[idx]);

        // Update internal variables
        d_plasticity->updatePlastic(idx, delGamma);

      } else {

	// Elastic case
	tensorS = trialS;

	// Update deviatoric part of elastic left Cauchy-Green tensor
	pBbarElastic_new[idx] = trialBbarElastic;

        // Update the scalar damage parameter
        pDamage_new[idx] = pDamage[idx];

        // Update the internal variables
        d_plasticity->updateElastic(idx);
      }

      // get the hydrostatic part of the stress .. the pressure should ideally
      // be obtained from a strain energy functional of the form U'(J)
      // which is usually satisfied by equations of states that may or may not
      // satisfy small strain elasticity
      tensorF_new = pDeformGrad_new[idx];
      double rho_cur = rho_0/J;
      Matrix3 tensorHy = d_eos->computePressure(matl, bulk, shear, 
                                                tensorF_new, tensorEta, 
                                                tensorS, pTemperature[idx], 
                                                rho_cur, delT);

      //cout << "tensorS = \n" << tensorS << endl << "tensorHy = \n" 
      //     << tensorHy << endl;

      // Compute the total Cauchy stress = 
      // (Kirchhoff stress/J) (volumetric + deviatoric)
      pStress_new[idx] = tensorHy + tensorS/J;
      //cout << "Updated stress =\n " << pStress_new[idx] << endl;

      // Update the volume
      pVolume_new[idx]=pMass[idx]/rho_cur;

      // Compute the strain energy for all the particles
      double U = 0.5*bulk*(0.5*(J*J - 1.0) - log(J));
      double W = 0.5*shear*(pBbarElastic_new[idx].Trace() - 3.0);
      double e = (U + W)*pVolume_new[idx]/J;
      totalStrainEnergy += e;

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.*shear/3.)*pVolume_new[idx]/pMass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
  }
}

void HyperElasticPlastic::carryForward(const PatchSubset* patches,
                                       const MPMMaterial* matl,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<double> pDamage;
    constParticleVariable<Matrix3> pDeformGrad;
    ParticleVariable<Matrix3> pDeformGrad_new;
    ParticleVariable<double> pDamage_new;
    ParticleVariable<Matrix3> pBbarElastic_new;
    constParticleVariable<Matrix3> pBbarElastic;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume_deformed;
    ParticleVariable<Matrix3> pdefm_new,pstress_new;
    constParticleVariable<Matrix3> pdefm,pstress;
    old_dw->get(pBbarElastic, pBbarElasticLabel, pset);
    old_dw->get(pDeformGrad, lb->pDeformationMeasureLabel, pset);
    old_dw->get(pDamage, pDamageLabel, pset);
    old_dw->get(pmass,                 lb->pMassLabel,                 pset);
    old_dw->get(pstress,       lb->pStressLabel,                       pset);
    new_dw->allocateAndPut(pDeformGrad_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pDamage_new,
                           pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pBbarElastic_new, pBbarElasticLabel_preReloc,  pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel, pset);
    new_dw->allocateAndPut(pstress_new,lb->pStressLabel_preReloc,      pset);

    // Get the plastic strain
    d_plasticity->getInternalVars(pset, old_dw);
    d_plasticity->initializeInternalVars(pset, new_dw);

    double rho_orig = matl->getInitialDensity();
    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pstress_new[idx] = pstress[idx];
      pDeformGrad_new[idx] = pDeformGrad[idx];
      pDamage_new[idx] = pDamage[idx];
      pBbarElastic_new[idx] = pBbarElastic[idx];
      pvolume_deformed[idx]=(pmass[idx]/rho_orig);
    }

    new_dw->put(delt_vartype(1.e10), lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void 
HyperElasticPlastic::addInitialComputesAndRequires(Task* task,
						   const MPMMaterial* matl,
						   const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pBbarElasticLabel, matlset);
  task->computes(pDamageLabel, matlset);
 
  // Add internal evolution variables computed by plasticity model
  d_plasticity->addInitialComputesAndRequires(task, matl, patch);
}

void 
HyperElasticPlastic::addComputesAndRequires(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet* patch) const
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel, matlset,Ghost::None);
  if(d_8or27==27)
    task->requires(Task::OldDW, lb->pSizeLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,  matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,  matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel, matlset,Ghost::None);
  task->requires(Task::NewDW, lb->gVelocityLabel,  matlset,gac, NGN);

  task->requires(Task::OldDW, lb->pStressLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);

  task->requires(Task::OldDW, pBbarElasticLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pDamageLabel, matlset,Ghost::None);

#ifdef FRACTURE
  task->requires(Task::NewDW,  lb->pgCodeLabel,    matlset, Ghost::None);
  task->requires(Task::NewDW,  lb->GVelocityLabel, matlset, gac, NGN);
#endif

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(pBbarElasticLabel_preReloc,  matlset);
  task->computes(pDamageLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

  // Add internal evolution variables computed by plasticity model
  d_plasticity->addComputesAndRequires(task, matl, patch);
}

void 
HyperElasticPlastic::addComputesAndRequires(Task* ,
					    const MPMMaterial* ,
					    const PatchSet* ,
					    const bool ) const
{
}


// Needed by MPMICE
double 
HyperElasticPlastic::computeRhoMicroCM(double pressure,
                                       const double p_ref,
				       const MPMMaterial* matl)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;

  double rho_cur;
  if(d_useMPMICEModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  }
  return rho_cur;
}

// Needed by MPMICE
void 
HyperElasticPlastic::computePressEOSCM(double rho_cur,double& pressure,
				       double p_ref,  
				       double& dp_drho, double& C0_sq,
				       const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;

  if(d_useMPMICEModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    C0_sq    = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    C0_sq      = bulk/rho_cur;  // speed of sound squared
  }
}

// Needed by MPMICE
double 
HyperElasticPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

