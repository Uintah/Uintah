
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
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Util/NotFinished.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

HyperElasticPlastic::HyperElasticPlastic(ProblemSpecP& ps, MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  
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
  Matrix3 one; one.Identity(); 

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

    // Create and allocate arrays for storing the updated information
    ParticleVariable<Matrix3> pBbarElastic_new, pDeformGrad_new;
    ParticleVariable<Matrix3> pStress_new;
    ParticleVariable<double> pDamage_new;
    ParticleVariable<double> pVolume_new;
    new_dw->allocateAndPut(pBbarElastic_new, pBbarElasticLabel_preReloc,            pset);
    new_dw->allocateAndPut(pDeformGrad_new,  lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pDamage_new,      pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pVolume_new, lb->pVolumeDeformedLabel,              pset);

    // Get the plastic strain
    d_plasticity->getInternalVars(pset, old_dw);
    d_plasticity->allocateAndPutInternalVars(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Calculate the velocity gradient (L) from the grid velocity
      if (d_8or27==27) 
        tensorL = computeVelocityGradient(patch, oodx, px[idx], psize[idx], gVelocity);
      else 
	tensorL = computeVelocityGradient(patch, oodx, px[idx], gVelocity);

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

      // get the volume preserving part of the deformation gradient increment
      tensorFbar = tensorFinc*pow(Jinc,-oneThird);

      // predict the elastic part of the volume preserving part of the left
      // Cauchy-Green deformation tensor
      trialBbarElastic = tensorFbar*(pBbarElastic[idx]*tensorFbar.Transpose());
      double traceBbarElastic = oneThird*trialBbarElastic.Trace();

      // Compute the trial deviatoric stress
      // trialS is equal to the shear modulus times dev(bElBar)
      trialS = (trialBbarElastic - one*traceBbarElastic)*shear;

      // Calculate the norm of the deviatoric stress (assuming isotropic yield surface)
      trialSNorm = trialS.Norm();

      // Calculate the flow stress
      flowStress = d_plasticity->computeFlowStress(tensorEta, tensorS, pTemperature[idx],
                                                   delT, d_tol, matl, idx);
      flowStress *= sqrtTwoThird;

      // Check for plastic loading
      if(trialSNorm > flowStress){

	// Plastic case
        // Calculate delGamma
        double Ielastic = oneThird*traceBbarElastic;
        double muBar = shear*Ielastic;
	double delGamma = (trialSNorm - flowStress)*0.5*muBar;

        // Calculate normal
	normal = trialS/trialSNorm;

        // The actual deviatoric stress
	tensorS = trialS - normal*2.0*muBar*delGamma;

	// Update deviatoric part of elastic left Cauchy-Green tensor
	pBbarElastic_new[idx] = tensorS/shear + one*Ielastic;

        // Calculate the updated scalar damage parameter
        pDamage_new[idx] = d_damage->computeScalarDamage(tensorEta, tensorS, 
                                                         pTemperature[idx],
                                                         delT, matl, d_tol, pDamage[idx]);

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
      Matrix3 tensorHy = d_eos->computePressure(matl, bulk, shear, tensorF_new, tensorEta, 
                                               tensorS, pTemperature[idx], rho_cur, delT);

      // compute the total Cauchy stress = (Kirchhoff stress/J) (volumetric + deviatoric)
      pStress_new[idx] = tensorHy + tensorS/J;

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

  task->requires(Task::OldDW, pBbarElasticLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pDamageLabel, matlset,Ghost::None);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(pBbarElasticLabel_preReloc,  matlset);
  task->computes(pDamageLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

  // Add internal evolution variables computed by plasticity model
  d_plasticity->addComputesAndRequires(task, matl, patch);
}

double 
HyperElasticPlastic::computeRhoMicroCM(double pressure,
                                       const double p_ref,
				       const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  return rho_cur;
}

void 
HyperElasticPlastic::computePressEOSCM(double rho_cur,double& pressure,
				       double p_ref,  
				       double& dp_drho, double& C_0sq,
				       const MPMMaterial* matl)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  C_0sq = bulk/rho_cur;  // speed of sound squared
}

double 
HyperElasticPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

/*
Matrix3
HyperElasticPlastic::computeVelocityGradient(const Patch* patch,
					   const double* oodx, 
					   const Point& px, 
					   const Vector& psize, 
					   constNCVariable<Vector>& gVelocity) 
{
  // Initialize
  Matrix3 velGrad(0.0);

  // Get the node indices that surround the cell
  IntVector ni[MAX_BASIS];
  Vector d_S[MAX_BASIS];

  patch->findCellAndShapeDerivatives27(px, ni, d_S, psize);

  //cout << "ni = " << ni << endl;
  for(int k = 0; k < d_8or27; k++) {
    //if(patch->containsNode(ni[k])) {
    const Vector& gvel = gVelocity[ni[k]];
    //cout << "GridVel = " << gvel << endl;
    for (int j = 0; j<3; j++){
      for (int i = 0; i<3; i++) {
	velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
      }
    }
    //}
  }
  //cout << "VelGrad = " << velGrad << endl;
  return velGrad;
}

Matrix3
HyperElasticPlastic::computeVelocityGradient(const Patch* patch,
					   const double* oodx, 
					   const Point& px, 
					   constNCVariable<Vector>& gVelocity) 
{
  // Initialize
  Matrix3 velGrad(0.0);

  // Get the node indices that surround the cell
  IntVector ni[MAX_BASIS];
  Vector d_S[MAX_BASIS];

  patch->findCellAndShapeDerivatives(px, ni, d_S);

  for(int k = 0; k < d_8or27; k++) {
    const Vector& gvel = gVelocity[ni[k]];
    //cout << "GridVel = " << gvel << endl;
    for (int j = 0; j<3; j++){
      for (int i = 0; i<3; i++) {
	velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
      }
    }
  }
  //cout << "VelGrad = " << velGrad << endl;
  return velGrad;
}
*/

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

