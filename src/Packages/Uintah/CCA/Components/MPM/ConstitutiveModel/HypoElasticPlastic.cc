
#include "HypoElasticPlastic.h"
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
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

HypoElasticPlastic::HypoElasticPlastic(ProblemSpecP& ps, MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  d_tol = 1.0e-10;
  ps->get("tolerance",d_tol);

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

  pLeftStretchLabel = VarLabel::create("p.leftStretch",
	ParticleVariable<Matrix3>::getTypeDescription());
  pLeftStretchLabel_preReloc = VarLabel::create("p.leftStretch+",
	ParticleVariable<Matrix3>::getTypeDescription());
  pRotationLabel = VarLabel::create("p.rotation",
	ParticleVariable<Matrix3>::getTypeDescription());
  pRotationLabel_preReloc = VarLabel::create("p.rotation+",
	ParticleVariable<Matrix3>::getTypeDescription());
  pDamageLabel = VarLabel::create("p.damage",
	ParticleVariable<double>::getTypeDescription());
  pDamageLabel_preReloc = VarLabel::create("p.damage+",
	ParticleVariable<double>::getTypeDescription());
}

HypoElasticPlastic::~HypoElasticPlastic()
{
  // Destructor 
  VarLabel::destroy(pLeftStretchLabel);
  VarLabel::destroy(pLeftStretchLabel_preReloc);
  VarLabel::destroy(pRotationLabel);
  VarLabel::destroy(pRotationLabel_preReloc);
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pDamageLabel_preReloc);

  delete d_plasticity;
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
  from.push_back(pLeftStretchLabel);
  from.push_back(pRotationLabel);
  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);
  from.push_back(pDamageLabel);

  to.push_back(pLeftStretchLabel_preReloc);
  to.push_back(pRotationLabel_preReloc);
  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);
  to.push_back(pDamageLabel_preReloc);

  // Add the particle state for the plasticity model
  d_plasticity->addParticleState(from, to);
}

void HypoElasticPlastic::initializeCMData(const Patch* patch,
					  const MPMMaterial* matl,
					  DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pDeformGrad, pStress;
  ParticleVariable<Matrix3> pLeftStretch, pRotation;
  ParticleVariable<double> pDamage;

  new_dw->allocateAndPut(pLeftStretch, pLeftStretchLabel, pset);
  new_dw->allocateAndPut(pRotation, pRotationLabel, pset);
  new_dw->allocateAndPut(pDeformGrad, lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress, lb->pStressLabel, pset);
  new_dw->allocateAndPut(pDamage, pDamageLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){

    // To fix : For a material that is initially stressed we need to
    // modify the leftStretch and the stress tensors to comply with the
    // initial stress state
    pLeftStretch[*iter] = one;
    pRotation[*iter] = one;
    pDeformGrad[*iter] = one;
    pStress[*iter] = zero;
    pDamage[*iter] = 0.0;
  }

  // Initialize the data for the plasticity model
  d_plasticity->initializeInternalVars(pset, new_dw);

  computeStableTimestep(patch, matl, new_dw);
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

  double equivStress = 0.0;
  double flowStress = 0.0;
  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double rho_0 = matl->getInitialDensity();
  double sqrtTwo = sqrt(2.0);
  double sqrtThree = sqrt(3.0);
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

    // Get the deformation gradient (F), left stretch (V) and rotation (R)
    // Note : The deformation gradient from the old datawarehouse is no
    // longer used, but it is updated for possible use elsewhere
    constParticleVariable<Matrix3> pLeftStretch, pRotation, pDeformGrad;
    old_dw->get(pLeftStretch, pLeftStretchLabel, pset);
    old_dw->get(pRotation, pRotationLabel, pset);
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
    ParticleVariable<Matrix3> pLeftStretch_new, pRotation_new, pDeformGrad_new;
    ParticleVariable<Matrix3> pStress_new;
    ParticleVariable<double> pDamage_new;
    ParticleVariable<double> pVolume_deformed;
    new_dw->allocateAndPut(pLeftStretch_new, pLeftStretchLabel_preReloc,            pset);
    new_dw->allocateAndPut(pRotation_new,    pRotationLabel_preReloc,               pset);
    new_dw->allocateAndPut(pDeformGrad_new,  lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pDamage_new,      pDamageLabel_preReloc,                 pset);
    new_dw->allocateAndPut(pVolume_deformed, lb->pVolumeDeformedLabel,              pset);

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
      tensorW = (tensorL - tensorL.Transpose())*0.5;
      for (int ii = 1; ii < 4; ++ii) {
        for (int jj = 1; jj < 4; ++jj) {
	  tensorD(ii,jj) = (fabs(tensorD(ii,jj)) < d_tol) ? 0.0 : tensorD(ii,jj);
	  tensorW(ii,jj) = (fabs(tensorW(ii,jj)) < d_tol) ? 0.0 : tensorW(ii,jj);
        }
      }

      // Calculate the incremental update of the left stretch (V) and the rotation (R)
      tensorV = pLeftStretch[idx];
      tensorR = pRotation[idx];
      computeUpdatedVR(delT, tensorD, tensorW, tensorV, tensorR);
      tensorF_new = tensorV*tensorR;
      double J = tensorF_new.Determinant();
      //cout << "For particle " << idx << " kinematics \n";
      //cout << "Velocity Gradient = \n" << tensorL << endl;
      //cout << "D = \n" << tensorD << endl;
      //cout << "V = \n" << tensorV << endl;
      //cout << "R = \n" << tensorR << endl;

      // Update the kinematic variables
      pLeftStretch_new[idx] = tensorV;
      pRotation_new[idx] = tensorR;
      pDeformGrad_new[idx] = tensorF_new;

      // Rotate the total rate of deformation tensor,
      // the plastic rate of deformation tensor, and the Cauchy stress
      // back to the material configuration and calculate their
      // deviatoric parts
      tensorD = (tensorR.Transpose())*(tensorD*tensorR);
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);

      tensorSig = pStress[idx];
      tensorSig = (tensorR.Transpose())*(tensorSig*tensorR);
      Matrix3 tensorP = one*(tensorSig.Trace()/3.0);
      tensorS = tensorSig - tensorP;

      // Integrate the stress rate equation to get a trial stress
      // and calculate the J2 equivalent stress (assuming isotropic yield surface)
      Matrix3 trialS = tensorS + tensorEta*(2.0*shear*delT);
      equivStress = (trialS.NormSquared())*1.5;

      // To determine if the stress is above or below yield used a von Mises yield
      // criterion Assumption: Material yields, on average, like a von Mises solid
      flowStress = d_plasticity->computeFlowStress(tensorEta, tensorS, pTemperature[idx],
                                                   delT, d_tol, matl, idx);
      double flowStressSq = flowStress*flowStress;

      //cout << "Trial values : Particle " << idx << "\n";
      //cout << "FlowStress = " << flowStress << " Equiv. Stress = " 
      //     << sqrt(equivStress) << endl;
      if (flowStressSq >= equivStress) {

        // Calculate the deformed volume
        double rho_cur = rho_0/J;
        pVolume_deformed[idx]=pMass[idx]/rho_cur;

        // For the elastic region : the updated stress is the trial stress
        Matrix3 tensorHy = d_eos->computePressure(matl, bulk, shear, tensorF_new, tensorD, 
                                                 tensorP, pTemperature[idx], rho_cur, delT);
        Matrix3 tensorSig = trialS + tensorHy;
        //cout << "For particle " << idx << " elastic \n";
        //cout << "tensorD = \n" << tensorD << endl;
        //cout << "tensorS = \n" << tensorS << endl;
        //cout << "tensorP = \n" << tensorP << endl;
        //cout << "tensorHy = \n" << tensorHy << endl;
        //cout << "tensorSig = \n" << tensorSig << endl;

        // Rotate the stress rate back to the laboratory coordinates
        // to get the "true" Cauchy stress
        tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

        // Save the updated data
        pStress_new[idx] = tensorSig;
        pDamage_new[idx] = pDamage[idx];
        
        // Update the internal variables
        d_plasticity->updateElastic(idx);

      } else {

        // Using the algorithm from Zocher, Maudlin, Chen, Flower-Maudlin
        // European Congress on Computational Methods in Applied Sciences and Engineering
        // September 11-14, 2000.
        // Basic assumption is that all strain rate is plastic strain rate
        ASSERT(flowStress != 0);

        Matrix3 Stilde;
        double delGamma;
        double sqrtSxS = tensorS.Norm(); 
        // cout << " Before update : sqrtSxS = " << sqrtSxS << endl;
        // cout << "tensorS = \n" << tensorS << endl;
        if ((sqrtSxS == 0) && (pDamage[idx] < 1)) { 
	  // If the material goes plastic in the first step, 
	  Stilde = trialS;
          delGamma = ((sqrt(equivStress)-flowStress)/(2.0*shear))/(1.0+bulk/(3.0*shear));
        } else {
	  // Calculate the tensor u (at start of time interval)
	  ASSERT(sqrtSxS != 0);
	  Matrix3 tensorU = tensorS*(sqrtThree/sqrtSxS);

	  // Calculate cplus and initial values of dstar, gammadot and theta
	  ASSERT(tensorS.Determinant() != 0.0);
	  double gammadotplus = tensorEta.Contract(tensorS.Inverse())*sqrtSxS/sqrtThree;
	  double cplus = tensorU.NormSquared(); ASSERT(cplus != 0);
	  double sqrtcplus = sqrt(cplus);
	  double dstar = tensorU.Contract(tensorEta); ASSERT(dstar != 0);
	  double theta = (dstar - cplus*gammadotplus)/dstar;

	  // Calculate u_eta and u_q
	  double sqrtEtaxEta = tensorEta.Norm();
	  Matrix3 tensorU_eta = tensorEta/sqrtEtaxEta;
	  Matrix3 tensorU_q = tensorS/sqrtSxS;

	  // Calculate new dstar
	  int count = 0;
	  double dstar_old = 0.0;
	  do {
	    dstar_old = dstar;
	    Matrix3 temp = (tensorU_q+tensorU_eta)*(0.5*theta) + tensorU_eta*(1.0-theta);
	    dstar = (tensorU_eta.Contract(temp))*sqrtcplus;
	    theta = (dstar - cplus*gammadotplus)/dstar;
	    ++count;
	  } while (fabs(dstar-dstar_old) > d_tol && count < 5);

	  // Calculate delGammaEr
	  double delGammaEr =  (sqrtTwo*flowStress - sqrtThree*sqrtSxS)/(2.0*shear*cplus);

	  // Calculate delGamma
	  delGamma = dstar/cplus*delT - delGammaEr;

	  // Calculate Stilde
	  double denom = 1.0 + (3.0*sqrtTwo*shear*delGamma)/flowStress; ASSERT(denom != 0);
	  Stilde = trialS/denom;
        }
        
        // Do radial return adjustment
        tensorS = Stilde*(flowStress*sqrtTwo/(sqrtThree*Stilde.Norm()));
        equivStress = sqrt((tensorS.NormSquared())*1.5);
        //cout << "After plastic adjustment : \n";
        //cout << "FlowStress = " << flowStress << " Equiv. Stress = " << equivStress << endl;

        // Calculate the updated scalar damage parameter
        pDamage_new[idx] = d_damage->computeScalarDamage(tensorEta, tensorS, 
                                                         pTemperature[idx],
                                                         delT, matl, d_tol, pDamage[idx]);

        // Calculate the deformed volume
        double rho_cur = rho_0/J;
        pVolume_deformed[idx]=pMass[idx]/rho_cur;

        // Update the total stress tensor
        Matrix3 tensorHy = d_eos->computePressure(matl, bulk, shear, tensorF_new, tensorD, 
                                                 tensorP, pTemperature[idx], rho_cur, delT);
        Matrix3 tensorSig = tensorS + tensorHy;
        //cout << "For particle " << idx << " plastic \n";
        //cout << "tensorD = \n" << tensorD << endl;
        //cout << "tensorS = \n" << tensorS << endl;
        //cout << "tensorP = \n" << tensorP << endl;
        //cout << "tensorHy = \n" << tensorHy << endl;
        //cout << "tensorSig = \n" << tensorSig << endl;

        // Rotate the stress and deformation rate back to the laboratory coordinates
        tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

        // Save the new data
        pStress_new[idx] = tensorSig;
        
        // Update internal variables
        d_plasticity->updatePlastic(idx, delGamma);

      }
      // Compute the strain energy for the particles
      // Rotate the deformation rate back to the laboratory coordinates
      tensorD = (tensorR*tensorD)*(tensorR.Transpose());
      Matrix3 avgStress = (pStress_new[idx] + pStress[idx])*0.5;
      double pStrainEnergy = (tensorD(1,1)*avgStress(1,1) +
				tensorD(2,2)*avgStress(2,2) + tensorD(3,3)*avgStress(3,3) +
				2.0*(tensorD(1,2)*avgStress(1,2) + 
				tensorD(1,3)*avgStress(1,3) + tensorD(2,3)*avgStress(2,3)))*
	                        pVolume_deformed[idx]*delT;
      totalStrainEnergy += pStrainEnergy;		   

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.0*shear/3.0)*pVolume_deformed[idx]/pMass[idx]);
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
HypoElasticPlastic::computeStressTensor(const PatchSubset* ,
				const MPMMaterial* ,
				DataWarehouse* ,
				DataWarehouse* ,
				Solver* ,
				const bool )
{
}
	 

void 
HypoElasticPlastic::addInitialComputesAndRequires(Task* task,
						  const MPMMaterial* matl,
						  const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pLeftStretchLabel, matlset);
  task->computes(pRotationLabel, matlset);
  task->computes(pDamageLabel, matlset);
 
  // Add internal evolution variables computed by plasticity model
  d_plasticity->addInitialComputesAndRequires(task, matl, patch);
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

  task->requires(Task::OldDW, pLeftStretchLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pRotationLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pDamageLabel, matlset,Ghost::None);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(pLeftStretchLabel_preReloc,  matlset);
  task->computes(pRotationLabel_preReloc, matlset);
  task->computes(pDamageLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

  // Add internal evolution variables computed by plasticity model
  d_plasticity->addComputesAndRequires(task, matl, patch);
}

void 
HypoElasticPlastic::addComputesAndRequires(Task* ,
				   const MPMMaterial* ,
				   const PatchSet* ,
				   const bool ) const
{
}

void
HypoElasticPlastic::computeUpdatedVR(const double& delT,
				     const Matrix3& DD, 
				     const Matrix3& WW,
				     Matrix3& VV, 
				     Matrix3& RR)  
{
  // Note:  The incremental polar decomposition algorithm is from
  // Flanagan and Taylor, 1987, Computer Methods in Applied Mechanics and Engineering,
  // v. 62, p.315.

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

double HypoElasticPlastic::computeRhoMicroCM(double pressure,
					     const double p_ref,
					     const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  double p_gauge = pressure - p_ref;
  return (rho_orig/(1.0-p_gauge/bulk));
}

void HypoElasticPlastic::computePressEOSCM(double rho_cur,double& pressure,
					   double p_ref,  
					   double& dp_drho, double& tmp,
					   const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared
}

double HypoElasticPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

