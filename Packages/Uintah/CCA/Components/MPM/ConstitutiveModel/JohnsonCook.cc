#include "JohnsonCook.h"
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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Util/NotFinished.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

JohnsonCook::JohnsonCook(ProblemSpecP& ps, MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->require("A",d_initialData.A);
  ps->require("B",d_initialData.B);
  ps->require("C",d_initialData.C);
  ps->require("n",d_initialData.n);
  ps->require("m",d_initialData.m);
  ps->require("room_temp",d_initialData.TRoom);
  ps->require("melt_temp",d_initialData.TMelt);
  ps->require("D1",d_initialData.D1);
  ps->require("D2",d_initialData.D2);
  ps->require("D3",d_initialData.D3);
  ps->require("D4",d_initialData.D4);
  ps->require("D5",d_initialData.D5);
  
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
  pDeformRatePlasticLabel = VarLabel::create("p.deformRatePlastic",
			ParticleVariable<Matrix3>::getTypeDescription());
  pDeformRatePlasticLabel_preReloc = VarLabel::create("p.deformRatePlastic+",
			ParticleVariable<Matrix3>::getTypeDescription());
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
			ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
			ParticleVariable<double>::getTypeDescription());
  pDamageLabel = VarLabel::create("p.damage",
			ParticleVariable<double>::getTypeDescription());
  pDamageLabel_preReloc = VarLabel::create("p.damage+",
			ParticleVariable<double>::getTypeDescription());

}

JohnsonCook::~JohnsonCook()
{
  // Destructor 
  VarLabel::destroy(pLeftStretchLabel);
  VarLabel::destroy(pLeftStretchLabel_preReloc);
  VarLabel::destroy(pRotationLabel);
  VarLabel::destroy(pRotationLabel_preReloc);
  VarLabel::destroy(pDeformRatePlasticLabel);
  VarLabel::destroy(pDeformRatePlasticLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pDamageLabel_preReloc);
}

void JohnsonCook::addParticleState(std::vector<const VarLabel*>& from,
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

  from.push_back(pDeformRatePlasticLabel);
  from.push_back(pPlasticStrainLabel);
  from.push_back(pDamageLabel);

  to.push_back(pLeftStretchLabel_preReloc);
  to.push_back(pRotationLabel_preReloc);
  to.push_back(lb->pDeformationMeasureLabel_preReloc);

  to.push_back(lb->pStressLabel_preReloc);

  to.push_back(pDeformRatePlasticLabel_preReloc);
  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pDamageLabel_preReloc);
}

void JohnsonCook::initializeCMData(const Patch* patch,
				   const MPMMaterial* matl,
				   DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<Matrix3> pDeformGrad, pStress, pDeformRatePlastic;
  ParticleVariable<Matrix3> pLeftStretch, pRotation;
  ParticleVariable<double> pPlasticStrain, pDamage;

  new_dw->allocateAndPut(pLeftStretch, pLeftStretchLabel, pset);
  new_dw->allocateAndPut(pRotation, pRotationLabel, pset);
  new_dw->allocateAndPut(pDeformGrad, lb->pDeformationMeasureLabel, pset);

  new_dw->allocateAndPut(pStress, lb->pStressLabel, pset);

  new_dw->allocateAndPut(pDeformRatePlastic, pDeformRatePlasticLabel, pset);
  new_dw->allocateAndPut(pPlasticStrain, pPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pDamage, pDamageLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){

    // To fix : For a material that is initially stressed we need to
    // modify the leftStretch and the stress tensors to comply with the
    // initial stress state
    pLeftStretch[*iter] = one;
    pRotation[*iter] = one;
    pDeformGrad[*iter] = one;

    pStress[*iter] = zero;

    pDeformRatePlastic[*iter] = zero;
    pPlasticStrain[*iter] = 0.0;
    pDamage[*iter] = 0.0;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void JohnsonCook::computeStableTimestep(const Patch* patch,
					const MPMMaterial* matl,
					DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double shear = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    Vector pvelocity_idx = pvelocity[idx];
    if(pmass[idx] > 0){
      c_dil = sqrt((bulk + 4.0*shear/3.0)*pvolume[idx]/pmass[idx]);
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
JohnsonCook::computeStressTensor(const PatchSubset* patches,
				 const MPMMaterial* matl,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw)
{
  // General stuff
  Matrix3 one;   one.Identity(); Matrix3 zero(0.0);
  Matrix3 tensorL; // Velocity gradient
  Matrix3 tensorD; // Rate of deformation
  Matrix3 tensorW; // Spin 
  Matrix3 tensorF; // Deformation gradient
  Matrix3 tensorV; // Left Cauchy-Green stretch
  Matrix3 tensorR; // Rotation 
  Matrix3 tensorSig; // The Cauchy stress
  Matrix3 tensorDp; // Rate of plastic deformation
  Matrix3 tensorEta; // Deviatoric part of tensor D
  Matrix3 tensorEtap; // Deviatoric part of tensor Dp
  Matrix3 tensorS; // Devaitoric part of tensor Sig
  Matrix3 tensorF_new; // Deformation gradient
  double equivStress = 0.0;
  double plasticStrainRate = 0.0;
  double plasticStrain = 0.0;
  double temperature = 0.0;
  double flowStress = 0.0;
  double totalStrainEnergy = 0.0;
  double tolerance = 5.0e-10;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double lambda = bulk - (2.0/3.0)*shear;
  double sqrtTwo = sqrt(2.0);
  double sqrtThree = sqrt(3.0);

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Create array for the particle position
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

    // Get the particle stress
    // Get the plastic part of rate of deformation
    // Get the plastic strain
    // Get the particle temperature
    constParticleVariable<Matrix3> pStress;
    constParticleVariable<Matrix3> pDeformRatePlastic;
    constParticleVariable<double> pPlasticStrain;
    constParticleVariable<double> pTemperature;
    constParticleVariable<double> pDamage;

    old_dw->get(pStress, lb->pStressLabel, pset);
    old_dw->get(pDeformRatePlastic, pDeformRatePlasticLabel, pset);
    old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
    old_dw->get(pTemperature, lb->pTemperatureLabel, pset);
    old_dw->get(pDamage, pDamageLabel, pset);
    
    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    // Create and allocate arrays for storing the updated information
    ParticleVariable<Matrix3> pLeftStretch_new, pRotation_new, pDeformGrad_new;

    ParticleVariable<Matrix3> pStress_new;

    ParticleVariable<Matrix3> pDeformRatePlastic_new;
    ParticleVariable<double> pPlasticStrain_new;
    ParticleVariable<double> pDamage_new;
    ParticleVariable<double> pVolume_deformed;

    new_dw->allocateAndPut(pLeftStretch_new, pLeftStretchLabel_preReloc, pset);
    new_dw->allocateAndPut(pRotation_new, pRotationLabel_preReloc, pset);
    new_dw->allocateAndPut(pDeformGrad_new, lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new, lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pDeformRatePlastic_new, pDeformRatePlasticLabel_preReloc, pset);
    new_dw->allocateAndPut(pPlasticStrain_new, pPlasticStrainLabel_preReloc, pset);
    new_dw->allocateAndPut(pDamage_new, pDamageLabel_preReloc, pset);
    new_dw->allocateAndPut(pVolume_deformed, lb->pVolumeDeformedLabel, pset);

    // Loop thru particles
    for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Calculate the velocity gradient (L) from the grid velocity
      if(d_8or27==27) 
         tensorL = computeVelocityGradient(patch, oodx, px[idx], psize[idx], gVelocity);
      else 
         tensorL = computeVelocityGradient(patch, oodx, px[idx], gVelocity);

      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorW = (tensorL - tensorL.Transpose())*0.5;
      for (int ii = 1; ii < 4; ++ii) {
        for (int jj = 1; jj < 4; ++jj) {
          if (fabs(tensorD(ii,jj)) < tolerance) tensorD(ii,jj) = 0.0;
          if (fabs(tensorW(ii,jj)) < tolerance) tensorW(ii,jj) = 0.0;
        }
      }

      // Calculate the incremental update of the left stretch (V) and the rotation (R)
      tensorV = pLeftStretch[idx];
      tensorR = pRotation[idx];
      computeUpdatedVR(delT, tensorD, tensorW, tensorV, tensorR);
      for (int ii = 1; ii < 4; ++ii) {
        for (int jj = 1; jj < 4; ++jj) {
          if (fabs(tensorV(ii,jj)) < tolerance) tensorV(ii,jj) = 0.0;
          if (fabs(tensorR(ii,jj)) < tolerance) tensorR(ii,jj) = 0.0;
        }
      }
      tensorF_new = tensorV*tensorR;

      // Update the kinematic variables
      pLeftStretch_new[idx] = tensorV;
      pRotation_new[idx] = tensorR;
      pDeformGrad_new[idx] = tensorF_new;

      // Rotate the total rate of deformation tensor,
      // the plastic rate of deformation tensor, and the Cauchy stress
      // back to the material configuration and calculate their
      // deviatoric parts
      //cout << endl << "Particle # = " << idx << " Position = " << px[idx] << endl; 
      //cout << "Bulk modulus = " << bulk << " Shear modulus = " << shear << endl;
      //cout << "Velocity Gradient = \n" << tensorL << endl;
      tensorD = (tensorR.Transpose())*(tensorD*tensorR);
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);
      //cout << "Rate of Deformation = \n" << tensorD << endl;
      //cout << "Deviatoric part of rate of deformation = \n" << tensorEta << endl;

      tensorSig = pStress[idx];
      tensorSig = (tensorR.Transpose())*(tensorSig*tensorR);
      tensorS = tensorSig - one*(tensorSig.Trace()/3.0);
      Matrix3 tensorHy = tensorSig - tensorS;
      //cout << "Stress at beginning of time step = \n" << tensorSig << endl;
      //cout << "Hydrostatic part of stress = \n" << tensorHy << endl;
      //cout << "Deviatoric part of stress = \n" << tensorS << endl;

      // Integrate the stress rate equation to get a elastic trial stress
      Matrix3 trialSig = tensorSig + 
	                 (one*(tensorD.Trace()*lambda) + tensorD*(2.0*shear))*delT;
      Matrix3 trialS_old = trialSig - one*(trialSig.Trace()/3.0);
      Matrix3 trialS = tensorS + tensorEta*(2.0*shear*delT);
      //cout << "Trial Stress = \n" << trialSig << endl;
      //cout << "Deviatoric part of Trial Stress (old) = \n" << trialS_old << endl;
      //cout << "Deviatoric part of Trial Stress (new) = \n" << trialS << endl;

      // To determine if the stress is above or below yield used a von Mises yield
      // criterion 
      // Assumption: Material yields, on average, like a von Mises solid

      // Calculate the square of the flow stress
      // from the plastic strain rate, the plastic strain and  
      // the particle temperature
      
      plasticStrainRate = sqrt(tensorEta.NormSquared()*2.0/3.0);
      plasticStrain = pPlasticStrain[idx] + plasticStrainRate*delT;
      //cout << "epdot = " << plasticStrainRate << " ep = " << plasticStrain << endl;

      temperature = pTemperature[idx];
      flowStress = evaluateFlowStress(plasticStrain, plasticStrainRate, temperature);
      flowStress *= flowStress;

      // Calculate the J2 equivalent stress (assuming isotropic yield surface)
      equivStress = (trialS.NormSquared())*1.5;

      //cout << "Checking for plasticity : sig_eq = " << sqrt(equivStress) 
      //     << " sig_flow = " << sqrt(flowStress) << endl;
      
      if (flowStress > equivStress) {

        // For the elastic region : the updated stress is the trial stress
        tensorSig = trialSig;
        //cout << "Elastic Stress = \n" << tensorSig << endl;

        // Compute the strain energy for the particles
        double pStrainEnergy = (tensorD(1,1)*tensorSig(1,1) +
	   tensorD(2,2)*tensorSig(2,2) + tensorD(3,3)*tensorSig(3,3) +
	   2.0*(tensorD(1,2)*tensorSig(1,2) + 
           tensorD(1,3)*tensorSig(1,3) + tensorD(2,3)*tensorSig(2,3)))*
           pVolume_deformed[idx]*delT;
        totalStrainEnergy += pStrainEnergy;		   

        // Rotate the stress rate back to the laboratory coordinates
        // to get the "true" Cauchy stress
        tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

        // Save the updated data
        pDeformRatePlastic_new[idx] = zero;
        pStress_new[idx] = tensorSig;
        pPlasticStrain_new[idx] = pPlasticStrain[idx];

      } else {

        // Using the algorithm from Zocher, Maudlin, Chen, Flower-Maudlin
        // European Congress on Computational Methods in Applied Sciences and Engineering
        // September 11-14, 2000.
 
        // Basic assumption is that all strain rate is plastic strain rate

        // Calculate the tensor u (at start of time interval)
        double sqrtSxS = tensorS.Norm(); ASSERT(sqrtSxS != 0);
        Matrix3 tensorU = tensorS*(sqrtThree/sqrtSxS);

        // Calculate cplus and initial values of dstar, gammadot and theta
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
        } while (fabs(dstar-dstar_old) > tolerance && count < 5);

        double sig = sqrt(flowStress); ASSERT(sig != 0);

        // Calculate delGammaEr
        double delGammaEr =  (sqrtTwo*sig - sqrtThree*sqrtSxS)/(2.0*shear*cplus);

        // Calculate delGamma
        double delGamma = dstar/cplus*delT - delGammaEr;

        // Calculate Stilde
        double denom = 1.0 + (3.0*sqrtTwo*shear*delGamma)/sig; ASSERT(denom != 0);
        //Matrix3 Stilde = (tensorS + tensorEta*(2.0*shear*delT))/denom;
        Matrix3 Stilde = trialS/denom;
        
        // Do radial return adjustment
        tensorS = Stilde*(sig*sqrtTwo/(sqrtThree*Stilde.Norm()));
        equivStress = sqrt((tensorS.NormSquared())*1.5);
        //cout << "After plastic adjustment : \n";
        //cout << "FlowStress = " << sqrt(flowStress) << " Equiv. Stress = " 
        //     << equivStress << endl;

        // Update the total stress tensor
        tensorSig = tensorS + tensorHy;
        //cout << "New Stress after plastic update for particle "<<idx<<" = \n" << tensorS << endl
        //     << "+" << tensorHy << endl;
        //cout << "Plastic Stress = \n" << tensorSig << endl;

        // Compute the strain energy for the particles
        double pStrainEnergy = (tensorD(1,1)*tensorSig(1,1) +
	   tensorD(2,2)*tensorSig(2,2) + tensorD(3,3)*tensorSig(3,3) +
	   2.0*(tensorD(1,2)*tensorSig(1,2) + 
           tensorD(1,3)*tensorSig(1,3) + tensorD(2,3)*tensorSig(2,3)))*
           pVolume_deformed[idx]*delT;
        totalStrainEnergy += pStrainEnergy;		   

        // Rotate the stress and deformation rate back to the laboratory coordinates
        tensorEta = (tensorR*tensorEta)*(tensorR.Transpose());
        tensorSig = (tensorR*tensorSig)*(tensorR.Transpose());

        // Save the new data
        pDeformRatePlastic_new[idx] = tensorEta;
        pStress_new[idx] = tensorSig;
        pPlasticStrain_new[idx] = plasticStrain;
      }

      // Calculate the updated scalar damage parameter
      double epsFrac = calcStrainAtFracture(pStress_new[idx], plasticStrainRate, temperature);
      double epsInc = pPlasticStrain_new[idx] - pPlasticStrain[idx];
      if (epsFrac == 0) 
        pDamage_new[idx] = pDamage[idx];
      else 
        pDamage_new[idx] = pDamage[idx] + epsInc/epsFrac;
      //cout << "Dnew = " << pDamage_new[idx] << " Dold = " << pDamage[idx]
      //     << " epsFrac = " << epsFrac << " epsInc = " << epsInc << endl;

      // Calculate the deformed volume
      tensorF = pDeformGrad[idx];
      Matrix3 deformationGradientInc = tensorF_new*(tensorF.Inverse());
      double Jinc = deformationGradientInc.Determinant();
      pVolume_deformed[idx]=Jinc*pVolume[idx];
      //cout << "Old Volume = " << pVolume[idx] 
      //     << " New Volume = " << pVolume_deformed[idx] << endl;

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
JohnsonCook::addInitialComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pLeftStretchLabel, matlset);
  task->computes(pRotationLabel, matlset);
  task->computes(pDeformRatePlasticLabel, matlset);
  task->computes(pPlasticStrainLabel, matlset);
  task->computes(pDamageLabel, matlset);
}

void 
JohnsonCook::addComputesAndRequires(Task* task,
				    const MPMMaterial* matl,
				    const PatchSet*) const
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
  task->requires(Task::OldDW, pDeformRatePlasticLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pPlasticStrainLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pDamageLabel, matlset,Ghost::None);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(pLeftStretchLabel_preReloc,  matlset);
  task->computes(pRotationLabel_preReloc, matlset);
  task->computes(pDeformRatePlasticLabel_preReloc,  matlset);
  task->computes(pPlasticStrainLabel_preReloc, matlset);
  task->computes(pDamageLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);
}

Matrix3
JohnsonCook::computeVelocityGradient(const Patch* patch,
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
JohnsonCook::computeVelocityGradient(const Patch* patch,
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

void
JohnsonCook::computeUpdatedVR(const double& delT,
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
  Matrix3 oneMinusOmegaInv = oneMinusOmega.Inverse();
  Matrix3 onePlusOmega = one + Omega*(0.5*delT);
  RR = RR*(oneMinusOmegaInv*onePlusOmega);

  // Check the ortogonality of R
  if (!RR.Orthogonal()) {
    // Do something here that restores orthogonality
  }

  // Update the left Cauchy-Green stretch tensor (V)
  VV = VV + ((DD+WW)*VV - VV*Omega)*delT;
}

Matrix3 
JohnsonCook::computeRateofRotation(const Matrix3& tensorV, 
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

double 
JohnsonCook::evaluateFlowStress(const double& ep, 
                                const double& epdot,
                                const double& T)
{
  double strainPart = d_initialData.A + d_initialData.B*pow(ep,d_initialData.n);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_initialData.C);
  else
    strainRatePart = 1.0 + d_initialData.C*log(epdot);
  double Tr = d_initialData.TRoom;
  double Tm = d_initialData.TMelt;
  double m = d_initialData.m;
  double Tstar = (T-Tr)/(Tm-Tr);
  if (fabs(Tstar) < 1.0e-8) Tstar = 0.0;
  if (Tstar < 0.0) {
    cout << " ep = " << ep << " Strain Part = " << strainPart << endl;
    cout << "epdot = " << epdot << " Strain Rate Part = " << strainRatePart << endl;
    cout << "Tstar = " << Tstar << " T = " << T << " Tr = " << Tr << " Tm = " << Tm << endl;
  }
  ASSERT(Tstar > -1.0e-8);
  double tm = pow(Tstar,m);
  double tempPart = 1.0 - tm;
  return (strainPart*strainRatePart*tempPart);
}

double 
JohnsonCook::calcStrainAtFracture(const Matrix3& stress, 
                                  const double& epdot,
                                  const double& T)
{
  double sigMean = stress.Trace()/3.0;
  double sigEquiv = sqrt((stress.NormSquared())*1.5);
  double sigStar = 0.0;
  if (sigEquiv != 0) sigStar = sigMean/sigEquiv;
  double stressPart = d_initialData.D1 + d_initialData.D2*exp(d_initialData.D3*sigStar);
  //cout << "sigMean = " << sigMean << " sigEquiv = " << sigEquiv << " sigStar = " << sigStar 
  //     << " Stress Part = " << stressPart << endl;
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_initialData.D4);
  else
    strainRatePart = 1.0 + d_initialData.D4*log(epdot);
  //cout << "epdot = " << epdot << " Strain Rate Part = " << strainRatePart << endl;
  double Tstar = (T-d_initialData.TRoom)/(d_initialData.TMelt-d_initialData.TRoom);
  double tempPart = 1.0 + d_initialData.D5*Tstar;
  //cout << "Tstar = " << Tstar << " Temp. Part = " << tempPart << endl;
  return (stressPart*strainRatePart*tempPart);
}

double JohnsonCook::computeRhoMicroCM(double pressure,
                                      const double p_ref,
				      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  return rho_cur;
}

void JohnsonCook::computePressEOSCM(double rho_cur,double& pressure,
                                        double p_ref,  
                                        double& dp_drho, double& tmp,
                                        const MPMMaterial* matl)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure   = p_ref + p_g;
  dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp        = bulk/rho_cur;  // speed of sound squared
}

double JohnsonCook::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

