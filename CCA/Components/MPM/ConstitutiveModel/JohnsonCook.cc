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
}

JohnsonCook::~JohnsonCook()
{
  // Destructor 
}

void JohnsonCook::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);
  from.push_back(lb->pDeformRatePlasticLabel);
  from.push_back(lb->pPlasticStrainLabel);

  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);
  to.push_back(lb->pDeformRatePlasticLabel_preReloc);
  to.push_back(lb->pPlasticStrainLabel_preReloc);
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
  ParticleVariable<double> pPlasticStrain;

  new_dw->allocateAndPut(pDeformGrad, lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress, lb->pStressLabel, pset);
  new_dw->allocateAndPut(pDeformRatePlastic, lb->pDeformRatePlasticLabel, pset);
  new_dw->allocateAndPut(pPlasticStrain, lb->pPlasticStrainLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
    pDeformGrad[*iter] = one;
    pStress[*iter] = zero;
    pDeformRatePlastic[*iter] = one;
    pPlasticStrain[*iter] = 0.0;
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
  Matrix3 one;   one.Identity();
  Matrix3 tensorF; // Deformation gradient
  Matrix3 tensorB; // Left Cauchy-Green deformation
  Matrix3 tensorV; // Left Cauchy-Green stretch
  Matrix3 tensorR; // Rotation 
  Matrix3 tensorL; // Velocity gradient
  Matrix3 tensorD; // Rate of deformation
  Matrix3 tensorW; // Spin 
  Matrix3 tensorOmega; // Rate of rotation
  Matrix3 oneMinusOmega;
  Matrix3 oneMinusOmegaInv;
  Matrix3 onePlusOmega;
  Matrix3 tensorV_new; // Updated Cauchy-Green stretch
  Matrix3 tensorR_new; // Updated rotation 
  Matrix3 tensorF_new; // Updated deformation gradient 
  Matrix3 tensorE_new;
  Matrix3 tensorFInv;
  Matrix3 tensorD_new;
  Matrix3 tensorEta_new;
  Matrix3 tensorDp;
  Matrix3 tensorEtap;
  Matrix3 tensorSig; // The Cauchy stress
  Matrix3 trialS;
  Matrix3 tensorSig_new;
  Matrix3 tensorDe_new;
  Matrix3 tensorEtae_new;
  double equivStress = 0.0;
  double plasticStrainRate = 0.0;
  double plasticStrain = 0.0;
  double temperature = 0.0;
  double flowStress = 0.0;
  double bulk = 0.0;
  double shear = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double totalStrainEnergy = 0.0;
  double tolerance = 1.0e-8;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Create array for the particle position
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the deformation gradient (F) for all particles
    constParticleVariable<Matrix3> pDeformGrad;
    old_dw->get(pDeformGrad, lb->pDeformationMeasureLabel, pset);

    // Get the particle location, particle size, particle mass, particle volume
    constParticleVariable<Point> px;
    old_dw->get(px, lb->pXLabel, pset);
    constParticleVariable<Vector> psize;
    if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);
    constParticleVariable<double> pMass;
    old_dw->get(pMass, lb->pMassLabel, pset);
    constParticleVariable<double> pVolume;
    old_dw->get(pVolume, lb->pVolumeLabel, pset);

    // Get the velocity from the grid and particle velocity
    constNCVariable<Vector> gVelocity;
    new_dw->get(gVelocity, lb->gVelocityLabel, dwi, patch, Ghost::AroundCells, NGN);
    constParticleVariable<Vector> pVelocity;
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);

    // Get the particle stress
    constParticleVariable<Matrix3> pStress;
    old_dw->get(pStress, lb->pStressLabel, pset);

    // Get the plastic part of rate of deformation
    constParticleVariable<Matrix3> pDeformRatePlastic;
    old_dw->get(pDeformRatePlastic, lb->pDeformRatePlasticLabel, pset);

    // Get the plastic strain
    constParticleVariable<double> pPlasticStrain;
    old_dw->get(pPlasticStrain, lb->pPlasticStrainLabel, pset);

    // Get the particle temperature
    constParticleVariable<double> pTemperature;
    old_dw->get(pTemperature, lb->pTemperatureLabel, pset);
    
    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    // Create and allocate arrays for storing the updated information
    ParticleVariable<Matrix3> pDeformGrad_new;
    new_dw->allocateAndPut(pDeformGrad_new, 
                           lb->pDeformationMeasureLabel_preReloc, pset);
    ParticleVariable<Matrix3> pStress_new;
    new_dw->allocateAndPut(pStress_new, 
                           lb->pStressLabel_preReloc, pset);
    ParticleVariable<Matrix3> pDeformRatePlastic_new;
    new_dw->allocateAndPut(pDeformRatePlastic_new, 
                           lb->pDeformRatePlasticLabel_preReloc, pset);
    ParticleVariable<double> pPlasticStrain_new;
    new_dw->allocateAndPut(pPlasticStrain_new, 
                           lb->pPlasticStrainLabel_preReloc, pset);
    ParticleVariable<double> pVolume_deformed;
    new_dw->allocateAndPut(pVolume_deformed, 
                           lb->pVolumeDeformedLabel, pset);

    // Loop thru particles
    for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Calculate the polar decomposition of F into the left stretch (V)
      // and the rotation (R)
      tensorF = pDeformGrad[idx]; 
      polarDecomposition(tensorF, tensorB, tensorV, tensorR, tolerance, LEFT_POLAR);
      
      // Calculate the velocity gradient (L) from the grid velocity
      if(d_8or27==27) 
        tensorL = computeVelocityGradient(patch, oodx, 
                                                px[idx], psize[idx], gVelocity);
      else
        tensorL = computeVelocityGradient(patch, oodx, 
                                                px[idx], 0, gVelocity);

      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorW = (tensorL - tensorL.Transpose())*0.5;

      // Calculate rate of rotation tensor (Omega)
      tensorOmega = computeRateofRotation(tensorV, tensorD, tensorW);

      // Update the rotation tensor (R)
      oneMinusOmega = one - tensorOmega*(0.5*delT);
      oneMinusOmegaInv = oneMinusOmega.Inverse();
      onePlusOmega = one + tensorOmega*(0.5*delT);
      tensorR_new = (oneMinusOmegaInv*onePlusOmega)*tensorR;

      // Check the ortogonality of R
      if (!tensorR_new.Orthogonal()) {
         // Do something here that restores orthogonality
      }

      // Update the left Cauchy-Green stretch tensor (V)
      tensorV_new = tensorV + (tensorL*tensorV - tensorV*tensorOmega)*delT;

      // Update the deformation gradient
      tensorF_new = tensorV_new*tensorR_new; 
      pDeformGrad_new[idx] = tensorF_new;

      // Calculate the Green-Lagrange strain tensor (E)
      tensorE_new = ((tensorF_new.Transpose())*tensorF_new - one)*0.5;

      // Calculate the new rate of deformation tensor
      // and Rotate the rate of deformation tensor to the material configuration
      tensorFInv = tensorF_new.Inverse();
      tensorD_new = ((tensorFInv.Transpose())*tensorE_new)*tensorFInv;
      tensorD_new = (tensorR*tensorD_new)*tensorR.Transpose();
      tensorEta_new = tensorD_new - one*(1.0/3.0*tensorD_new.Trace());

      // Get the plastic part of the rate of deformation tensor (Dp)
      // and Rotate to material frame
      tensorDp = pDeformRatePlastic[idx];
      tensorDp = (tensorR*tensorDp)*(tensorR.Transpose());
      tensorEtap = tensorDp - one*(1.0/3.0*tensorDp.Trace());

      // Rotate the particle Cauchy stress to the material configuration
      // and calculate the deviatoric stress
      tensorSig = pStress[idx];
      tensorSig = (tensorR*tensorSig)*tensorR.Transpose();

      // Calculate the trial elastic stress
      bulk  = d_initialData.Bulk;
      shear = d_initialData.Shear;
      trialS = (tensorEta_new - tensorEtap)*2.0*shear;

      // Now that the stress and rate of deformation are available, find if
      // the yield stress has been exceeded
      // (** WARNING ** For now just uses the von Mises J2 yield condition)

      // Calculate the J2 equivalent stress
      equivStress = sqrt(1.5*trialS.NormSquared());

      // Calculate the flow stress
      // from the plastic strain rate, the plastic strain and  
      // the particle temperature
      plasticStrainRate = sqrt(2.0/3.0*tensorEtap.NormSquared());
      plasticStrain = pPlasticStrain[idx];
      temperature = pTemperature[idx];
      flowStress = evaluateFlowStress(plasticStrain, plasticStrainRate,
                                             temperature);

      cout << "Equivalent Stress = " << equivStress 
           << " Flow Stress = " << flowStress << "\n";
      if (flowStress > equivStress) {

        // For the elastic region just do a forward Euler integration
        // of the Hypoelastic constitutive equation to get the updated stress
        tensorDe_new = tensorD_new - tensorDp;
        tensorEtae_new = tensorDe_new - one*(1.0/3.0*tensorDe_new.Trace());
        tensorSig_new = tensorSig + (tensorEtae_new*(2.0*shear) + 
                              one*(bulk*tensorDe_new.Trace()))*delT;

        // Rotate the stress and deformation rate back to the undeformed coordinates
        tensorDp = tensorR.Transpose()*(tensorDp*tensorR.Transpose());
        tensorSig_new = tensorR.Transpose()*(tensorSig_new*tensorR.Transpose());

        // Save the updated data
        pDeformRatePlastic_new[idx] = tensorDp;
        pStress_new[idx] = tensorSig_new;
        pPlasticStrain_new[idx] = plasticStrain;

      } else {

        // Calculate the tensor u
        double scalarSijxSij = trialS.NormSquared();
        Matrix3 tensorU;
        tensorU = trialS*(sqrt(3.0/scalarSijxSij));

        // Do the double contractions of the second order tensors
        double scalarSijxEtaij = 0.0;
        double scalarSijxUij = 0.0;
        for (int ii = 1; ii < 4; ++ii) {
          for (int jj = 1; jj < 4; ++jj) {
	    scalarSijxEtaij += (trialS(ii,jj)*tensorEtap(ii,jj));
	    scalarSijxUij += (trialS(ii,jj)*tensorU(ii,jj));
          }
        }

        // Calculate the flow rate scalar (gammaDot)
        double gammaDot = scalarSijxEtaij/scalarSijxUij;
        double delGamma = gammaDot*delT;
         
        // Update the plastic rate of deformation
        Matrix3 tensorDp_new = tensorDp + tensorU*delGamma;
        
        // Update the plastic strain rate
        double plasticStrainRate_new = sqrt(2.0/3.0*tensorDp_new.NormSquared());

        // Update the plastic strain
        double plasticStrain_new = plasticStrain + plasticStrainRate_new*delT;

        // Update the stress (explicit)
        tensorSig_new = one*bulk*tensorD_new.Trace() + trialS -
				tensorU*2.0*shear*delGamma;

        // Rotate the stress and deformation rate back to the undeformed coordinates
        tensorDp_new = tensorR.Transpose()*(tensorDp_new*tensorR.Transpose());
        tensorSig_new = tensorR.Transpose()*(tensorSig_new*tensorR.Transpose());

        // Save the new data
        pDeformRatePlastic_new[idx] = tensorDp_new;
        pStress_new[idx] = tensorSig_new;
        pPlasticStrain_new[idx] = plasticStrain_new;
      }

      // Compute the deformation gradient increment 
      Matrix3 deformationGradientInc = tensorF_new - tensorF;
      double Jinc = deformationGradientInc.Determinant();

      // get the volumetric part of the deformation
      pVolume_deformed[idx]=Jinc*pVolume[idx];

      // Compute the strain energy for the particles
      double pStrainEnergy = (tensorD_new(1,1)*tensorSig_new(1,1) +
	 tensorD_new(2,2)*tensorSig_new(2,2) + tensorD_new(3,3)*tensorSig_new(3,3) +
	 2.0*(tensorD_new(1,2)*tensorSig_new(1,2) + 
         tensorD_new(1,3)*tensorSig_new(1,3) + tensorD_new(2,3)*tensorSig_new(2,3)))*
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


void JohnsonCook::addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet*) const
{
}

void JohnsonCook::addComputesAndRequires(Task* task,
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
  task->requires(Task::OldDW, lb->pDeformRatePlasticLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pPlasticStrainLabel,     matlset,Ghost::None);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pDeformRatePlasticLabel_preReloc,  matlset);
  task->computes(lb->pPlasticStrainLabel_preReloc,      matlset);
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

  if(d_8or27==8){
    patch->findCellAndShapeDerivatives(px, ni, d_S);
  }
  else if(d_8or27==27){
    patch->findCellAndShapeDerivatives27(px, ni, d_S,psize);
  }

  for(int k = 0; k < d_8or27; k++) {
    const Vector& gvel = gVelocity[ni[k]];
    for (int j = 0; j<3; j++){
      for (int i = 0; i<3; i++) {
	velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
      }
    }
  }
  return velGrad;
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
JohnsonCook::evaluateFlowStress(double& ep, 
                                double& epdot,
                                double& T)
{
  double A  = d_initialData.A;
  double B  = d_initialData.B;
  double C  = d_initialData.C;
  double n  = d_initialData.n;
  double m  = d_initialData.m;
  double Tr  = d_initialData.TRoom;
  double Tm  = d_initialData.TMelt;
  double strainPart = A + B*pow(ep,n);
  double strainRatePart = 1 + C*log(epdot);
  double tempPart = 1 - pow((T-Tr)/(Tm-Tr),m);
  return (strainPart*strainRatePart*tempPart);
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

