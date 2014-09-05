
#include "CompMooneyRivlin.h"
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <values.h>
#include <iostream>

#include <Packages/Uintah/CCA/Components/MPM/Fracture/Connectivity.h>

using std::cerr;

using namespace Uintah;
using namespace SCIRun;

// Material Constants are C1, C2 and PR (poisson's ratio).  
// The shear modulus = 2(C1 + C2).

CompMooneyRivlin::CompMooneyRivlin(ProblemSpecP& ps)
{
  ps->require("he_constant_1",d_initialData.C1);
  ps->require("he_constant_2",d_initialData.C2);
  ps->require("he_PR",d_initialData.PR);
}

CompMooneyRivlin::~CompMooneyRivlin()
{
  // Destructor
}

void CompMooneyRivlin::initializeCMData(const Patch* patch,
					const MPMMaterial* matl,
					DataWarehouseP& new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();
   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
         deformationGradient[*iter] = Identity;
         pstress[*iter] = zero;
   }
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);

   computeStableTimestep(patch, matl, new_dw);
}

void CompMooneyRivlin::computeStableTimestep(const Patch* patch,
					     const MPMMaterial* matl,
					     DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double C1 = d_initialData.C1;
  double C2 = d_initialData.C2;
  double PR = d_initialData.PR;

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed + particle velocity at each particle, 
     // store the maximum
     double mu = 2.*(C1 + C2);
     //double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);
     c_dil = sqrt(2.*mu*(1.- PR)*pvolume[idx]/((1.-2.*PR)*pmass[idx]));
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    if(delT_new < 1.e-12) delT_new = MAXDOUBLE;
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void CompMooneyRivlin::computeStressTensor(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouseP& old_dw,
                                           DataWarehouseP& new_dw)
{
  Matrix3 Identity,deformationGradientInc,B,velGrad;
  double invar1,invar2,invar3,J,w1,w2,w3,i3w3,w1pi1w2;
  Identity.Identity();
  double c_dil = 0.0,se=0.0,Jinc;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  Vector dx = patch->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

  int matlindex = matl->getDWIndex();

  // Create array for the particle position
  ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<Point> px;
  old_dw->get(px, lb->pXLabel, pset);
  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

  // Create array for the particle stress
  ParticleVariable<Matrix3> pstress;
  
  old_dw->get(pstress, lb->pStressLabel, pset);

  // Retrieve the array of constitutive parameters
  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  old_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  old_dw->get(pvelocity, lb->pVelocityLabel, pset);
  
  NCVariable<Vector> gvelocity;

  new_dw->get(gvelocity, lb->gMomExedVelocityLabel, matlindex,patch,
	      Ghost::AroundCells, 1);
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  ParticleVariable<int> pConnectivity;
  ParticleVariable<Vector> pRotationRate;
  ParticleVariable<double> pStrainEnergy;
  if(matl->getFractureModel()) {
    new_dw->get(pConnectivity, lb->pConnectivityLabel, pset);
    new_dw->allocate(pRotationRate, lb->pRotationRateLabel, pset);
    new_dw->allocate(pStrainEnergy, lb->pStrainEnergyLabel, pset);
  }

  double C1 = d_initialData.C1;
  double C2 = d_initialData.C2;
  double C3 = .5*C1 + C2;
  double PR = d_initialData.PR;
  double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     velGrad.set(0.0);
     
     // Get the node indices that surround the cell
     IntVector ni[8];
     Vector d_S[8];

     patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
     
     if(matl->getFractureModel()) {
       //ratation rate: (omega1,omega2,omega3)
       double omega1 = 0;
       double omega2 = 0;
       double omega3 = 0;

       Connectivity connectivity(pConnectivity[idx]);
       int conn[8];
       connectivity.getInfo(conn);
       connectivity.modifyShapeDerivatives(conn,d_S,Connectivity::connect);

       for(int k = 0; k < 8; k++) {
	    Vector& gvel = gvelocity[ni[k]];
	    for (int j = 0; j<3; j++){
	       for (int i = 0; i<3; i++) {
	          velGrad(i+1,j+1) += gvel(i) * d_S[k](j) * oodx[j];		  
	       }
	    }

	    //rotation rate computation, required for fracture
	    //NOTE!!! gvel(0) = gvel.x() !!!
	    omega1 += gvel(2) * d_S[k](1) * oodx[1] - gvel(1) * d_S[k](2) * oodx[2];
	    omega2 += gvel(0) * d_S[k](2) * oodx[2] - gvel(2) * d_S[k](0) * oodx[0];
	    omega3 += gvel(1) * d_S[k](0) * oodx[0] - gvel(0) * d_S[k](1) * oodx[1];
       }
       pRotationRate[idx] = Vector(omega1/2,omega2/2,omega3/2);
     }
     else {
       for(int k = 0; k < 8; k++) {
	  Vector& gvel = gvelocity[ni[k]];
	  for (int j = 0; j<3; j++){
	     for (int i = 0; i<3; i++) {
	        velGrad(i+1,j+1) += gvel(i) * d_S[k](j) * oodx[j];		  
	     }
	  }
        }
     }

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;
      
      Jinc = deformationGradientInc.Determinant();

      // Update particle volumes

      pvolume[idx]=Jinc*pvolume[idx];

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient[idx] = deformationGradientInc * deformationGradient[idx];

      // Actually calculate the stress from the n+1 deformation gradient.

      // Compute the left Cauchy-Green deformation tensor
      B = deformationGradient[idx] * deformationGradient[idx].Transpose();

      // Compute the invariants
      invar1 = B.Trace();
      invar2 = 0.5*((invar1*invar1) - (B*B).Trace());
      J = deformationGradient[idx].Determinant();
      invar3 = J*J;

      w1 = C1;
      w2 = C2;
      w3 = -2.0*C3/(invar3*invar3*invar3) + 2.0*C4*(invar3 -1.0);

      // Compute T = 2/sqrt(I3)*(I3*W3*Identity + (W1+I1*W2)*B - W2*B^2)
      w1pi1w2 = w1 + invar1*w2;
      i3w3 = invar3*w3;

      pstress[idx]=(B*w1pi1w2 - (B*B)*w2 + Identity*i3w3)*2.0/J;
      
      // Compute wave speed + particle velocity at each particle, 
      // store the maximum
      c_dil = sqrt((4.*(C1+C2*invar2)/J
		    +8.*(2.*C3/(invar3*invar3*invar3)+C4*(2.*invar3-1.))
		    -Min((pstress[idx])(1,1),(pstress[idx])(2,2)
			 ,(pstress[idx])(3,3))/J)
		   *pvolume[idx]/pmass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

      // Compute the strain energy for all the particles
      double e = (C1*(invar1-3.0) + C2*(invar2-3.0) +
            C3*(1.0/(invar3*invar3) - 1.0) +
            C4*(invar3-1.0)*(invar3-1.0))*pvolume[idx]/J;

      if(matl->getFractureModel()) pStrainEnergy[idx] = e;
      
      se += e;
    }
        
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    
    if(delT_new < 1.e-12) delT_new = MAXDOUBLE;
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);    
    new_dw->put(pstress, lb->pStressAfterStrainRateLabel);
    new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);

    //
    if( matl->getFractureModel() ) {
      new_dw->put(pRotationRate, lb->pRotationRateLabel);
      new_dw->put(pStrainEnergy, lb->pStrainEnergyLabel);
    }

    new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);

    new_dw->put(pvolume, lb->pVolumeDeformedLabel);
}

void CompMooneyRivlin::addParticleState(std::vector<const VarLabel*>& from,
					std::vector<const VarLabel*>& to)
{
}

void CompMooneyRivlin::addComputesAndRequires(Task* task,
					      const MPMMaterial* matl,
					      const Patch* patch,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw) const
{
   int idx = matl->getDWIndex();

   task->requires(old_dw, lb->pXLabel, idx, patch, Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, idx, patch, Ghost::None);
   task->requires(old_dw, lb->pMassLabel,   idx,  patch, Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, idx,  patch, Ghost::None);
   
   task->requires(old_dw, lb->pStressLabel, idx,  patch, Ghost::None);
   //task->requires(old_dw, lb->pVelocityLabel, idx,  patch, Ghost::None);
   
   task->requires(new_dw, lb->gMomExedVelocityLabel, idx, patch,
						Ghost::AroundCells, 1);
   task->requires(old_dw, lb->delTLabel);

   task->computes(new_dw, lb->pStressAfterStrainRateLabel,       idx, patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc, idx, patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel,              idx, patch);
   
   if(matl->getFractureModel()) {
      task->requires(new_dw, lb->pConnectivityLabel, idx, patch, Ghost::None);
      task->computes(new_dw, lb->pRotationRateLabel, idx, patch);
      task->computes(new_dw, lb->pStrainEnergyLabel, idx, patch);
   }
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {

#if 0
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(CompMooneyRivlin::CMData), sizeof(double)*3);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(CompMooneyRivlin::CMData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other, "CompMooneyRivlin::CMData", true, &makeMPI_CMData);
   }
   return td;   
}
#endif

} // End namespace Uintah
