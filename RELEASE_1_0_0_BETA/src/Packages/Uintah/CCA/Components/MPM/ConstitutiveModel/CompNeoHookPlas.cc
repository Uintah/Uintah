
#include "CompNeoHookPlas.h"
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

#include <Packages/Uintah/CCA/Components/MPM/Fracture/Connectivity.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

CompNeoHookPlas::CompNeoHookPlas(ProblemSpecP& ps)
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->require("yield_stress",d_initialData.FlowStress);
  ps->require("hardening_modulus",d_initialData.K);
  ps->require("alpha",d_initialData.Alpha);

  p_statedata_label = scinew VarLabel("p.statedata_cnhp",
                             ParticleVariable<StateData>::getTypeDescription());
  p_statedata_label_preReloc = scinew VarLabel("p.statedata_cnhp+",
                             ParticleVariable<StateData>::getTypeDescription());
 
  bElBarLabel = scinew VarLabel("p.bElBar",
		ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc = scinew VarLabel("p.bElBar+",
		ParticleVariable<Matrix3>::getTypeDescription());
}

void CompNeoHookPlas::addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to)
{
   from.push_back(p_statedata_label);
   from.push_back(bElBarLabel);
   to.push_back(p_statedata_label_preReloc);
   to.push_back(bElBarLabel_preReloc);
}

CompNeoHookPlas::~CompNeoHookPlas()
{
  // Destructor 
  delete p_statedata_label;
  delete p_statedata_label_preReloc;
  delete bElBarLabel;
  delete bElBarLabel_preReloc;
  
}

void CompNeoHookPlas::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<StateData> statedata;
   new_dw->allocate(statedata, p_statedata_label, pset);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);
   ParticleVariable<Matrix3> bElBar;
   new_dw->allocate(bElBar,  bElBarLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
	  statedata[*iter].Alpha = d_initialData.Alpha;
          deformationGradient[*iter] = Identity;
          bElBar[*iter] = Identity;
          pstress[*iter] = zero;
   }
   new_dw->put(statedata, p_statedata_label);
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);
   new_dw->put(bElBar, bElBarLabel);

   computeStableTimestep(patch, matl, new_dw);

}

void CompNeoHookPlas::computeStableTimestep(const Patch* patch,
					     const MPMMaterial* matl,
					     DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<StateData> statedata;
  new_dw->get(statedata, p_statedata_label, pset);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     if(pmass[idx] > 0){
       c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
     }
     else{
       c_dil = 0.0;
       pvelocity[idx] = Vector(0.0,0.0,0.0);
     }
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void CompNeoHookPlas::computeStressTensor(const Patch* patch,
					  const MPMMaterial* matl,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{

  Matrix3 bElBarTrial,deformationGradientInc;
  Matrix3 shearTrial,Shear,normal;
  Matrix3 fbar,velGrad;
  double J,p,fTrial,IEl,muBar,delgamma,sTnorm,Jinc;
  double onethird = (1.0/3.0);
  double sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity;
  double c_dil = 0.0,se = 0.;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double U,W;

  Identity.Identity();

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
  ParticleVariable<Matrix3> bElBar;
  old_dw->get(bElBar, bElBarLabel, pset);

  // Create array for the particle stress
  ParticleVariable<Matrix3> pstress;
  new_dw->allocate(pstress, lb->pStressLabel, pset);

  // Retrieve the array of constitutive parameters
  ParticleVariable<StateData> statedata;
  old_dw->get(statedata, p_statedata_label, pset);
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

  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double flow  = d_initialData.FlowStress;
  double K     = d_initialData.K;

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
	 if(conn[k] == Connectivity::connect) {
	    Vector& gvel = gvelocity[ni[k]];
	    for (int j = 0; j<3; j++){
	       for (int i = 0; i<3; i++) {
	          velGrad(i+1,j+1) += gvel(i) * d_S[k](j) * oodx[j];		  
	       }
	    }

	    //rotation rate computation, required for fracture
            if(matl->getFractureModel()) {
	      //NOTE!!! gvel(0) = gvel.x() !!!
	      omega1 += gvel(2) * d_S[k](1) * oodx[1] - gvel(1) * d_S[k](2) * oodx[2];
	      omega2 += gvel(0) * d_S[k](2) * oodx[2] - gvel(2) * d_S[k](0) * oodx[0];
	      omega3 += gvel(1) * d_S[k](0) * oodx[0] - gvel(0) * d_S[k](1) * oodx[1];
	    }
	 }
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

    // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
    // time step and the velocity gradient and the material constants
    double alpha = statedata[idx].Alpha;

    // Compute the deformation gradient increment using the time_step
    // velocity gradient
    // F_n^np1 = dudx * dt + Identity
    deformationGradientInc = velGrad * delT + Identity;

    Jinc = deformationGradientInc.Determinant();

    // Update the deformation gradient tensor to its time n+1 value.
    deformationGradient[idx] = deformationGradientInc *
                             deformationGradient[idx];

    // get the volume preserving part of the deformation gradient increment
    fbar = deformationGradientInc * pow(Jinc,-onethird);

    // predict the elastic part of the volume preserving part of the left
    // Cauchy-Green deformation tensor
    bElBarTrial = fbar*bElBar[idx]*fbar.Transpose();
    IEl = onethird*bElBarTrial.Trace();

    // shearTrial is equal to the shear modulus times dev(bElBar)
    shearTrial = (bElBarTrial - Identity*IEl)*shear;

    // get the volumetric part of the deformation
    J = deformationGradient[idx].Determinant();

    // get the hydrostatic part of the stress
    p = 0.5*bulk*(J - 1.0/J);

    // Compute ||shearTrial||
    sTnorm = shearTrial.Norm();

    muBar = IEl * shear;

    // Check for plastic loading
    fTrial = sTnorm - sqtwthds*(K*alpha + flow);

    if(fTrial > 0.0){
	// plastic

	delgamma = (fTrial/(2.0*muBar)) / (1.0 + (K/(3.0*muBar)));

	normal = shearTrial/sTnorm;

        // The actual elastic shear stress
	Shear = shearTrial - normal*2.0*muBar*delgamma;

	// Deal with history variables
	statedata[idx].Alpha = alpha + sqtwthds*delgamma;
	bElBar[idx] = Shear/shear + Identity*IEl;
    }
    else {
	// not plastic

	bElBar[idx] = bElBarTrial;
	Shear = shearTrial;
    }

    // compute the total stress (volumetric + deviatoric)
    pstress[idx] = Identity*p + Shear/J;

    // Compute the strain energy for all the particles
    U = .5*bulk*(.5*(pow(J,2.0) - 1.0) - log(J));
    W = .5*shear*(bElBar[idx].Trace() - 3.0);

    pvolume[idx]=Jinc*pvolume[idx];

    se += (U + W)*pvolume[idx]/J;

    // Compute wave speed at each particle, store the maximum

    if(pmass[idx] > 0){
      c_dil = sqrt((bulk + 4.*shear/3.)*pvolume[idx]/pmass[idx]);
    }
    else{
      c_dil = 0.0;
      pvelocity[idx] = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
  new_dw->put(pstress, lb->pStressAfterStrainRateLabel);
  new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);
  new_dw->put(bElBar, bElBarLabel_preReloc);

  //
  if( matl->getFractureModel() ) {
    new_dw->put(pRotationRate, lb->pRotationRateLabel);
  }


  // Put the strain energy in the data warehouse
  new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);

  // This is just carried forward with the updated alpha
  new_dw->put(statedata, p_statedata_label_preReloc);
  // Store the deformed volume
  new_dw->put(pvolume,lb->pVolumeDeformedLabel);
}

void CompNeoHookPlas::addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const
{
   int matlindex = matl->getDWIndex();
   task->requires(old_dw, lb->pXLabel, matlindex,        patch, Ghost::None);
   task->requires(old_dw, p_statedata_label, matlindex,  patch, Ghost::None);
   task->requires(old_dw, lb->pMassLabel, matlindex,     patch, Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, matlindex,   patch, Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, matlindex,
		  patch, Ghost::None);
   task->requires(new_dw, lb->gMomExedVelocityLabel, matlindex, patch,
                  Ghost::AroundCells, 1);
   task->requires(old_dw, bElBarLabel, matlindex,        patch, Ghost::None);
   task->requires(old_dw, lb->delTLabel);
   task->computes(new_dw, lb->pStressAfterStrainRateLabel, matlindex, patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc,
						      matlindex, patch);
   task->computes(new_dw, bElBarLabel_preReloc,       matlindex, patch);
   task->computes(new_dw, p_statedata_label_preReloc, matlindex, patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel,   matlindex, patch);

   if(matl->getFractureModel()) {
     task->requires(new_dw, lb->pConnectivityLabel, matlindex,patch, Ghost::None);
     task->computes(new_dw, lb->pRotationRateLabel, matlindex, patch);
   }
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {

static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(CompNeoHookPlas::StateData), sizeof(double)*1);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 1, 1, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(CompNeoHookPlas::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
			  "CompNeoHookPlas::StateData", true, &makeMPI_CMData);
   }
   return td;   
}
} // End namespace Uintah
