#include "ConstitutiveModelFactory.h"
#include "HypoElastic.h"
#include <Core/Malloc/Allocator.h>
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
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <fstream>
#include <iostream>

#include <Packages/Uintah/CCA/Components/MPM/Fracture/Connectivity.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

HypoElastic::HypoElastic(ProblemSpecP& ps, MPMLabel* Mlb)
{
  lb = Mlb;

  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);
  d_se=0;
}

HypoElastic::~HypoElastic()
{
  // Destructor

}

void HypoElastic::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();
   d_se=0;

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

void HypoElastic::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
}

void HypoElastic::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<double> pmass, pvolume;
  ParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double G = d_initialData.G;
  double bulk = d_initialData.K;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void HypoElastic::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    //
    //  FIX  To do:  Read in table for vres
    //               Obtain and modify particle temperature (deg K)
    //
    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int matlindex = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    ParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<double> pmass, pvolume, ptemperature;
    ParticleVariable<Vector> pvelocity;
    NCVariable<Vector> gvelocity;
    delt_vartype delT;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);

    new_dw->get(gvelocity,           lb->gVelocityLabel, matlindex,patch,
		Ghost::AroundCells, 1);

    old_dw->get(delT, lb->delTLabel);

    ParticleVariable<int> pConnectivity;
    ParticleVariable<Vector> pRotationRate;
    ParticleVariable<double> pStrainEnergy;
    if(matl->getFractureModel()) {
      new_dw->get(pConnectivity, lb->pConnectivityLabel, pset);
      new_dw->allocate(pRotationRate, lb->pRotationRateLabel, pset);
      new_dw->allocate(pStrainEnergy, lb->pStrainEnergyLabel, pset);
    }

    double G = d_initialData.G;
    double bulk = d_initialData.K;

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
	  if( conn[k] ) {
	    const Vector& gvel = gvelocity[ni[k]];
	    for (int j = 0; j<3; j++){
	      for (int i = 0; i<3; i++) {
	        velGrad(i+1,j+1) += gvel(i) * d_S[k](j) * oodx[j];
              }
	    }
	    //rotation rate computation, required for fracture
	    //NOTE!!! gvel(0) = gvel.x() !!!
	    omega1 += -gvel(2) * d_S[k](1) * oodx[1] +
	              gvel(1) * d_S[k](2) * oodx[2];
	    omega2 += -gvel(0) * d_S[k](2) * oodx[2] +
	              gvel(2) * d_S[k](0) * oodx[0];
            omega3 += -gvel(1) * d_S[k](0) * oodx[0] +
	              gvel(0) * d_S[k](1) * oodx[1];
	  }
	}
	pRotationRate[idx] = Vector(omega1/2,omega2/2,omega3/2);
      }
      else {
        for(int k = 0; k < 8; k++) {
	  Vector& gvel = gvelocity[ni[k]];
	  for (int j = 0; j<3; j++){
	    for (int i = 0; i<3; i++) {
	      velGrad(i+1,j+1)+=gvel(i) * d_S[k](j) * oodx[j];
	    }
	  }
	}
      }

      // Calculate rate of deformation D, and deviatoric rate DPrime
    
      Matrix3 D = (velGrad + velGrad.Transpose())*.5;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

      // This is the (updated) Cauchy stress

      Matrix3 OldStress = pstress[idx];
      pstress[idx] += (DPrime*2.*G + Identity*bulk*D.Trace())*delT;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient[idx] = deformationGradientInc *
                             deformationGradient[idx];

      // get the volumetric part of the deformation
      // unused variable - Steve
      // double J = deformationGradient[idx].Determinant();

      pvolume[idx]=Jinc*pvolume[idx];

      // Compute the strain energy for all the particles
      OldStress = (pstress[idx] + OldStress)*.5;

      double e = (D(1,1)*OldStress(1,1) +
	          D(2,2)*OldStress(2,2) +
	          D(3,3)*OldStress(3,3) +
	       2.*(D(1,2)*OldStress(1,2) +
		   D(1,3)*OldStress(1,3) +
		   D(2,3)*OldStress(2,3))) * pvolume[idx]*delT;

      if(matl->getFractureModel()) pStrainEnergy[idx] = e;
      		   
      d_se += e;		   

      // Compute wave speed at each particle, store the maximum

      if(pmass[idx] > 0){
        c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
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
    new_dw->put(delt_vartype(delT_new),lb->delTLabel);
    new_dw->put(pstress,               lb->pStressLabel_afterStrainRate);
    new_dw->put(deformationGradient,   lb->pDeformationMeasureLabel_preReloc);
    new_dw->put(sum_vartype(d_se),     lb->StrainEnergyLabel);
    new_dw->put(pvolume,               lb->pVolumeDeformedLabel);

    if( matl->getFractureModel() ) {
      new_dw->put(pRotationRate, lb->pRotationRateLabel);
      new_dw->put(pStrainEnergy, lb->pStrainEnergyLabel);
    }
  }
}

void HypoElastic::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,           matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,        matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pStressLabel,      matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,      matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
  task->requires(Task::NewDW, lb->gVelocityLabel,    matlset,
                  Ghost::AroundCells, 1);

  task->computes(lb->pStressLabel_afterStrainRate,      matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

  if(matl->getFractureModel()) {
    task->requires(Task::NewDW, lb->pConnectivityLabel,  matlset,Ghost::None);
    task->computes(lb->pRotationRateLabel, matlset);
    task->computes(lb->pStrainEnergyLabel, matlset);
  }
}

double HypoElastic::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  //double p_ref=101325.0;
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double G = d_initialData.G;
  double bulk = d_initialData.K;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 0
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

void HypoElastic::computePressEOSCM(const double rho_cur, double& pressure,
                                    const double p_ref,
                                    double& dp_drho,      double& tmp,
                                    const MPMMaterial* matl)
{

//  double p_ref=101325.0;
  double G = d_initialData.G;
  double bulk = d_initialData.K;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = (bulk + 4.*G/3.)/rho_cur;  // speed of sound squared

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {

#if 0
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(HypoElastic::StateData), sizeof(double)*2);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 2, 2, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(HypoElastic::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew
	TypeDescription(TypeDescription::Other,
			"HypoElastic::StateData", true, &makeMPI_CMData);
   }
   return td;
}
#endif

} // End namespace Uintah
