#include "ConstitutiveModelFactory.h"
#include "HypoElastic.h"
#include <Core/Malloc/Allocator.h>
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
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

HypoElastic::HypoElastic(ProblemSpecP& ps)
{
  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);
  d_se=0;

  p_statedata_label = scinew VarLabel("p.statedata_hypo",
                                ParticleVariable<StateData>::getTypeDescription());
  p_statedata_label_preReloc = scinew VarLabel("p.statedata_hypo+",
                                ParticleVariable<StateData>::getTypeDescription());
}

HypoElastic::~HypoElastic()
{
  // Destructor

  delete p_statedata_label;
  delete p_statedata_label_preReloc;
 
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

   ParticleVariable<StateData> statedata;
   new_dw->allocate(statedata, p_statedata_label, pset);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {

      deformationGradient[*iter] = Identity;
      pstress[*iter] = zero;
   }
   new_dw->put(statedata, p_statedata_label);
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);

   computeStableTimestep(patch, matl, new_dw);

}

void HypoElastic::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(p_statedata_label);
   to.push_back(p_statedata_label_preReloc);
}

void HypoElastic::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
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

  double G = d_initialData.G;
  double bulk = d_initialData.K;
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
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
    old_dw->get(px, lb->pXLabel, pset);

    // Create array for the particle deformation
    ParticleVariable<Matrix3> deformationGradient;
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    // Create array for the particle stress
    ParticleVariable<Matrix3> pstress;
    old_dw->get(pstress, lb->pStressLabel, pset);

    // Retrieve the array of constitutive parameters
    ParticleVariable<StateData> statedata;
    old_dw->get(statedata, p_statedata_label, pset);
    ParticleVariable<double> pmass;
    old_dw->get(pmass, lb->pMassLabel, pset);
    ParticleVariable<double> pvolume;
    old_dw->get(pvolume, lb->pVolumeLabel, pset);
    ParticleVariable<Vector> pvelocity;
    old_dw->get(pvelocity, lb->pVelocityLabel, pset);
    ParticleVariable<double> ptemperature;
    old_dw->get(ptemperature, lb->pTemperatureLabel, pset);

    NCVariable<Vector> gvelocity;

    new_dw->get(gvelocity, lb->gMomExedVelocityLabel, matlindex,patch,
		Ghost::AroundCells, 1);
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    double G = d_initialData.G;
    double bulk = d_initialData.K;
    // Unused variable - Steve
    // double Cp0 = matl->getSpecificHeat();
    //  cout << "Cp0 = " << Cp0 << endl;

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
      particleIndex idx = *iter;

      velGrad.set(0.0);
      // Get the node indices that surround the cell
      IntVector ni[8];
      Vector d_S[8];
      patch->findCellAndShapeDerivatives(px[idx], ni, d_S);

      for(int k = 0; k < 8; k++) {
	Vector& gvel = gvelocity[ni[k]];
	for (int j = 0; j<3; j++){
	  for (int i = 0; i<3; i++) {
	    velGrad(i+1,j+1)+=gvel(i) * d_S[k](j) * oodx[j];
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
      d_se += (D(1,1)*OldStress(1,1) +
	       D(2,2)*OldStress(2,2) +
	       D(3,3)*OldStress(3,3) +
	       2.*(D(1,2)*OldStress(1,2) +
		   D(1,3)*OldStress(1,3) +
		   D(2,3)*OldStress(2,3))
	       )*pvolume[idx]*delT;

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
    new_dw->put(pstress, lb->pStressAfterStrainRateLabel);
    new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);

    // Put the strain energy in the data warehouse
    new_dw->put(sum_vartype(d_se), lb->StrainEnergyLabel);

    // This is updated
    new_dw->put(statedata, p_statedata_label_preReloc);
    // Store deformed volume
    new_dw->put(pvolume,lb->pVolumeDeformedLabel);
  }
}

void HypoElastic::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet* patches) const
{
#if 0
   task->requires(old_dw, lb->pXLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, p_statedata_label, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pTemperatureLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(new_dw, lb->gMomExedVelocityLabel, matl->getDWIndex(), patch,
                  Ghost::AroundCells, 1);
   task->requires(old_dw, lb->delTLabel);

   task->computes(new_dw, lb->pStressAfterStrainRateLabel, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc, matl->getDWIndex(), patch);
   task->computes(new_dw, p_statedata_label_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(), patch);
#else
   NOT_FINISHED("new task stuff");
#endif
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {

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

} // End namespace Uintah


