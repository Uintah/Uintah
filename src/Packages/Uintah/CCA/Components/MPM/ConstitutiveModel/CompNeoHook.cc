#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHook.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

CompNeoHook::CompNeoHook(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;
  d_useModifiedEOS = false;
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->get("useModifiedEOS",d_useModifiedEOS); 

  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

}

CompNeoHook::~CompNeoHook()
{
}

void CompNeoHook::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<Matrix3> deformationGradient, pstress;

   new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,
			  pset);
   new_dw->allocateAndPut(pstress,lb->pStressLabel,pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          pstress[*iter] = zero;
   }

   computeStableTimestep(patch, matl, new_dw);
}

void CompNeoHook::allocateCMData(DataWarehouse* new_dw,
				 ParticleSubset* subset,
				 map<const VarLabel*, ParticleVariableBase*>* newState)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleVariable<Matrix3> deformationGradient, pstress;

   new_dw->allocateTemporary(deformationGradient,subset);
   new_dw->allocateTemporary(pstress,subset);


   for(ParticleSubset::iterator iter = subset->begin();
          iter != subset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          pstress[*iter] = zero;
   }

   (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
   (*newState)[lb->pStressLabel]=pstress.clone();

}

void CompNeoHook::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
}

void CompNeoHook::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel, pset);
  new_dw->get(pvolume,   lb->pVolumeLabel, pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void CompNeoHook::computeStressTensor(const PatchSubset* patches,
				      const MPMMaterial* matl,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,Shear,bElBar_new,deformationGradientInc;
    double J,p,IEl,U,W,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);
    Matrix3 Identity;

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,pvolume;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity;
    constNCVariable<Vector> gvelocity;
    constParticleVariable<Vector> psize;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    
    if(d_8or27==27){
      old_dw->get(psize,             lb->pSizeLabel,              pset);
    }
    new_dw->allocateAndPut(pstress,        lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(deformationGradient_new,
		     lb->pDeformationMeasureLabel_preReloc, pset);

    new_dw->get(gvelocity, lb->gVelocityLabel,dwi,patch,gac,NGN);
    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    constNCVariable<Vector> Gvelocity; 
    new_dw->get(Gvelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
#endif

    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
      particleIndex idx = *iter;
      
      // Get the node indices that surround the cell
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      
      if(d_8or27==8){
	patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
      }
      else if(d_8or27==27){
	patch->findCellAndShapeDerivatives27(px[idx], ni, d_S,psize[idx]);
      }
      
      Vector gvel;
      velGrad.set(0.0);
      for(int k = 0; k < d_8or27; k++) {
#ifdef FRACTURE
	 if(pgCode[idx][k]==1) gvel = gvelocity[ni[k]];
	 if(pgCode[idx][k]==2) gvel = Gvelocity[ni[k]]; 
#else
	 gvel = gvelocity[ni[k]];
#endif
	 for (int j = 0; j<3; j++){
	    double d_SXoodx = d_S[k][j] * oodx[j];
	    for (int i = 0; i<3; i++) {
	       velGrad(i+1,j+1) += gvel[i] * d_SXoodx;
	    }
	 }
      }
      
      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
				     deformationGradient[idx];

      // get the volumetric part of the deformation
      J    = deformationGradient_new[idx].Determinant();

      bElBar_new = deformationGradient_new[idx]
		 * deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));

      IEl = onethird*bElBar_new.Trace();

      // Shear is equal to the shear modulus times dev(bElBar)
      Shear = (bElBar_new - Identity*IEl)*shear;

      // get the hydrostatic part of the stress
      p = 0.5*bulk*(J - 1.0/J);

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*p + Shear/J;

      // Compute the strain energy for all the particles
      U = .5*bulk*(.5*(J*J - 1.0) - log(J));
      W = .5*shear*(bElBar_new.Trace() - 3.0);

      pvolume_deformed[idx]=(pmass[idx]/rho_orig)*J;
      
      double e = (U + W)*pvolume_deformed[idx]/J;

      se += e;

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt((bulk + 4.*shear/3.)*pvolume_deformed[idx]/pmass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
  		       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);

  }
}

void 
CompNeoHook::computeStressTensor(const PatchSubset* ,
				const MPMMaterial* ,
				DataWarehouse* ,
				DataWarehouse* ,
				Solver* ,
				const bool )
{
}

void CompNeoHook::carryForward(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleVariable<Matrix3> pdefm_new,pstress_new;
    constParticleVariable<Matrix3> pdefm;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume_deformed;
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pdefm,         lb->pDeformationMeasureLabel,           pset);
    old_dw->get(pmass,                 lb->pMassLabel,                 pset);
    new_dw->allocateAndPut(pdefm_new,lb->pDeformationMeasureLabel_preReloc,
                                                                       pset);
    new_dw->allocateAndPut(pstress_new,lb->pStressLabel_preReloc,      pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel, pset);
    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pdefm_new[idx] = pdefm[idx];
      pstress_new[idx] = Matrix3(0.0);
      pvolume_deformed[idx]=(pmass[idx]/rho_orig);
    }
    new_dw->put(delt_vartype(1.e10), lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void CompNeoHook::addInitialComputesAndRequires(Task*,
                                                const MPMMaterial*,
                                                const PatchSet*) const
{
}

void CompNeoHook::addComputesAndRequires(Task* task,
					  const MPMMaterial* matl,
					  const PatchSet*) const
{
    const MaterialSubset* matlset = matl->thisMaterial();
    Ghost::GhostType  gac   = Ghost::AroundCells;
    task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
    task->requires(Task::OldDW, lb->pMassLabel,   matlset, Ghost::None);
    task->requires(Task::OldDW, lb->pVelocityLabel, matlset, Ghost::None);
    task->requires(Task::OldDW, lb->pDeformationMeasureLabel,
						  matlset, Ghost::None);
    if(d_8or27==27){
      task->requires(Task::OldDW,lb->pSizeLabel,     matlset, Ghost::None);
    }
    task->requires(Task::NewDW,lb->gVelocityLabel,matlset, gac, NGN);

    task->requires(Task::OldDW, lb->delTLabel);

#ifdef FRACTURE
    task->requires(Task::NewDW,  lb->pgCodeLabel,    matlset, Ghost::None);
    task->requires(Task::NewDW,  lb->GVelocityLabel, matlset, gac, NGN);
#endif

    task->computes(lb->pStressLabel_preReloc,             matlset);
    task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
    task->computes(lb->pVolumeDeformedLabel,              matlset);


}

void 
CompNeoHook::addComputesAndRequires(Task* ,
				   const MPMMaterial* ,
				   const PatchSet* ,
				   const bool ) const
{
}


// The "CM" versions use the pressure-volume relationship of the CNH model
double CompNeoHook::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  }
  return rho_cur;
}

void CompNeoHook::computePressEOSCM(const double rho_cur,double& pressure, 
				    const double p_ref,
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

double CompNeoHook::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(CompNeoHook::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(CompNeoHook::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "CompNeoHook::StateData", true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
