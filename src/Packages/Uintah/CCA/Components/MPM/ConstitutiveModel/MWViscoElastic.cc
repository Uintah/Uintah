#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MWViscoElastic.h>
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
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h> // just added
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
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

MWViscoElastic::MWViscoElastic(ProblemSpecP& ps, MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;
  
  ps->require("e_shear_modulus",d_initialData.E_Shear);
  ps->require("e_bulk_modulus",d_initialData.E_Bulk);
  ps->require("ve_shear_modulus",d_initialData.VE_Shear);
  ps->require("ve_bulk_modulus",d_initialData.VE_Bulk);
  ps->require("ve_volumetric_viscosity",d_initialData.V_Viscosity);
  ps->require("ve_deviatoric_viscosity",d_initialData.D_Viscosity);
  
  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }
}

MWViscoElastic::~MWViscoElastic()
{
  // Destructor
}

void MWViscoElastic::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();
  
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> deformationGradient;
  ParticleVariable<Matrix3> pstress_e,pstress_ve_d,pstress_e_d;
  ParticleVariable<double> pstress_ve_v,pstress_e_v;
  new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,pset);
  new_dw->allocateAndPut(pstress_e,          lb->pStress_eLabel,          pset);
  new_dw->allocateAndPut(pstress_ve_v,       lb->pStress_ve_vLabel,       pset);
  new_dw->allocateAndPut(pstress_ve_d,       lb->pStress_ve_dLabel,       pset);
  new_dw->allocateAndPut(pstress_e_v,        lb->pStress_e_vLabel,        pset);
  new_dw->allocateAndPut(pstress_e_d,        lb->pStress_e_dLabel,        pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
      deformationGradient[*iter] = Identity;
      pstress_e[*iter] = zero;
      pstress_ve_v[*iter] = 0.0;
      pstress_ve_d[*iter] = zero;
      pstress_e_v[*iter] = 0.0;
      pstress_e_d[*iter] = zero;
  }

  computeStableTimestep(patch, matl, new_dw);
}


void MWViscoElastic::allocateCMData(DataWarehouse* new_dw,
				    ParticleSubset* subset,
				    map<const VarLabel*, ParticleVariableBase*>* newState)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();
  
  ParticleVariable<Matrix3> deformationGradient,pstress_e,pstress_ve_d,
    pstress_e_d;
  ParticleVariable<double> pstress_ve_v,pstress_e_v;
  new_dw->allocateTemporary(deformationGradient,subset);
  new_dw->allocateTemporary(pstress_e,subset);
  new_dw->allocateTemporary(pstress_ve_v,subset);
  new_dw->allocateTemporary(pstress_ve_d,subset);
  new_dw->allocateTemporary(pstress_e_v,subset);
  new_dw->allocateTemporary(pstress_e_d,subset);

  for(ParticleSubset::iterator iter = subset->begin();iter != subset->end();
      iter++){
    deformationGradient[*iter] = Identity;
    pstress_e[*iter] = zero;
    pstress_ve_v[*iter] = 0.0;
    pstress_ve_d[*iter] = zero;
    pstress_e_v[*iter] = 0.0;
    pstress_e_d[*iter] = zero;
  }

  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStress_eLabel]=pstress_e.clone();
  (*newState)[lb->pStress_ve_vLabel]=pstress_ve_v.clone();
  (*newState)[lb->pStress_ve_dLabel]=pstress_ve_d.clone();
  (*newState)[lb->pStress_e_vLabel]=pstress_e_v.clone();
  (*newState)[lb->pStress_e_dLabel]=pstress_e_d.clone();

}

void MWViscoElastic::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
}

void MWViscoElastic::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double e_shear = d_initialData.E_Shear;
  double e_bulk = d_initialData.E_Bulk;
  double ve_shear = d_initialData.VE_Shear;
  double ve_bulk = d_initialData.VE_Bulk;
  double bulk = e_bulk + ve_bulk;
  double shear = e_shear +ve_shear;
  
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*shear/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void MWViscoElastic::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    double se = 0.0, ve = 0.0;
    const Patch* patch = patches->get(p);
    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress_e;
    constParticleVariable<Matrix3> pstress_ve_d, pstress_e_d;
    ParticleVariable<Matrix3> pstress_e_new, pstress_ve_new;
    ParticleVariable<Matrix3> pstress_ve_d_new, pstress_e_d_new;
    ParticleVariable<Matrix3> pstress_new, deformationGradient_new;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    constParticleVariable<double> pstress_ve_v, pstress_e_v;
    ParticleVariable<double> pvolume_deformed, pstress_ve_v_new,pstress_e_v_new;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;

    new_dw->allocateTemporary(pstress_ve_new,                             pset);
    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pstress_e_new,    lb->pStress_eLabel,          pset);
    new_dw->allocateAndPut(pstress_ve_v_new, lb->pStress_ve_vLabel,       pset);
    new_dw->allocateAndPut(pstress_ve_d_new, lb->pStress_ve_dLabel,       pset);
    new_dw->allocateAndPut(pstress_e_v_new,  lb->pStress_e_vLabel,        pset);
    new_dw->allocateAndPut(pstress_e_d_new,  lb->pStress_e_dLabel,        pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,    pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                   lb->pDeformationMeasureLabel_preReloc, pset);
    
    if(d_8or27==27){
      old_dw->get(psize,             lb->pSizeLabel,               pset);
    }
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pstress_e,           lb->pStress_eLabel,           pset);
    old_dw->get(pstress_ve_v,        lb->pStress_ve_vLabel,        pset);
    old_dw->get(pstress_ve_d,        lb->pStress_ve_dLabel,        pset);
    old_dw->get(pstress_e_v,         lb->pStress_e_vLabel,         pset);
    old_dw->get(pstress_e_d,         lb->pStress_e_dLabel,         pset);
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->get(gvelocity, lb->gVelocityLabel, dwi,patch, gac, NGN);

    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    constNCVariable<Vector> Gvelocity;
    new_dw->get(Gvelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
#endif

    double e_shear = d_initialData.E_Shear;
    double e_bulk = d_initialData.E_Bulk;
    double ve_shear = d_initialData.VE_Shear;
    double ve_bulk = d_initialData.VE_Bulk;
    double v_viscosity = d_initialData.V_Viscosity;
    double d_viscosity = d_initialData.D_Viscosity;
    double bulk = e_bulk + ve_bulk;
    double shear = e_shear +ve_shear;
    
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
	    for (int i = 0; i<3; i++) {
	      velGrad(i+1,j+1)+=gvel[i] * d_S[k][j] * oodx[j];
	    }
	 }
      }

      // Calculate rate of deformation D, and deviatoric rate DPrime
    
      Matrix3 D = (velGrad + velGrad.Transpose())*.5;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

// standard solid element:      
      pstress_e_new[idx] =pstress_e[idx] + (DPrime*2.*e_shear + Identity*e_bulk*D.Trace())*delT;

      pstress_ve_v_new[idx] = (-onethird*D.Trace()*v_viscosity + pstress_ve_v[idx]*(v_viscosity/3/ve_bulk/delT))/
                              (1+v_viscosity/3/ve_bulk/delT);
      
      pstress_ve_d_new[idx] = (DPrime*d_viscosity + pstress_ve_d[idx]*(d_viscosity/2/ve_shear/delT))/
                              (1+d_viscosity/2/ve_shear/delT);

      pstress_ve_new[idx] = pstress_ve_d_new[idx]-Identity*pstress_ve_v_new[idx];

      pstress_new[idx] = pstress_e_new[idx] + pstress_ve_new[idx];

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                             deformationGradient[idx];

      // get the volumetric part of the deformation
      // unused variable - Steve
      // double J = deformationGradient[idx].Determinant();

      pvolume_deformed[idx]=Jinc*pvolume[idx];

      // Compute the strain energy for all the particles
      pstress_e_v_new[idx] = pstress_e_v[idx]-D.Trace()*e_bulk*delT;
      
     pstress_e_d_new[idx] = pstress_e_d[idx]+DPrime*2*e_shear*delT;
     
     double p = pstress_e_v_new[idx] + pstress_ve_v_new[idx];
     
     double ee = (pstress_ve_d_new[idx].NormSquared()/4/ve_shear +
		    pstress_e_d_new[idx].NormSquared()/4/e_shear +
		    p*p/2/(ve_bulk + e_bulk))* pvolume_deformed[idx];
		    
           ve = ve +(pstress_ve_d_new[idx].NormSquared()/d_viscosity)
		     * pvolume_deformed[idx]*delT;
   
      double e = ee + ve; 

      se += e;		   

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt((bulk + 4.*shear/3.)*pvolume_deformed[idx]/pmass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new),lb->delTLabel);
    new_dw->put(sum_vartype(se),               lb->StrainEnergyLabel);
  }
}


void MWViscoElastic::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet* ) const
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel,       matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
  if(d_8or27==27){
    task->requires(Task::OldDW, lb->pSizeLabel,            matlset,Ghost::None);
  }
  task->requires(Task::NewDW, lb->gVelocityLabel,          matlset,gac, NGN);
#ifdef FRACTURE
  task->requires(Task::NewDW, lb->pgCodeLabel,            matlset,Ghost::None);
  task->requires(Task::NewDW, lb->GVelocityLabel,         matlset, gac, NGN);
#endif

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);
}

void 
MWViscoElastic::addComputesAndRequires(Task* ,
				       const MPMMaterial* ,
				       const PatchSet* ,
				       const bool ) const
{
}

double MWViscoElastic::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  //double p_ref=101325.0;
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double e_bulk = d_initialData.E_Bulk;
  double ve_bulk = d_initialData.VE_Bulk;
  double bulk = e_bulk + ve_bulk;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 0
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR MWViscoElastic"
       << endl;
#endif
}

void MWViscoElastic::computePressEOSCM(const double rho_cur, double& pressure,
				       const double p_ref,
				       double& dp_drho,      double& tmp,
				       const MPMMaterial* matl)
{

  double e_shear = d_initialData.E_Shear;
  double e_bulk = d_initialData.E_Bulk;
  double ve_shear = d_initialData.VE_Shear;
  double ve_bulk = d_initialData.VE_Bulk;
  double bulk = e_bulk + ve_bulk;
  double shear = e_shear +ve_shear;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = sqrt((bulk + 4.*shear/3.)/rho_cur);  // speed of sound squared

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR MWViscoElastic"
       << endl;
#endif
}

double MWViscoElastic::getCompressibility()
{
  return 1.0/(d_initialData.E_Bulk+d_initialData.VE_Bulk);
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

#if 0
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(MWViscoElastic::StateData), sizeof(double)*2);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 2, 2, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(MWViscoElastic::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew
	TypeDescription(TypeDescription::Other,
			"MWViscoElastic::StateData", true, &makeMPI_CMData);
   }
   return td;
}
#endif

} // End namespace Uintah
