#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElastic.h>
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
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h> 
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

HypoElastic::HypoElastic(ProblemSpecP& ps, MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);
  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }
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

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> deformationGradient;
  new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,pset);
  ParticleVariable<Matrix3> pstress;
  new_dw->allocateAndPut(pstress, lb->pStressLabel, pset);
  // for J-Integral
#ifdef FRACTURE
  ParticleVariable<Matrix3> pdispGrads;
  new_dw->allocateAndPut(pdispGrads, lb->pDispGradsLabel, pset);
  ParticleVariable<double>  pstrainEnergyDensity;
  new_dw->allocateAndPut(pstrainEnergyDensity, lb->pStrainEnergyDensityLabel, pset);
#endif
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     deformationGradient[*iter] = Identity;
     pstress[*iter] = zero;
#ifdef FRACTURE
     pdispGrads[*iter] = zero;
     pstrainEnergyDensity[*iter] = 0.0;
#endif
  }

  computeStableTimestep(patch, matl, new_dw);
}

void HypoElastic::allocateCMData(DataWarehouse* new_dw,
				 ParticleSubset* subset,
				 map<const VarLabel*, ParticleVariableBase*>* newState)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();

  ParticleVariable<Matrix3> deformationGradient,pstress;
  new_dw->allocateTemporary(deformationGradient,subset);
  new_dw->allocateTemporary(pstress,subset);
  // for J-Integral

#ifdef FRACTURE
  ParticleVariable<Matrix3> pdispGrads;
  new_dw->allocateTemporary(pdispGrads, subset);
  ParticleVariable<double>  pstrainEnergyDensity;
  new_dw->allocateTemporary(pstrainEnergyDensity, , subset);
#endif
  for(ParticleSubset::iterator iter = subset->begin();iter != subset->end();
      iter++){
     deformationGradient[*iter] = Identity;
     pstress[*iter] = zero;
#ifdef FRACTURE
     pdispGrads[*iter] = zero;
     pstrainEnergyDensity[*iter] = 0.0;
#endif
  }

  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();

#ifdef FRACTURE
  (*newState)[lb->lb->pDispGradsLabel]=pdispGrads.clone();
  (*newState)[lb->pStrainEnergyDensityLabel]=pstrainEnergyDensity.clone();
#endif


}




void HypoElastic::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
#ifdef FRACTURE
   from.push_back(lb->pDispGradsLabel);
   from.push_back(lb->pStrainEnergyDensityLabel);
   to.push_back(lb->pDispGradsLabel_preReloc);
   to.push_back(lb->pStrainEnergyDensityLabel_preReloc);
#endif
}

void HypoElastic::computeStableTimestep(const Patch* patch,
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
    double se = 0.0;
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

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;

    if(d_8or27==27){
      old_dw->get(psize,             lb->pSizeLabel,               pset);
    }
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->get(gvelocity,lb->gVelocityLabel, dwi,patch, gac, NGN);

    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constNCVariable<Vector> Gvelocity;
    new_dw->get(Gvelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);

    constParticleVariable<Short27> pgCode;
    constParticleVariable<Matrix3> pdispGrads;
    constParticleVariable<double>  pstrainEnergyDensity;
    new_dw->get(pgCode,              lb->pgCodeLabel,              pset);
    old_dw->get(pdispGrads,          lb->pDispGradsLabel,          pset);
    old_dw->get(pstrainEnergyDensity,lb->pStrainEnergyDensityLabel,pset);

    ParticleVariable<Matrix3> pdispGrads_new;
    ParticleVariable<double> pstrainEnergyDensity_new;
    new_dw->allocateAndPut(pdispGrads_new, lb->pDispGradsLabel_preReloc, pset);
    new_dw->allocateAndPut(pstrainEnergyDensity_new,
                                 lb->pStrainEnergyDensityLabel_preReloc, pset);
#endif

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(deformationGradient_new,
			   lb->pDeformationMeasureLabel_preReloc, pset);
 
    double G    = d_initialData.G;
    double bulk = d_initialData.K;

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

      // This is the (updated) Cauchy stress

      pstress_new[idx] = pstress[idx] + 
                                   (DPrime*2.*G + Identity*bulk*D.Trace())*delT;

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
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(1,1)*AvgStress(1,1) +
	          D(2,2)*AvgStress(2,2) +
	          D(3,3)*AvgStress(3,3) +
	       2.*(D(1,2)*AvgStress(1,2) +
		   D(1,3)*AvgStress(1,3) +
		   D(2,3)*AvgStress(2,3))) * pvolume_deformed[idx]*delT;

      se += e;

#ifdef FRACTURE
      // Update particle displacement gradients
      pdispGrads_new[idx] = pdispGrads[idx] + velGrad * delT;
      // Update particle strain energy density 
      pstrainEnergyDensity_new[idx] = pstrainEnergyDensity[idx] + 
                                         e/pvolume_deformed[idx];
#endif

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt((bulk + 4.*G/3.)*pvolume_deformed[idx]/pmass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new),lb->delTLabel);
    new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
  }
}

void HypoElastic::carryForward(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleVariable<Matrix3> pdefm_new,pstress_new;
    constParticleVariable<Matrix3> pdefm,pstress;
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pdefm,         lb->pDeformationMeasureLabel,           pset);
    new_dw->allocateAndPut(pdefm_new,lb->pDeformationMeasureLabel_preReloc,
                                                                       pset);
    old_dw->get(pstress,       lb->pStressLabel,                       pset);
    new_dw->allocateAndPut(pstress_new,lb->pStressLabel_preReloc,      pset);
    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pdefm_new[idx] = pdefm[idx];
      pstress_new[idx] = pstress[idx];
    }
    new_dw->put(delt_vartype(1.e10), lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

// Convert J-integral into stress intensity factors for hypoelastic materials
void 
HypoElastic::ConvertJToK(const MPMMaterial* matl,const Vector& J,
                     const Vector& C,const Vector& V,Vector& SIF)
{                    
  /* J--J integral, C--Crack velocity, V--COD near crack tip
     in local coordinates. */ 
     
  double J1,C1,C2,CC,V1,V2;
  
  J1=J.x();                           // total energy release rate
  V1=V.y();  V2=V.x();                // V1--opening COD, V2--sliding COD
  C1=C.x();  C2=C.y();                
  CC=C1*C1+C2*C2;                     // square of crack propagating velocity
  
  // get material properties
  double rho_orig,G,K,v,k;
  rho_orig=matl->getInitialDensity();
  G=d_initialData.G;                  // shear modulus
  K=d_initialData.K;                  // bulk modulus
  v=0.5*(3.*K-2.*G)/(3*K+G);          // Poisson ratio
  k=(3.-v)/(1.+v);                    // plane stress
  //k=3.-4*v;                           // plane strain

  double Cs2,Cd2,D,B1,B2,A1,A2;
  if(sqrt(CC)<1.e-16) {               // for static crack
    B1=B2=1.;
    A1=A2=(k+1.)/4.;
  }
  else {                              // for dynamic crack
    Cs2=G/rho_orig;
    Cd2=(k+1.)/(k-1.)*Cs2;
    B1=sqrt(1.-CC/Cd2);
    B2=sqrt(1.-CC/Cs2);
    D=4.*B1*B2-(1.+B2*B2)*(1.+B2*B2);
    A1=B1*(1.-B2*B2)/D;
    A2=B2*(1.-B2*B2)/D;
  }

  double COD2,KI,KII;
  COD2=V1*V1*B2+V2*V2*B1;
  if(sqrt(COD2)<1.e-32) {            // COD=0
    KI  = 0.;
    KII = 0.;
  }
  else {
    KI =V1*sqrt(2.*G*B2*fabs(J1)/A1/COD2);
    KII=V2*sqrt(2.*G*B1*fabs(J1)/A2/COD2);
  }
  SIF=Vector(KI,KII,0.);
}

	 
void HypoElastic::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet* ) const
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pStressLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel,          matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel,       matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
  if(d_8or27==27){
    task->requires(Task::OldDW, lb->pSizeLabel,            matlset,Ghost::None);
  }
  task->requires(Task::NewDW, lb->gVelocityLabel,          matlset, gac, NGN);

#ifdef FRACTURE
  task->requires(Task::NewDW, lb->GVelocityLabel,          matlset, gac, NGN);
  task->requires(Task::NewDW, lb->pgCodeLabel,             matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDispGradsLabel,         matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pStrainEnergyDensityLabel,matlset,Ghost::None);
  task->computes(lb->pDispGradsLabel_preReloc,             matlset);
  task->computes(lb->pStrainEnergyDensityLabel_preReloc,   matlset);
#endif

  task->computes(lb->pStressLabel_preReloc,                matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc,    matlset);
  task->computes(lb->pVolumeDeformedLabel,                 matlset);
}

void 
HypoElastic::addComputesAndRequires(Task* ,
				   const MPMMaterial* ,
				   const PatchSet* ,
				   const bool ) const
{
}

double HypoElastic::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  //double p_ref=101325.0;
  double p_gauge = pressure - p_ref;
  double rho_cur;
  //double G = d_initialData.G;
  double bulk = d_initialData.K;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 0
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

void HypoElastic::computePressEOSCM(double rho_cur, double& pressure,
                                    double p_ref,
                                    double& dp_drho,      double& tmp,
                                    const MPMMaterial* matl)
{

  //double G = d_initialData.G;
  double bulk = d_initialData.K;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

double HypoElastic::getCompressibility()
{
  return 1.0/d_initialData.K;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
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
