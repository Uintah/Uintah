#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompMooneyRivlin.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <values.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;

using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

// Material Constants are C1, C2 and PR (poisson's ratio).  
// The shear modulus = 2(C1 + C2).

CompMooneyRivlin::CompMooneyRivlin(ProblemSpecP& ps, MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;
  ps->require("he_constant_1",d_initialData.C1);
  ps->require("he_constant_2",d_initialData.C2);
  ps->require("he_PR",d_initialData.PR);
  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }
}

CompMooneyRivlin::~CompMooneyRivlin()
{
  // Destructor
}

void CompMooneyRivlin::initializeCMData(const Patch* patch,
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
  // for J-Integral
#ifdef FRACTURE
  ParticleVariable<Matrix3> pdispGrads;
  new_dw->allocateAndPut(pdispGrads, lb->pDispGradsLabel, pset);
  ParticleVariable<double>  pstrainEnergyDensity;
  new_dw->allocateAndPut(pstrainEnergyDensity, lb->pStrainEnergyDensityLabel, pset);
#endif
  for(ParticleSubset::iterator iter =pset->begin();iter != pset->end();iter++){
        deformationGradient[*iter] = Identity;
        pstress[*iter] = zero;
#ifdef FRACTURE
     pdispGrads[*iter] = zero;
     pstrainEnergyDensity[*iter] = 0.0;
#endif
  }

  computeStableTimestep(patch, matl, new_dw);
}

void CompMooneyRivlin::allocateCMDataAddRequires(Task* task,
						   const MPMMaterial* matl,
						   const PatchSet* patch,
						   MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel, Ghost::None);
#ifdef FRACTURE
  task->requires(Task::OldDW,lb->pDispGradsLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStrainEnergyDensityLabel, Ghost::None);
#endif
}


void CompMooneyRivlin::allocateCMDataAdd(DataWarehouse* new_dw,
					 ParticleSubset* addset,
					 map<const VarLabel*, ParticleVariableBase*>* newState,
					 ParticleSubset* delset,
					 DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3  zero(0.);

  ParticleVariable<Matrix3> deformationGradient, pstress;
  constParticleVariable<Matrix3> o_deformationGradient, o_stress;

  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress, addset);

  old_dw->get(o_deformationGradient,lb->pDeformationMeasureLabel,delset);
  old_dw->get(o_stress,lb->pStressLabel,delset);
  // for J-Integral
  
  ParticleVariable<Matrix3> pdispGrads;
  ParticleVariable<double>  pstrainEnergyDensity;

  constParticleVariable<Matrix3> o_dispGrads;
  constParticleVariable<double>  o_strainEnergyDensity;
#ifdef FRACTURE
  new_dw->allocateTemporary(pdispGrads, addset);
  new_dw->allocateTemporary(pstrainEnergyDensity, addset);
  old_dw->get(o_dispGrads,lb->pDispGradsLabel,delset);
  old_dw->get(o_strainEnergyDensity,lb->pStrainEnergyDensityLabel,delset);
#endif

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_deformationGradient[*o];
    pstress[*n] = zero;
#ifdef FRACTURE
    pdispGrads[*n] = o_dispGrads[*o];
    pstrainEnergyDensity[*n] = o_strainEnergy[*o];
#endif
  }

  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[ lb->pStressLabel]=pstress.clone();
#ifdef FRACTURE
  (*newState)[lb->pDispGradsLabel]=pdispGrads.clone();
  (*newState)[ lb->pStrainEnergyDensityLabel]=pstrainEnergyDensity.clone();
#endif

}


void CompMooneyRivlin::computeStableTimestep(const Patch* patch,
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

void CompMooneyRivlin::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Matrix3 Identity,defGradInc,B,velGrad;
    double invar1,invar2,invar3,J,w3,i3w3,C1pi1C2;
    Identity.Identity();
    double c_dil = 0.0,se=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int matlindex = matl->getDWIndex();

    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume_deform;
    constParticleVariable<Vector> pvelocity,psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;

    if(d_8or27==27){
      old_dw->get(psize,             lb->pSizeLabel,                     pset);
    }
    old_dw->get(px,                  lb->pXLabel,                        pset);
    old_dw->get(pmass,               lb->pMassLabel,                     pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,                 pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel,       pset);
    new_dw->allocateAndPut(pstress,        lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pvolume_deform, lb->pVolumeDeformedLabel,     pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);

#ifdef FRACTURE
    constNCVariable<Vector> Gvelocity;
    new_dw->get(Gvelocity,lb->GVelocityLabel, matlindex, patch, gac, NGN);

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

    new_dw->get(gvelocity, lb->gVelocityLabel, matlindex,patch, gac, NGN);
    old_dw->get(delT, lb->delTLabel);

    double C1 = d_initialData.C1;
    double C2 = d_initialData.C2;
    double C3 = .5*C1 + C2;
    double PR = d_initialData.PR;
    double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;
      
      // Get the node indices that surround the cell
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      
      ASSERT(patch->getBox().contains(px[idx]));
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
	    //	      velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
	  }
	}
      }
      
      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      defGradInc = velGrad * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = defGradInc*deformationGradient[idx];

      // Compute the left Cauchy-Green deformation tensor
      B = deformationGradient_new[idx]*deformationGradient_new[idx].Transpose();

      // Compute the invariants
      invar1 = B.Trace();
      invar2 = 0.5*((invar1*invar1) - (B*B).Trace());
      J = deformationGradient_new[idx].Determinant();
      invar3 = J*J;

      w3 = -2.0*C3/(invar3*invar3*invar3) + 2.0*C4*(invar3 -1.0);

      // Compute T = 2/sqrt(I3)*(I3*W3*Identity + (W1+I1*W2)*B - W2*B^2)
      // W1 = C1, W2 = C2
      C1pi1C2 = C1 + invar1*C2;
      i3w3 = invar3*w3;

      pstress[idx]=(B*C1pi1C2 - (B*B)*C2 + Identity*i3w3)*2.0/J;
      
      // Update particle volumes
      pvolume_deform[idx]=(pmass[idx]/rho_orig)*J;

      // Compute wave speed + particle velocity at each particle, 
      // store the maximum
      c_dil = sqrt((4.*(C1+C2*invar2)/J
		    +8.*(2.*C3/(invar3*invar3*invar3)+C4*(2.*invar3-1.))
		    -Min((pstress[idx])(1,1),(pstress[idx])(2,2)
			 ,(pstress[idx])(3,3))/J)
		   *pvolume_deform[idx]/pmass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

      // Compute the strain energy for all the particles
      double e = (C1*(invar1-3.0) + C2*(invar2-3.0) +
            C3*(1.0/(invar3*invar3) - 1.0) +
            C4*(invar3-1.0)*(invar3-1.0))*pvolume_deform[idx]/J;

      se += e;
#ifdef FRACTURE
      // Update particle displacement gradients
      pdispGrads_new[idx] = pdispGrads[idx] + velGrad * delT;
      // Update particle strain energy density
      pstrainEnergyDensity_new[idx] = pstrainEnergyDensity[idx] +
                                         e/pvolume_deform[idx];
#endif
    }
        
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    
    if(delT_new < 1.e-12) delT_new = MAXDOUBLE;
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);    
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
  }
}

void CompMooneyRivlin::carryForward(const PatchSubset* patches,
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

	 
void CompMooneyRivlin::addParticleState(std::vector<const VarLabel*>& from,
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

void CompMooneyRivlin::addComputesAndRequires(Task* task,
					      const MPMMaterial* matl,
					      const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gac   = Ghost::AroundCells;
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel,          matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
  if(d_8or27==27){
    task->requires(Task::OldDW, lb->pSizeLabel,            matlset,Ghost::None);
  }
  task->requires(Task::NewDW, lb->gVelocityLabel,          matlset,gac, NGN);

  task->requires(Task::OldDW, lb->delTLabel);

#ifdef FRACTURE
  task->requires(Task::NewDW, lb->GVelocityLabel,          matlset,gac,NGN);
  task->requires(Task::NewDW, lb->pgCodeLabel,             matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDispGradsLabel,         matlset,Ghost::None);
  task->requires(Task::OldDW,lb->pStrainEnergyDensityLabel,matlset,Ghost::None);
  task->computes(lb->pDispGradsLabel_preReloc,             matlset);
  task->computes(lb->pStrainEnergyDensityLabel_preReloc,   matlset);
#endif

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);
}

void 
CompMooneyRivlin::addComputesAndRequires(Task* ,
				   const MPMMaterial* ,
				   const PatchSet* ,
				   const bool ) const
{
}

double CompMooneyRivlin::computeRhoMicroCM(double /*pressure*/,
                                      const double /*p_ref*/,
					   const MPMMaterial* /*matl*/)
{
#if 0
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
#endif

  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR CompMooneyRivlin"
       << endl;

  double rho_cur=0.;

  return rho_cur;
}

void CompMooneyRivlin::computePressEOSCM(double /*rho_cur*/,double& /*pressure*/,
                                         double /*p_ref*/,
                                         double& /*dp_drho*/, double& /*tmp*/,
                                         const MPMMaterial* /*matl*/)
{
#if 0
  double bulk = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared
#endif

  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR CompMooneyRivlin"
       << endl;
}

double CompMooneyRivlin::getCompressibility()
{
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR CompMooneyRivlin"
       << endl;
  return 1.0;
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
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
