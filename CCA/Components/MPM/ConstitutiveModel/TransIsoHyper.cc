#include <Packages/Uintah/CCA/Components/MPM/Crack/FractureDefine.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/TransIsoHyper.h>
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
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h> //added this for stiffness
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

// ____________________transversely isotropic hyperelastic material [Jeff Weiss's]

TransIsoHyper::TransIsoHyper(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
//______________________CONSTRUCTOR (READS INPUT, INITIALIZES SOME MODULI)
{
  lb = Mlb;
  d_useModifiedEOS = false;

//______________________material properties
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("direction_of_symm", d_initialData.a0);
  ps->require("mooney_rivlin_1", d_initialData.c1);
  ps->require("mooney_rivlin_2", d_initialData.c2);
  ps->require("scales_exp_stresses", d_initialData.c3);
  ps->require("controls_uncrimping", d_initialData.c4);
  ps->require("straightened_fibers", d_initialData.c5);
  ps->require("fiber_stretch", d_initialData.lambda_star);
  ps->get("useModifiedEOS",d_useModifiedEOS);// use modified eq. of state, meaning no negative pressure

//______________________interpolation
  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

}

TransIsoHyper::~TransIsoHyper()
// _______________________DESTRUCTOR
{
}

void TransIsoHyper::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
// _____________________STRESS FREE REFERENCE CONFIG
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


void TransIsoHyper::allocateCMDataAddRequires(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet* patch,
					    MPMLabel* lb) const
// _________________________________________STILL EXPERIMENTAL
{
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel, Ghost::None);
}


void TransIsoHyper::allocateCMDataAdd(DataWarehouse* new_dw,
				    ParticleSubset* addset,
				    map<const VarLabel*, ParticleVariableBase*>* newState,
				    ParticleSubset* delset,
				    DataWarehouse* old_dw)
// _________________________________________STILL EXPERIMENTAL
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 zero(0.);
    
  ParticleVariable<Matrix3> deformationGradient, pstress;
  constParticleVariable<Matrix3> o_deformationGradient, o_stress;
  
  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress,addset);
  

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_deformationGradient[*o];
    pstress[*n] = zero;
  }

  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();
}

void TransIsoHyper::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
//______________________________KEEPS TRACK OF THE PARTICLES AND THE RELATED VARIABLES
//______________________________(EACH CM ADD ITS OWN STATE VARS)
//______________________________AS PARTICLES MOVE FROM PATCH TO PATCH
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
}

void TransIsoHyper::computeStableTimestep(const Patch* patch,
					const MPMMaterial* matl,
					DataWarehouse* new_dw)
//__________________________TIME STEP DEPENDS ON:
//__________________________CELL SPACING, VEL OF PARTICLE, MATERIAL WAVE SPEED @ EACH PARTICLE
//__________________________REDUCTION OVER ALL dT'S FROM EVERY PATCH PERFORMED
//__________________________(USE THE SMALLEST dT)
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

  // __________________________________________Compute wave speed at each particle, store the maximum

  double Bulk = d_initialData.Bulk;
  double c1 = d_initialData.c1;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     c_dil = sqrt((Bulk+2./3.*c1)*pvolume[idx]/pmass[idx]); // this is valid only for F=Identity

     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void TransIsoHyper::computeStressTensor(const PatchSubset* patches,
				      const MPMMaterial* matl,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
//___________________________________COMPUTES THE STRESS ON ALL THE PARTICLES IN A GIVEN PATCH FOR A GIVEN MATERIAL
//___________________________________CALLED ONCE PER TIME STEP
//___________________________________CONTAINS A COPY OF computeStableTimestep
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,deformationGradientInc;
    double J,p;
    double U,W,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity;
    Matrix3 rightCauchyGreentilde_new, leftCauchyGreentilde_new;
    double I1tilde,I2tilde,I4tilde,lambda_tilde;
    double dWdI4tilde, d2WdI4tilde2;
    double shear;
    Vector deformed_fiber_vector;

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

//_____________________________________________material parameters
    double Bulk  = d_initialData.Bulk;
    Vector a0 = d_initialData.a0;
    double c1 = d_initialData.c1;
    double c2 = d_initialData.c2;
    double c3 = d_initialData.c3;
    double c4 = d_initialData.c4;
    double c5 = d_initialData.c5;
    double lambda_star = d_initialData.lambda_star;
    double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;//c6 = y-intercept
    double rho_orig = matl->getInitialDensity();

    deformed_fiber_vector=a0;// initialize direction here

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
	 gvel = gvelocity[ni[k]];
	 for (int j = 0; j<3; j++){
	    double d_SXoodx = d_S[k][j] * oodx[j];
	    for (int i = 0; i<3; i++) {
	       velGrad(i,j) += gvel[i] * d_SXoodx;
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
      //verification:
      //double lam = 1.05;
      //deformationGradient_new[idx] = Matrix3(1./sqrt(lam), 0.0, 0.0, 0.0, 1./sqrt(lam), 0.0, 0.0, 0.0 , lam);

      cout << "deformationGradient_new[idx] " << endl << deformationGradient_new[idx] << endl;

      // get the volumetric part of the deformation
      J = deformationGradient_new[idx].Determinant();

      //____________________________________________UNCOUPLE DEVIATORIC AND DILATIONAL PARTS
      //____________________________________________Ftilde=J^(-1/3)*F
      //____________________________________________Fvol=J^1/3*Identity

      //________________________________right Cauchy Green (C) tilde and invariants
      rightCauchyGreentilde_new = deformationGradient_new[idx].Transpose()
				* deformationGradient_new[idx]*pow(J,-(2./3.));

      I1tilde = rightCauchyGreentilde_new.Trace();
      //cout << "I1tilde " << I1tilde << endl;
      I2tilde = 1./2.*(I1tilde*I1tilde-(rightCauchyGreentilde_new*rightCauchyGreentilde_new).Trace());
      //cout << "I2tilde " << I2tilde << endl;
      I4tilde = Dot(deformed_fiber_vector,(rightCauchyGreentilde_new*deformed_fiber_vector));
      //cout << "I4tilde " << I2tilde << endl;
      lambda_tilde = sqrt(I4tilde);

      deformed_fiber_vector = deformationGradient_new[idx]*deformed_fiber_vector*(1./lambda_tilde*pow(J,-(1./3.)));

      Matrix3 DY(deformed_fiber_vector,deformed_fiber_vector);

      //________________________________left Cauchy Green (B) tilde
      leftCauchyGreentilde_new = deformationGradient_new[idx]
				* deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));

      //________________________________hydrostatic pressure term
      p = Bulk*log(J)/J;//present model pressure term
      //p= Bulk*(J-1./J);//neo-hookean model pressure term

      //________________________________strain energy derivatives
      if (lambda_tilde < 1.)
       {dWdI4tilde = 0.;
       d2WdI4tilde2 = 0.;
       shear = 2.*c1+c2;
	}
      else
      if (lambda_tilde < 1.062)
       {
       dWdI4tilde = 0.5*c3*(exp(c4*(lambda_tilde-1.))-1.)/lambda_tilde/lambda_tilde;
       d2WdI4tilde2 = 0.25*c3*(c4*exp(c4*(lambda_tilde-1.))
                                      -1./lambda_tilde*(exp(c4*(lambda_tilde-1.))-1.))
				      /(lambda_tilde*lambda_tilde*lambda_tilde);;
       shear = 2.*c1+0.+I4tilde*(d2WdI4tilde2*lambda_tilde*lambda_tilde/0.25-dWdI4tilde*lambda_tilde/0.5);
	}
      else
       {
       dWdI4tilde = 0.5*(c5+c6/lambda_tilde)/lambda_tilde;
       d2WdI4tilde2 = - 0.25*c6/(lambda_tilde*lambda_tilde*lambda_tilde*lambda_tilde);
       shear = 2.*1.44+0.+I4tilde*(d2WdI4tilde2*lambda_tilde*lambda_tilde/0.25-dWdI4tilde*lambda_tilde/0.5);
       }

      //_______________________________ assemble Cauchy stress
      pstress[idx] = Identity*p
                     +
		     (leftCauchyGreentilde_new*(c1+c2*I1tilde)
		     -
		     leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
		     +
		     DY*dWdI4tilde*I4tilde
		     -
		     Identity * 1./3.*(c1*I1tilde+2.*c2*I2tilde+dWdI4tilde*I4tilde)
		     )*2./J;
      cout << " pstress[idx] " << endl << pstress[idx] << endl;

      // Compute the strain energy for all the particles
      U = log(J)*log(J)*Bulk*1./2.;
      if (lambda_tilde < lambda_star)
       W = c1*(I1tilde-3.)+c2*(I2tilde-3.)+(exp(c4*(lambda_tilde-1.)-1.))*c3;
      else
       W = c1*(I1tilde-3.)+c2*(I2tilde-3.)+c5*lambda_tilde+c6*log(lambda_tilde);

      pvolume_deformed[idx]=(pmass[idx]/rho_orig)*J;

      double e = (U + W)*pvolume_deformed[idx]/J;

      se += e;

      Vector pvelocity_idx = pvelocity[idx];

      c_dil = sqrt((Bulk+1./3.*shear)*pvolume_deformed[idx]/pmass[idx]);

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


void TransIsoHyper::carryForward(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
//___________________________________________________________used with RigidMPM
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

void TransIsoHyper::addComputesAndRequires(Task* task,
					  const MPMMaterial* matl,
					  const PatchSet*) const
//______________________________________TELLS THE SCHEDULER WHAT DATA
//______________________________NEEDS TO BE AVAILABLE AT THE TIME computeStressTensor IS CALLED
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

    task->computes(lb->pStressLabel_preReloc,             matlset);
    task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
    task->computes(lb->pVolumeDeformedLabel,              matlset);


}

void 
TransIsoHyper::addComputesAndRequires(Task* ,
				   const MPMMaterial* ,
				   const PatchSet* ,
				   const bool ) const
//_________________________________________here this one's empty
{
}


// The "CM" versions use the pressure-volume relationship of the CNH model
double TransIsoHyper::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double Bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/Bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    rho_cur = rho_orig*(p_gauge/Bulk + sqrt((p_gauge/Bulk)*(p_gauge/Bulk) +1));
  }
  return rho_cur;
}

void TransIsoHyper::computePressEOSCM(const double rho_cur,double& pressure, 
				    const double p_ref,
				    double& dp_drho, double& tmp,
				    const MPMMaterial* matl)
{
  double Bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = Bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (Bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*Bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*Bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = Bulk/rho_cur;  // speed of sound squared
  }
}

double TransIsoHyper::getCompressibility()
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
    ASSERTEQ(sizeof(TransIsoHyper::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(TransIsoHyper::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "TransIsoHyper::StateData", true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
