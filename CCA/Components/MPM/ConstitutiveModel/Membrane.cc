
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Membrane.h>
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
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h> // just added
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

#define FRACTURE
#undef FRACTURE

Membrane::Membrane(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  defGradInPlaneLabel  = VarLabel::create( "p.defgrad_in_plane",
                        ParticleVariable<Matrix3>::getTypeDescription() );

  defGradInPlaneLabel_preReloc  = VarLabel::create( "p.defgrad_in_plane+",
                        ParticleVariable<Matrix3>::getTypeDescription() );

  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

}

Membrane::~Membrane()
{
  // Destructor
  VarLabel::destroy(defGradInPlaneLabel);
  VarLabel::destroy(defGradInPlaneLabel_preReloc);
}

void Membrane::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<Matrix3> deformationGradient, pstress, defGradIP;

  new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,pset);
  new_dw->allocateAndPut(defGradIP,          defGradInPlaneLabel,         pset);
  new_dw->allocateAndPut(pstress,            lb->pStressLabel,            pset);

  for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          defGradIP[*iter] = Identity;
          pstress[*iter] = zero;
  }
  computeStableTimestep(patch, matl, new_dw);
}


void Membrane::allocateCMDataAddRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet* patch,
					 MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel, Ghost::None);
  task->requires(Task::OldDW,defGradInPlaneLabel, Ghost::None);
}


void Membrane::allocateCMDataAdd(DataWarehouse* new_dw,
				 ParticleSubset* addset,
				 map<const VarLabel*, ParticleVariableBase*>* newState,
				 ParticleSubset* delset,
				 DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();

  ParticleVariable<Matrix3> deformationGradient, pstress, defGradIP;
  constParticleVariable<Matrix3> o_deformationGradient, o_stress, o_defGradIP;

  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(defGradIP,addset);
  new_dw->allocateTemporary(pstress,addset);

  old_dw->get(o_deformationGradient,lb->pDeformationMeasureLabel,delset);
  old_dw->get(o_defGradIP,defGradInPlaneLabel,delset);
  old_dw->get(o_stress,lb->pStressLabel,delset);

  ParticleSubset::iterator o,n=addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_deformationGradient[*o];
    defGradIP[*n] = o_defGradIP[*o];
    pstress[*n] = zero;
  }

  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[defGradInPlaneLabel]=defGradIP.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();

}


void Membrane::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
   from.push_back(defGradInPlaneLabel);

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
   to.push_back(defGradInPlaneLabel_preReloc);
}

void Membrane::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of cSTensor
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

void Membrane::computeStressTensor(const PatchSubset* patches,
				      const MPMMaterial* matl,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){

    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,Shear,bElBar_new,deformationGradientInc;
    double Jvol,p,IEl,U,W,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);
    Matrix3 Identity;
    Identity.Identity();

    Matrix3 Rotation;

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<Matrix3> pstress, defGradIPOld;
    ParticleVariable<Matrix3> pstress_new,defGradIP;
    constParticleVariable<double> pmass,pvolume;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity,psize;
    constParticleVariable<Vector> ptang1,ptang2,pnorm;
    ParticleVariable<Vector> T1,T2,T3;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;
    // BB : 11/09/02 Unused Vars
    //Vector I(1,0,0);
    //Vector J(0,1,0);
    //Vector K(0,0,1);

    Ghost::GhostType  gac   = Ghost::AroundCells;

    if(d_8or27==27){
      old_dw->get(psize,                    lb->pSizeLabel,               pset);
    }
    old_dw->get(px,                         lb->pXLabel,                  pset);
    old_dw->get(pmass,                      lb->pMassLabel,               pset);
    old_dw->get(pstress,                    lb->pStressLabel,             pset);
    old_dw->get(pvelocity,                  lb->pVelocityLabel,           pset);
    old_dw->get(defGradIPOld,               defGradInPlaneLabel,          pset);
    old_dw->get(ptang1,                     lb->pTang1Label,              pset);
    old_dw->get(ptang2,                     lb->pTang2Label,              pset);
    old_dw->get(pnorm,                      lb->pNormLabel,               pset);
    old_dw->get(deformationGradient,        lb->pDeformationMeasureLabel, pset);
    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,     pset);
    new_dw->allocateAndPut(T1,              lb->pTang1Label_preReloc,     pset);
    new_dw->allocateAndPut(T2,              lb->pTang2Label_preReloc,     pset);
    new_dw->allocateAndPut(T3,              lb->pNormLabel_preReloc,      pset);
    new_dw->allocateAndPut(defGradIP,       defGradInPlaneLabel_preReloc, pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                   lb->pDeformationMeasureLabel_preReloc, pset);

    new_dw->get(gvelocity, lb->gVelocityLabel, dwi,patch, gac,NGN);
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

      T1[idx] = ptang1[idx];
      T2[idx] = ptang2[idx];

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
				     deformationGradient[idx];

      // get the volumetric part of the deformation
      Jvol    = deformationGradient_new[idx].Determinant();

//    Compute the rotation using Simo page 244
      Matrix3 C = deformationGradient_new[idx].Transpose()*
		  deformationGradient_new[idx];

      double I1 = C.Trace();
      Matrix3 Csq = C*C;
      double I2 = .5*(I1*I1 - Csq.Trace());
      double I3 = Jvol*Jvol;

      double b = I2 - (I1*I1)/3.0;
      double c = -(2.0/27.0)*I1*I1*I1 + (I1*I2)/3.0 - I3;

      double TOL3 = 1e-8;
      double PI = 3.14159265359;
      double x[4];

//      cout << "Next Particle" << endl;
//      cout << "b = " << b << endl;

      if(fabs(b) <= TOL3){
        c = Max(c,0.);
	x[1] = -pow(c,1./3.);
	x[2] = x[1];
	x[3] = x[1];
      }
      else {
//	cout << "c = " << c << endl;
	double m = 2.*sqrt(-b/3.);
//	cout << "m = " << m << endl;
	double n = (3.*c)/(m*b);
//	cout << "n = " << n << endl;
	if (fabs(n) > 1.0){
          n = (n/fabs(n));
        }
	double t = atan(sqrt(1-n*n)/n)/3.0;
//	cout << "t = " << t << endl;
	for(int i=1;i<=3;i++){
	  x[i] = m * cos(t + 2.*(((double) i) - 1.)*PI/3.);
//	  cout << "x[i] = " << x[i] << endl;
	}
      }
      double lam[4];
      for(int i=1;i<=3;i++){
	lam[i] = sqrt(x[i] + I1/3.0);
      }

      double i1 = lam[1] + lam[2] + lam[3];
      double i2 = lam[1]*lam[2] + lam[1]*lam[3] + lam[2]*lam[3];
      double i3 = lam[1]*lam[2]*lam[3];
      double D = i1*i2 - i3;
//      cout << "D = " << D << endl;

      Matrix3 Ustretch = (C*(i1*i1-i2) + Identity*i1*i3 - Csq)*(1./D);
      Matrix3 Uinv     = (C - Ustretch*i1 + Identity*i2)*(1./i3);

      Matrix3 R = deformationGradient_new[idx]*Uinv;

//      cout << R << endl << endl;

//    End of rotation tensor computation

      T1[idx] = R*ptang1[idx];
      T2[idx] = R*ptang2[idx];

      // T3 = T1 X T2
      T3[idx] = Vector( T1[idx].y()*T2[idx].z() - T1[idx].z()*T2[idx].y(),
                      -(T1[idx].x()*T2[idx].z() - T1[idx].z()*T2[idx].x()),
                        T1[idx].x()*T2[idx].y() - T1[idx].y()*T2[idx].x());

      // The following code is carrying out:
      //Matrix3 Q(Dot(T1[idx],I), Dot(T1[idx],J), Dot(T1[idx],K),
      //          Dot(T2[idx],I), Dot(T2[idx],J), Dot(T2[idx],K),
      //          Dot(T3[idx],I), Dot(T3[idx],J), Dot(T3[idx],K));
      // assuming that I, J and K are the (1,0,0), (0,1,0) and (0,0,1)

      Matrix3 Q(T1[idx].x(), T1[idx].y(), T1[idx].z(),
                T2[idx].x(), T2[idx].y(), T2[idx].z(),
                T3[idx].x(), T3[idx].y(), T3[idx].z());

      Matrix3 L_ij_ip(0.0), L_ip(0.0), L_local;

      Vector vGT1 = velGrad*T1[idx];
      Vector vGT2 = velGrad*T2[idx];
      Vector vGT3 = velGrad*T3[idx];

      L_ij_ip(1,1) = Dot(T1[idx], vGT1);
      L_ij_ip(1,2) = Dot(T1[idx], vGT2);
      L_ij_ip(1,3) = Dot(T1[idx], vGT3);
      L_ij_ip(2,1) = Dot(T2[idx], vGT1);
      L_ij_ip(2,2) = Dot(T2[idx], vGT2);
      L_ij_ip(2,3) = Dot(T2[idx], vGT3);
      L_ij_ip(3,1) = Dot(T3[idx], vGT1);
      L_ij_ip(3,2) = Dot(T3[idx], vGT2);
      L_ij_ip(3,3) = Dot(T3[idx], vGT3);

      Matrix3 T1T1, T1T2, T2T1, T2T2;

      for(int i = 0; i<3; i++){
        for(int j = 0; j<3; j++){
          T1T1(i+1,j+1) = T1[idx][i]*T1[idx][j];
          T1T2(i+1,j+1) = T1[idx][i]*T2[idx][j];
          T2T1(i+1,j+1) = T2[idx][i]*T1[idx][j];
          T2T2(i+1,j+1) = T2[idx][i]*T2[idx][j];
        }
      }

      L_ip = T1T1*L_ij_ip(1,1) + T1T2*L_ij_ip(1,2) +
             T2T1*L_ij_ip(2,1) + T2T2*L_ij_ip(2,2);

      L_local = Q * L_ip * Q.Transpose();

      // BE SURE TO FIX THIS
      // I'm currently setting the tangent and normals back to
      // their original position, and then each timestep rotating
      // them by the total R.  It should be possible to do this
      // incrementally, but the error might be greater.
      T1[idx] = ptang1[idx];
      T2[idx] = ptang2[idx];
      T3[idx] = pnorm[idx];

      Matrix3 defGradIPInc = L_local * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      defGradIP[idx] = defGradIPInc * defGradIPOld[idx];

      // Use Newton's method to determine defGradIP(3,3)
      Matrix3 F=defGradIP[idx];

      double epsilon = 1.e-14;
      double delta = 1.;
      double f33, f33p, f33m, jv, jvp, jvm, sig33, sig33p, sig33m;

      f33 =  1./(F(1,1)*F(2,2));

      while(fabs(delta) > epsilon){
        double detF2=(F(1,1)*F(2,2) - F(2,1)*F(1,2));
        jv = f33*detF2;
        double FinF = F(1,1)*F(1,1)+F(1,2)*F(1,2)+F(2,1)*F(2,1)+F(2,2)*F(2,2);
        sig33 = (shear/(3.*pow(jv,2./3.)))*
                (2.*f33*f33 - FinF) + (.5*bulk)*(jv - 1./jv);

	f33p = 1.01*f33;
	f33m = 0.99*f33;
        jvp = f33p*detF2;
        jvm = f33m*detF2;

        sig33p = (shear/(3.*pow(jvp,2./3.)))*
                (2.*f33p*f33p - FinF) + (.5*bulk)*(jvp - 1./jvp);

        sig33m = (shear/(3.*pow(jvm,2./3.)))*
                (2.*f33m*f33m - FinF) + (.5*bulk)*(jvm - 1./jvm);

        delta = -sig33/((sig33p-sig33m)/(f33p-f33m));

        f33 = f33 + delta;
      }

      // get the volumetric part of the deformation
      jv = f33*(F(1,1)*F(2,2) - F(2,1)*F(1,2));
      defGradIP[idx](3,3) = f33;

      bElBar_new = defGradIP[idx]
		 * defGradIP[idx].Transpose()*pow(jv,-(2./3.));

      IEl = onethird*bElBar_new.Trace();

      // Shear is equal to the shear modulus times dev(bElBar)
      Shear = (bElBar_new - Identity*IEl)*shear;

      // get the hydrostatic part of the stress
      p = 0.5*bulk*(jv - 1.0/jv);

      // compute the total stress (volumetric + deviatoric)
      pstress_new[idx] = Identity*p + Shear/jv;
      pstress_new[idx](3,3) = 0.;

      pstress_new[idx] = Q.Transpose() * pstress_new[idx] * Q;


      // Compute the strain energy for all the particles
      U = .5*bulk*(.5*(jv*jv - 1.0) - log(jv));
      W = .5*shear*(bElBar_new.Trace() - 3.0);

      pvolume_deformed[idx]=(pmass[idx]/rho_orig)*jv;
      
      double e = (U + W)*pvolume_deformed[idx]/jv;

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

void Membrane::addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet*) const
{
   const MaterialSubset* matlset = matl->thisMaterial();
   task->computes(defGradInPlaneLabel,matlset);
}

void Membrane::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet*) const
{
   Ghost::GhostType  gac   = Ghost::AroundCells;
   const MaterialSubset* matlset = matl->thisMaterial();
   task->requires(Task::OldDW,lb->pXLabel,                 matlset,Ghost::None);
   task->requires(Task::OldDW,lb->pMassLabel,              matlset,Ghost::None);
   task->requires(Task::OldDW,lb->pVelocityLabel,          matlset,Ghost::None);
   task->requires(Task::OldDW,lb->pDeformationMeasureLabel,matlset,Ghost::None);
   task->requires(Task::OldDW,defGradInPlaneLabel,         matlset,Ghost::None);
   task->requires(Task::OldDW,lb->pStressLabel,            matlset,Ghost::None);
   task->requires(Task::OldDW,lb->pTang1Label,             matlset,Ghost::None);
   task->requires(Task::OldDW,lb->pTang2Label,             matlset,Ghost::None);
   task->requires(Task::OldDW,lb->pNormLabel,              matlset,Ghost::None);
   if(d_8or27==27){
     task->requires(Task::OldDW, lb->pSizeLabel,           matlset,Ghost::None);
   }
   task->requires(Task::NewDW, lb->gVelocityLabel,         matlset,gac, NGN);
   task->requires(Task::OldDW, lb->delTLabel);

#ifdef FRACTURE
   task->requires(Task::NewDW, lb->pgCodeLabel,          matlset,Ghost::None); 
   task->requires(Task::NewDW, lb->GVelocityLabel,       matlset, gac, NGN);
#endif

   task->computes(lb->pStressLabel_preReloc,             matlset);
   task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
   task->computes(defGradInPlaneLabel_preReloc,          matlset);
   task->computes(lb->pVolumeDeformedLabel,              matlset);
   task->computes(lb->pTang1Label_preReloc,              matlset);
   task->computes(lb->pTang2Label_preReloc,              matlset);
   task->computes(lb->pNormLabel_preReloc,               matlset);
}

void 
Membrane::addComputesAndRequires(Task* ,
				 const MPMMaterial* ,
				 const PatchSet* ,
				 const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double Membrane::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
 // double p_ref=101325.0;
  double bulk = d_initialData.Bulk;

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

  return rho_cur;
}

void Membrane::computePressEOSCM(double rho_cur,double& pressure, 
				 double p_ref,
				 double& dp_drho, double& tmp,
				 const MPMMaterial* matl)
{
  //double p_ref=101325.0;
  double bulk = d_initialData.Bulk;
  //double shear = d_initialData.Shear;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = bulk/rho_cur;  // speed of sound squared
}

double Membrane::getCompressibility()
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
   ASSERTEQ(sizeof(Membrane::StateData), sizeof(double)*0);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(Membrane::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
			       "Membrane::StateData", true, &makeMPI_CMData);
   }
   return td;
}
#endif
} // End namespace Uintah
