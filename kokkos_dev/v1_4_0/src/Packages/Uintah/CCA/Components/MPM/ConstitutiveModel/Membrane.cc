
#include "ConstitutiveModelFactory.h"
#include "Membrane.h"
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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

Membrane::Membrane(ProblemSpecP& ps,  MPMLabel* Mlb)
{
  lb = Mlb;

  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  defGradInPlaneLabel  = VarLabel::create( "p.defgrad_in_plane",
                        ParticleVariable<Matrix3>::getTypeDescription() );

  defGradInPlaneLabel_preReloc  = VarLabel::create( "p.defgrad_in_plane+",
                        ParticleVariable<Matrix3>::getTypeDescription() );

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

   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   new_dw->allocate(defGradIP,           defGradInPlaneLabel,          pset);
   new_dw->allocate(pstress,             lb->pStressLabel,             pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          defGradIP[*iter] = Identity;
          pstress[*iter] = zero;
   }
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(defGradIP,           defGradInPlaneLabel);
   new_dw->put(pstress,             lb->pStressLabel);

   computeStableTimestep(patch, matl, new_dw);
}

void Membrane::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
   from.push_back(lb->pTang1Label);
   from.push_back(lb->pTang2Label);
   from.push_back(defGradInPlaneLabel);

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
   to.push_back(lb->pTang1Label_preReloc);
   to.push_back(lb->pTang2Label_preReloc);
   to.push_back(defGradInPlaneLabel_preReloc);
}

void Membrane::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of cSTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
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

    int matlindex = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<Matrix3> pstress, defGradIPOld;
    ParticleVariable<Matrix3> pstress_new,defGradIP;
    constParticleVariable<double> pmass,pvolume;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> ptang1,ptang2;
    ParticleVariable<Vector> T1,T2;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;
    Vector I(1,0,0);
    Vector J(0,1,0);
    Vector K(0,0,1);

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(defGradIPOld,        defGradInPlaneLabel,          pset);
    old_dw->get(ptang1,              lb->pTang1Label,                  pset);
    old_dw->get(ptang2,              lb->pTang2Label,                  pset);
    new_dw->allocate(pstress_new,    lb->pStressLabel_afterStrainRate, pset);
    new_dw->allocate(pvolume_deformed, lb->pVolumeDeformedLabel,       pset);
    new_dw->allocate(deformationGradient_new,
				lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocate(T1,        lb->pTang1Label_preReloc,              pset);
    new_dw->allocate(T2,        lb->pTang2Label_preReloc,              pset);
    new_dw->allocate(defGradIP, defGradInPlaneLabel_preReloc,          pset);

    new_dw->get(gvelocity,           lb->gVelocityLabel, matlindex,patch,
            Ghost::AroundCells, 1);
    old_dw->get(delT, lb->delTLabel);

    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
       particleIndex idx = *iter;

       velGrad.set(0.0);
       // Get the node indices that surround the cell
       IntVector ni[8];
       Vector d_S[8];
       T1[idx] = ptang1[idx];
       T2[idx] = ptang2[idx];

       patch->findCellAndShapeDerivatives(px[idx], ni, d_S);

       for(int k = 0; k < 8; k++) {
	const Vector& gvel = gvelocity[ni[k]];
	 for (int j = 0; j<3; j++){
	    for (int i = 0; i<3; i++) {
	      velGrad(i+1,j+1) += gvel(i) * d_S[k](j) * oodx[j];
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
      Jvol    = deformationGradient_new[idx].Determinant();

//      double Jinc = deformationGradientInc.Determinant();

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

//      Matrix3 Check = R.Transpose()*R;

//      cout << "R^T*R " << Check << endl;

//    End of rotation tensor computation

      T1[idx] = R*Vector(1,0,0);
      T2[idx] = R*Vector(0,0,1);

      // T3 = T1 X T2
      Vector T3(T1[idx].y()*T2[idx].z() - T1[idx].z()*T2[idx].y(),
              -(T1[idx].x()*T2[idx].z() - T1[idx].z()*T2[idx].x()),
                T1[idx].x()*T2[idx].y() - T1[idx].y()*T2[idx].x());

      Matrix3 Q(Dot(T1[idx],I), Dot(T1[idx],J), Dot(T1[idx],K),
                Dot(T2[idx],I), Dot(T2[idx],J), Dot(T2[idx],K),
                Dot(T3     ,I), Dot(T3     ,J), Dot(T3     ,K));

      Matrix3 L_ij_ip(0.0), L_ip(0.0), L_local;

      L_ij_ip(1,1) = Dot(T1[idx], velGrad*T1[idx]);
      L_ij_ip(1,2) = Dot(T1[idx], velGrad*T2[idx]);
      L_ij_ip(1,3) = Dot(T1[idx], velGrad*T3     );
      L_ij_ip(2,1) = Dot(T2[idx], velGrad*T1[idx]);
      L_ij_ip(2,2) = Dot(T2[idx], velGrad*T2[idx]);
      L_ij_ip(2,3) = Dot(T2[idx], velGrad*T3     );
      L_ij_ip(3,1) = Dot(T3     , velGrad*T1[idx]);
      L_ij_ip(3,2) = Dot(T3     , velGrad*T2[idx]);
      L_ij_ip(3,3) = Dot(T3     , velGrad*T3     );

      Matrix3 T1T1, T1T2, T2T1, T2T2;

      for(int i = 1; i<=3; i++){
        for(int j = 1; j<=3; j++){
          T1T1(i,j) = T1[idx][i]*T1[idx][j];
          T1T2(i,j) = T1[idx][i]*T2[idx][j];
          T2T1(i,j) = T2[idx][i]*T1[idx][j];
          T2T2(i,j) = T2[idx][i]*T2[idx][j];
        }
      }

      L_ip = T1T1*L_ij_ip(1,1) + T1T2*L_ij_ip(1,2) +
             T2T1*L_ij_ip(2,1) + T2T2*L_ij_ip(2,2);

      L_local = Q * L_ip * Q.Transpose();

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
        jv = f33*(F(1,1)*F(2,2) - F(2,1)*F(1,2));

        sig33 = (shear/(3.*pow(jv,2./3.)))*
                (2.*f33*f33 -
                   (F(1,1)*F(1,1)+F(1,2)*F(1,2)+F(2,1)*F(2,1)+F(2,2)*F(2,2))) 
              + (bulk/2.)*(jv - 1/jv);

	f33p = 1.01*f33;
	f33m = 0.99*f33;
        jvp = f33p*(F(1,1)*F(2,2) - F(2,1)*F(1,2));
        jvm = f33m*(F(1,1)*F(2,2) - F(2,1)*F(1,2));

        sig33p = (shear/(3.*pow(jvp,2./3.)))*
                (2.*f33p*f33p -
                   (F(1,1)*F(1,1)+F(1,2)*F(1,2)+F(2,1)*F(2,1)+F(2,2)*F(2,2)))
               + (bulk/2.)*(jvp - 1/jvp);

        sig33m = (shear/(3.*pow(jvm,2./3.)))*
                (2.*f33m*f33m -
                   (F(1,1)*F(1,1)+F(1,2)*F(1,2)+F(2,1)*F(2,1)+F(2,2)*F(2,2)))
               + (bulk/2.)*(jvm - 1/jvm);

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
      U = .5*bulk*(.5*(pow(jv,2.0) - 1.0) - log(jv));
      W = .5*shear*(bElBar_new.Trace() - 3.0);

      pvolume_deformed[idx]=(pmass[idx]/rho_orig)*jv;
      
      double e = (U + W)*pvolume_deformed[idx]/jv;

      se += e;

      Vector pvelocity_idx = pvelocity[idx];
      if(pmass[idx] > 0){
        c_dil = sqrt((bulk + 4.*shear/3.)*pvolume_deformed[idx]/pmass[idx]);
      }
      else{
        c_dil = 0.0;
        pvelocity_idx = Vector(0.0,0.0,0.0);
      }
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
  		       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(pstress_new,            lb->pStressLabel_afterStrainRate);
    new_dw->put(deformationGradient_new,lb->pDeformationMeasureLabel_preReloc);
    new_dw->put(defGradIP,              defGradInPlaneLabel_preReloc);
    new_dw->put(pvolume_deformed,       lb->pVolumeDeformedLabel);
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    new_dw->put(T1,                     lb->pTang1Label_preReloc);
    new_dw->put(T2,                     lb->pTang2Label_preReloc);

  }
}

void Membrane::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet*) const
{
   const MaterialSubset* matlset = matl->thisMaterial();
   task->requires(Task::OldDW, lb->pXLabel,        matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pMassLabel,     matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pVelocityLabel, matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pDeformationMeasureLabel,
						   matlset, Ghost::None);
   task->requires(Task::OldDW, defGradInPlaneLabel,matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pStressLabel,
						   matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pTang1Label,    matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pTang2Label,    matlset, Ghost::None);
   task->requires(Task::NewDW, lb->gVelocityLabel,
						 matlset, Ghost::AroundCells,1);
   task->requires(Task::OldDW, lb->delTLabel);

   task->computes(lb->pStressLabel_afterStrainRate,      matlset);
   task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
   task->computes(defGradInPlaneLabel_preReloc,          matlset);
   task->computes(lb->pVolumeDeformedLabel,              matlset);
   task->computes(lb->pTang1Label_preReloc,              matlset);
   task->computes(lb->pTang2Label_preReloc,              matlset);
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

void Membrane::computePressEOSCM(const double rho_cur,double& pressure, 
                                               const double p_ref,
                                               double& dp_drho, double& tmp,
                                                const MPMMaterial* matl)
{
  //double p_ref=101325.0;
  double bulk = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared
}

double Membrane::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

#ifdef __sgi
#define IRIX
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
