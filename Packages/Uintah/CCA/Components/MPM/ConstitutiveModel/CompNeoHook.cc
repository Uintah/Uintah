#include "ConstitutiveModelFactory.h"
#include "CompNeoHook.h"
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
#include <Core/Datatypes/DenseMatrix.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define IMPLICIT
#undef IMPLICIT

CompNeoHook::CompNeoHook(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

#ifdef IMPLICIT
  bElBarLabel = VarLabel::create("bElBar",
		       ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc = VarLabel::create("bElBar_preReloc",
			 ParticleVariable<Matrix3>::getTypeDescription());
#endif
  d_8or27 = n8or27;

}

CompNeoHook::~CompNeoHook()
{
#ifdef IMPLICIT
  VarLabel::destroy(bElBarLabel);  
  VarLabel::destroy(bElBarLabel_preReloc);  
#endif
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

   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   new_dw->allocate(pstress,             lb->pStressLabel,             pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          pstress[*iter] = zero;
   }
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);

   computeStableTimestep(patch, matl, new_dw);
}

void CompNeoHook::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
#ifdef IMPLICIT
   from.push_back(bElBarLabel);
#endif

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
#ifdef IMPLICIT
   to.push_back(bElBarLabel_preReloc);
#endif
}

void CompNeoHook::computeStableTimestep(const Patch* patch,
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

    int matlindex = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,pvolume;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    new_dw->allocate(pstress,        lb->pStressLabel_preReloc,    pset);
    new_dw->allocate(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocate(deformationGradient_new,
		     lb->pDeformationMeasureLabel_preReloc, pset);

    new_dw->get(gvelocity,           lb->gVelocityLabel, matlindex,patch,
            Ghost::AroundCells, 1);
    old_dw->get(delT, lb->delTLabel);
    constParticleVariable<Vector> psize;
    if(d_8or27==27){
      old_dw->get(psize,             lb->pSizeLabel,                  pset);
    }

    constParticleVariable<int> pConnectivity;
    ParticleVariable<Vector> pRotationRate;
    ParticleVariable<double> pStrainEnergy;

    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
       particleIndex idx = *iter;

       velGrad.set(0.0);
       // Get the node indices that surround the cell
       IntVector ni[MAX_BASIS];
       Vector d_S[MAX_BASIS];

       if(d_8or27==8){
          patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
        }
        else if(d_8or27==27){
          patch->findCellAndShapeDerivatives27(px[idx], ni, d_S,psize[idx]);
        }

       for(int k = 0; k < d_8or27; k++) {
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
      U = .5*bulk*(.5*(pow(J,2.0) - 1.0) - log(J));
      W = .5*shear*(bElBar_new.Trace() - 3.0);

      pvolume_deformed[idx]=(pmass[idx]/rho_orig)*J;
      
      double e = (U + W)*pvolume_deformed[idx]/J;

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
    new_dw->put(pstress,                lb->pStressLabel_preReloc);
    new_dw->put(deformationGradient_new,lb->pDeformationMeasureLabel_preReloc);
    new_dw->put(pvolume_deformed,       lb->pVolumeDeformedLabel);
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);

  }
}

void CompNeoHook::computeStressTensorImplicit(const PatchSubset* patches,
					      const MPMMaterial* matl,
					      DataWarehouse* old_dw,
					      DataWarehouse* new_dw,
					      SparseMatrix<double,int>& KK,
					      const bool recursion)

{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,Shear,deformationGradientInc,
      dispGrad,fbar;
    DenseMatrix kmat(24,24);
    DenseMatrix kgeo(24,24);
    
    Matrix3 Identity;

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int matlindex = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new,bElBar_new;
    constParticleVariable<Matrix3> deformationGradient,bElBar_old;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolume;
    ParticleVariable<double> pvolume_deformed;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(bElBar_old, bElBarLabel, pset);
    old_dw->get(dispNew,lb->dispNewLabel,matlindex,patch,Ghost::AroundCells,1);

    if (recursion)
      new_dw->allocate(pstress,        lb->pStressLabel_preReloc, pset);
    else
      new_dw->allocate(pstress,        lb->pStressLabel, pset);

    new_dw->allocate(pvolume_deformed, lb->pVolumeDeformedLabel, pset);
    new_dw->allocate(deformationGradient_new,
		     lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocate(bElBar_new,bElBarLabel_preReloc, pset);



    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;


    DenseMatrix B(6,24);
    DenseMatrix Bnl(3,24);
    
    IntVector nodes = patch->getNNodes();
    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
       particleIndex idx = *iter;

       velGrad.set(0.0);
       dispGrad.set(0.0);
       // Get the node indices that surround the cell
       IntVector ni[8];
       Vector d_S[8];

       patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
       int dof[24];

       for(int k = 0; k < 8; k++) {
	 const Vector& disp = dispNew[ni[k]];

	 int ii = 0;
	 int node_num = ni[k].x() + (nodes.x()+1)*ni[k].y() + (nodes.x() + 1)*
	   (nodes.y()+1)*ni[k].z();
	 dof[ii++] = 3*node_num;
	 dof[ii++] = 3*node_num+1;
	 dof[ii++] = 3*node_num+2;
	 
	 for (int j = 0; j<3; j++){
	   for (int i = 0; i<3; i++) {
	     dispGrad(i+1,j+1) += disp(i) * d_S[k](j)* oodx[j];
	   }
	 }
	 B[0][3*k] = d_S[k](0)*oodx[0];
	 B[3][3*k] = d_S[k](1)*oodx[1];
	 B[5][3*k] = d_S[k](2)*oodx[2];
	 B[1][3*k] = 0.;
	 B[2][3*k] = 0.;
	 B[4][3*k] = 0.;

	 B[1][3*k+1] = d_S[k](1)*oodx[1];
	 B[3][3*k+1] = d_S[k](0)*oodx[0];
	 B[4][3*k+1] = d_S[k](2)*oodx[2];
	 B[0][3*k+1] = 0.;
	 B[2][3*k+1] = 0.;
	 B[5][3*k+1] = 0.;

	 B[2][3*k+2] = d_S[k](2)*oodx[2];
	 B[4][3*k+2] = d_S[k](1)*oodx[1];
	 B[5][3*k+2] = d_S[k](0)*oodx[0];
	 B[0][3*k+2] = 0.;
	 B[1][3*k+2] = 0.;
	 B[3][3*k+2] = 0.;

	 Bnl[0][3*k] = d_S[k](0)*oodx[0];
	 Bnl[1][3*k] = 0.;
	 Bnl[2][3*k] = 0.;
	 Bnl[0][3*k+1] = 0.;
	 Bnl[1][3*k+1] = d_S[k](1)*oodx[1];
	 Bnl[2][3*k+1] = 0.;
	 Bnl[0][3*k+2] = 0.;
	 Bnl[1][3*k+2] = 0.;
	 Bnl[2][3*k+2] = d_S[k](2)*oodx[2];
       }
       
       // Find the stressTensor using the displacement gradient

       // Inputs: dispGrad
       // Outputs: D,sig (stress tensor), J


      // Compute the deformation gradient increment using the dispGrad

      deformationGradientInc = dispGrad + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
				     deformationGradient[idx];

      // get the volumetric part of the deformation
      double J = deformationGradient_new[idx].Determinant();

      fbar = deformationGradientInc * 
	pow(deformationGradientInc.Determinant(),-1./3.);


      bElBar_new[idx] = fbar*bElBar_old[idx]*fbar.Transpose();

      // Shear is equal to the shear modulus times dev(bElBar)
      double mubar = 1./3. * bElBar_new[idx].Trace()*shear;
      Matrix3 shrTrl = (bElBar_new[idx]*shear - Identity*mubar);

      // get the hydrostatic part of the stress
      double p = bulk*log(J)/J;

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*p + shrTrl/J;


      double coef1 = bulk;
      double coef2 = 2.*bulk*log(J);

      DenseMatrix D(6,6);

      D[0][0] = coef1 - coef2 + 2.*mubar*1./3. - 2./3.*(2.*shrTrl(1,1));
      D[0][1] = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(1,1) + shrTrl(2,2));
      D[0][2] = coef1 -2.*mubar*1./3. - 2./3.*(shrTrl(1,1) + shrTrl(3,3));
      D[0][3] =  - 2./3.*(shrTrl(1,2));
      D[0][4] =  - 2./3.*(shrTrl(1,3));
      D[0][5] =  - 2./3.*(shrTrl(2,3));
      D[1][1] = coef1 - coef2 + 2.*mubar*1./3. - 2./3.*(2.*shrTrl(2,2));
      D[1][2] = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(2,2) + shrTrl(3,3));
      D[1][3] =  - 2./3.*(shrTrl(1,2));
      D[1][4] =  - 2./3.*(shrTrl(1,3));
      D[1][5] =  - 2./3.*(shrTrl(2,3));
      D[2][2] = coef1 - coef2 + 2.*mubar*1./3. - 2./3.*(2.*shrTrl(3,3));
      D[2][3] =  - 2./3.*(shrTrl(1,2));
      D[2][4] =  - 2./3.*(shrTrl(1,3));
      D[2][5] =  - 2./3.*(shrTrl(2,3));
      D[3][3] =  -.5*coef2 + mubar;
      D[3][4] = 0.;
      D[3][5] = 0.;
      D[4][4] =  -.5*coef2 + mubar;
      D[4][5] = 0.;
      D[5][5] =  -.5*coef2 + mubar;

      D[1][0]=D[0][1];
      D[2][0]=D[0][2];
      D[2][1]=D[1][2];
      D[3][0]=D[0][3];
      D[3][1]=D[1][3];
      D[3][2]=D[2][3];
      D[4][0]=D[0][4];
      D[4][1]=D[1][4];
      D[4][2]=D[2][4];
      D[4][3]=D[3][4];
      D[5][0]=D[0][5];
      D[5][1]=D[1][5];
      D[5][2]=D[2][5];
      D[5][3]=D[3][5];
      D[5][4]=D[4][5];

      DenseMatrix sig(3,3);
      for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++) {
	  sig.put(i,j, (pstress[idx])(i+1,j+1));
	}
      }
     

      double volold = pvolume[idx];
      double volnew = pvolume[idx]*J;

      pvolume_deformed[idx] = volnew;

      // Perform kmat = B.transpose()*D*B*volold
      DenseMatrix out(24,6);
      DenseMatrix Btrans = B;
      Btrans.transpose();
      Mult(out,Btrans,D);
      Mult(kmat,out,B);
      kmat.mult(volold);

      // Perform kgeo = Bnl.transpose*sig*Bnl*volnew;
      DenseMatrix out1(24,3);
      DenseMatrix Bnltrans = Bnl;
      Bnltrans.transpose();
      Mult(out1,Bnltrans,sig);
      Mult(kgeo,out1,Bnl);
      kgeo.mult(volnew);

      for (int I = 0; I < 24;I++) {
	int dofi = dof[I];
	for (int J = 0; J < 24; J++) {
	  int dofj = dof[J];
	  KK[dofi][dofj] = KK[dofi][dofj] + (kmat[I][J] + kgeo[I][J]);
	}
      }
      

    }
    if (recursion)
      new_dw->put(pstress,                lb->pStressLabel_preReloc);
    else
      new_dw->put(pstress,                lb->pStressLabel);
    new_dw->put(deformationGradient_new,lb->pDeformationMeasureLabel_preReloc);
    new_dw->put(pvolume_deformed,       lb->pVolumeDeformedLabel);
    new_dw->put(bElBar_new,             bElBarLabel_preReloc);

  }
}


void CompNeoHook::computeStressTensorImplicitOnly(const PatchSubset* patches,
						  const MPMMaterial* matl,
						  DataWarehouse* old_dw,
						  DataWarehouse* new_dw)
						  

{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,Shear,deformationGradientInc,
      dispGrad,fbar;
    
    Matrix3 Identity;

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int matlindex = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new,bElBar_new;
    constParticleVariable<Matrix3> deformationGradient,bElBar_old;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolume;
    ParticleVariable<double> pvolume_deformed;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(bElBar_old, bElBarLabel, pset);
    old_dw->get(dispNew,lb->dispNewLabel,matlindex,patch,Ghost::AroundCells,1);

    new_dw->allocate(pstress,        lb->pStressLabel_preReloc, pset);
    new_dw->allocate(pvolume_deformed, lb->pVolumeDeformedLabel, pset);
    new_dw->allocate(deformationGradient_new,
		     lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocate(bElBar_new,bElBarLabel_preReloc, pset);



    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;

    IntVector nodes = patch->getNNodes();
    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
       particleIndex idx = *iter;

       velGrad.set(0.0);
       dispGrad.set(0.0);
       // Get the node indices that surround the cell
       IntVector ni[8];
       Vector d_S[8];

       patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
       int dof[24];

       for(int k = 0; k < 8; k++) {
	 const Vector& disp = dispNew[ni[k]];

	 int ii = 0;
	 int node_num = ni[k].x() + (nodes.x()+1)*ni[k].y() + (nodes.x() + 1)*
	   (nodes.y()+1)*ni[k].z();
	 dof[ii++] = 3*node_num;
	 dof[ii++] = 3*node_num+1;
	 dof[ii++] = 3*node_num+2;
	 
	 for (int j = 0; j<3; j++){
	   for (int i = 0; i<3; i++) {
	     dispGrad(i+1,j+1) += disp(i) * d_S[k](j)* oodx[j];
	   }
	 }
       }
       
       // Find the stressTensor using the displacement gradient

       // Inputs: dispGrad
       // Outputs: D,sig (stress tensor), J


      // Compute the deformation gradient increment using the dispGrad

      deformationGradientInc = dispGrad + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
				     deformationGradient[idx];

      // get the volumetric part of the deformation
      double J = deformationGradient_new[idx].Determinant();

      fbar = deformationGradientInc * 
	pow(deformationGradientInc.Determinant(),-1./3.);


      bElBar_new[idx] = fbar*bElBar_old[idx]*fbar.Transpose();

      // Shear is equal to the shear modulus times dev(bElBar)
      double mubar = 1./3. * bElBar_new[idx].Trace()*shear;
      Matrix3 shrTrl = (bElBar_new[idx]*shear - Identity*mubar);

      // get the hydrostatic part of the stress
      double p = bulk*log(J)/J;

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*p + shrTrl/J;


      DenseMatrix sig(3,3);
      for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++) {
	  sig.put(i,j, (pstress[idx])(i+1,j+1));
	}
      }
     
      pvolume_deformed[idx] = pvolume[idx]*J;

    }

    new_dw->put(pstress,                lb->pStressLabel_preReloc);
    new_dw->put(deformationGradient_new,lb->pDeformationMeasureLabel_preReloc);
    new_dw->put(pvolume_deformed,       lb->pVolumeDeformedLabel);
    new_dw->put(bElBar_new,             bElBarLabel_preReloc);

  }
}

void CompNeoHook::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet*) const
{
   const MaterialSubset* matlset = matl->thisMaterial();
   task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pMassLabel,   matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pVelocityLabel, matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pDeformationMeasureLabel,
						 matlset, Ghost::None);
   task->requires(Task::NewDW,lb->gVelocityLabel,matlset,Ghost::AroundCells,1);
   
   task->requires(Task::OldDW, lb->delTLabel);

   if(d_8or27==27){
     task->requires(Task::OldDW, lb->pSizeLabel,      matlset, Ghost::None);
   }

   task->computes(lb->pStressLabel_preReloc,             matlset);
   task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
   task->computes(lb->pVolumeDeformedLabel,              matlset);

   
}


void CompNeoHook::addComputesAndRequiresImplicit(Task* task,
						 const MPMMaterial* matl,
						 const PatchSet*,
						 const bool recursion)
{
   const MaterialSubset* matlset = matl->thisMaterial();
   task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pMassLabel,   matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pVelocityLabel, matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pDeformationMeasureLabel,
						 matlset, Ghost::None);

   task->requires(Task::OldDW,bElBarLabel,matlset,Ghost::None);
   task->requires(Task::NewDW,lb->gVelocityLabel,matlset,Ghost::AroundCells,1);
   task->requires(Task::OldDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);
   
   task->requires(Task::OldDW, lb->delTLabel);

   if(d_8or27==27){
     task->requires(Task::OldDW, lb->pSizeLabel,      matlset, Ghost::None);
   }

   if (recursion)
     task->modifies(lb->pStressLabel_preReloc);
   else
     task->computes(lb->pStressLabel,matlset);


   task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
   task->computes(lb->pVolumeDeformedLabel,              matlset);
   task->computes(bElBarLabel_preReloc,matlset);
   
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double CompNeoHook::computeRhoMicroCM(double pressure, 
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

void CompNeoHook::computePressEOSCM(const double rho_cur,double& pressure, 
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
