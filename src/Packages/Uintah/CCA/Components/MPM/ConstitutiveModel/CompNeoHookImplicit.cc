#include "ConstitutiveModelFactory.h"
#include "CompNeoHookImplicit.h"
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
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <fstream>
#include <iostream>


using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define OLD_SPARSE
#undef OLD_SPARSE

CompNeoHookImplicit::CompNeoHookImplicit(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
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

CompNeoHookImplicit::~CompNeoHookImplicit()
{
}

void CompNeoHookImplicit::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<Matrix3> deformationGradient, pstress, bElBar;

   new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,
			  pset);
   new_dw->allocateAndPut(pstress,lb->pStressLabel,pset);
   new_dw->allocateAndPut(bElBar,lb->bElBarLabel,pset);


   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          pstress[*iter] = zero;
	  bElBar[*iter] = Identity;
   }

   computeStableTimestep(patch, matl, new_dw);
}

void CompNeoHookImplicit::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
   from.push_back(lb->bElBarLabel);

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
   to.push_back(lb->bElBarLabel_preReloc);
}

void CompNeoHookImplicit::computeStableTimestep(const Patch* patch,
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


void CompNeoHookImplicit::computeStressTensor(const PatchSubset* patches,
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
	          velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
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


void CompNeoHookImplicit::computeStressTensorImplicit(const PatchSubset* patches,
					      const MPMMaterial* matl,
					      DataWarehouse* old_dw,
					      DataWarehouse* new_dw,
					      SparseMatrix<double,int>& KK,
#ifdef HAVE_PETSC
					      Mat &A,
					      map<const Patch*, Array3<int> >& d_petscLocalToGlobal,
#endif
					      const bool recursion)

{
#if 1
  IntVector nodes(0,0,0);
  int num_nodes(0);
#if 0 
  cerr << "nodes = " << nodes << endl;
  cerr << "number of patches = " << patches->size() << endl;
#endif
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    nodes = patch->getNNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z())*3;
#if 0
    cerr << "num_nodes = " << num_nodes << "\n";
#endif
  }

#endif
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    cerr <<"Doing computeStressTensor on " << patch->getID()
	 <<"\t\t\t\t IMPM"<< "\n" << "\n";
#ifdef HAVE_PETSC
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    l2g.copy(d_petscLocalToGlobal[patch]);
#endif

    Matrix3 velGrad,Shear,deformationGradientInc,dispGrad,fbar;
    FastMatrix kmat(24,24);
    FastMatrix kgeo(24,24);
    
    Matrix3 Identity;
    
    Identity.Identity();
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new,bElBar_new;
    constParticleVariable<Matrix3> deformationGradient,bElBar_old;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolume,pvolumeold;
    ParticleVariable<double> pvolume_deformed;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;
    
    if (recursion) {
      DataWarehouse* parent_old_dw = 
	new_dw->getOtherDataWarehouse(Task::ParentOldDW);
      DataWarehouse* parent_new_dw = 
	new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      pset = parent_old_dw->getParticleSubset(dwi, patch);
      parent_old_dw->get(px,             lb->pXLabel,              pset);    
      parent_old_dw->get(pvolume,        lb->pVolumeLabel,         pset);
      parent_old_dw->get(pvolumeold,     lb->pVolumeOldLabel,      pset);      
      parent_new_dw->get(deformationGradient,
			 lb->pDeformationMeasureLabel_preReloc, pset);
      parent_new_dw->get(bElBar_old, lb->bElBarLabel_preReloc, pset);
    }
    else {
      pset = old_dw->getParticleSubset(dwi, patch);
      old_dw->get(px,             lb->pXLabel,              pset);      
      old_dw->get(deformationGradient,lb->pDeformationMeasureLabel,pset);
      old_dw->get(bElBar_old, lb->bElBarLabel, pset);
      old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
      old_dw->get(pvolumeold,          lb->pVolumeOldLabel,          pset);    
    }
    
    cerr << "number of particles = " << pset->numParticles() << endl;
  
    if (recursion)
      old_dw->get(dispNew,lb->dispNewLabel,dwi,patch,Ghost::AroundCells,1);
    else
      new_dw->get(dispNew,lb->dispNewLabel,dwi,patch, Ghost::AroundCells,1);
    
    new_dw->allocateAndPut(pstress,        lb->pStressLabel_preReloc, pset);
    
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel, pset);
    new_dw->allocateAndPut(deformationGradient_new,
		     lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(bElBar_new,lb->bElBarLabel_preReloc, pset);
    
    cerr << "after allocates and puts" << endl;    
    if (pset->begin() == pset->end())
      cerr << "no particles to iterate over . . ." << endl;
    
    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;
    
    FastMatrix B(6,24);
    FastMatrix Btrans(24,6);
    FastMatrix Bnl(3,24);
    FastMatrix Bnltrans(24,3);

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
      particleIndex idx = *iter;
#if 1
      cerr << "Particle " << px[idx] << endl;
#endif
      velGrad.set(0.0);
      dispGrad.set(0.0);
      // Get the node indices that surround the cell
      IntVector ni[8];
      Vector d_S[8];
      
      patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
      vector<int> dof(0);
      
      int l2g_node_num;
      for(int k = 0; k < 8; k++) {
	// Need to loop over the neighboring patches l2g to get the right
	// dof number.
#if 0
	cerr << "working on node = " << ni[k] << endl;
	cerr << "patch extents = " << patch->getNodeLowIndex() << " " 
	     << patch->getNodeHighIndex() << endl;
#endif
	  l2g_node_num = l2g[ni[k]];
	  // cerr << "l2g_node_num = " << l2g_node_num << endl;
	  dof.push_back(l2g_node_num);
	  dof.push_back(l2g_node_num+1);
	  dof.push_back(l2g_node_num+2);
	  

#ifndef HAVE_PETSC 		
	int node_num = ni[k].x() + (nodes.x())*ni[k].y() + (nodes.x())*
	  (nodes.y())*ni[k].z();

	dof.push_back(3*node_num);
	dof.push_back(3*node_num+1);
	dof.push_back(3*node_num+2);
#endif
	const Vector& disp = dispNew[ni[k]];
	
	for (int j = 0; j<3; j++){
	  for (int i = 0; i<3; i++) {
	    dispGrad(i+1,j+1) += disp[i] * d_S[k][j]* oodx[j];
	  }
	}
#if 0
	cerr << "d_Shape = " << d_S[k] << endl;
	cerr << "oodx = " << oodx[0] << "\t" << oodx[1] << "\t" << oodx[2] <<
	  endl;
	cerr << "d_S = " << d_S[k][0]*oodx[0] << "\t" << d_S[k][1]*oodx[1]
	     << "\t" << d_S[k][2]*oodx[2] << endl;
#endif
	
	B(0,3*k) = d_S[k][0]*oodx[0];
	B(3,3*k) = d_S[k][1]*oodx[1];
	B(5,3*k) = d_S[k][2]*oodx[2];
	B(1,3*k) = 0.;
	B(2,3*k) = 0.;
	B(4,3*k) = 0.;
	
	B(1,3*k+1) = d_S[k][1]*oodx[1];
	B(3,3*k+1) = d_S[k][0]*oodx[0];
	B(4,3*k+1) = d_S[k][2]*oodx[2];
	B(0,3*k+1) = 0.;
	B(2,3*k+1) = 0.;
	B(5,3*k+1) = 0.;
	
	B(2,3*k+2) = d_S[k][2]*oodx[2];
	B(4,3*k+2) = d_S[k][1]*oodx[1];
	B(5,3*k+2) = d_S[k][0]*oodx[0];
	B(0,3*k+2) = 0.;
	B(1,3*k+2) = 0.;
	B(3,3*k+2) = 0.;
	
	Bnl(0,3*k) = d_S[k][0]*oodx[0];
	Bnl(1,3*k) = 0.;
	Bnl(2,3*k) = 0.;
	Bnl(0,3*k+1) = 0.;
	Bnl(1,3*k+1) = d_S[k][1]*oodx[1];
	Bnl(2,3*k+1) = 0.;
	Bnl(0,3*k+2) = 0.;
	Bnl(1,3*k+2) = 0.;
	Bnl(2,3*k+2) = d_S[k][2]*oodx[2];
      }
      
      // Find the stressTensor using the displacement gradient
      
      // Inputs: dispGrad
      // Outputs: D,sig (stress tensor), J
      
      
      // Compute the deformation gradient increment using the dispGrad
      
      deformationGradientInc = dispGrad + Identity;
#if 1
      for (int i = 1; i<= 3; i++) {
	for (int j = 1; j <= 3; j++) {
	  cerr << "dispGrad(" << i << "," << j << ")= " << dispGrad(i,j) 
	       << "\t";
	}
	cerr << endl;
      }
      
      for (int i = 1; i<= 3; i++) {
	for (int j = 1; j <= 3; j++) {
	  cerr << "defGradInc(" << i << "," << j << ")= " 
	       << deformationGradientInc(i,j)  << "\t";
	}
	cerr << endl;
      }
      
      for (int i = 1; i<= 3; i++) {
	for (int j = 1; j <= 3; j++) {
	  cerr << "defGrad(" << i << "," << j << ")= " 
	       << deformationGradient[idx](i,j)  << "\t";
	}
	cerr << endl;
      }
#if 0
      for (int i = 1; i<= 3; i++) {
	for (int j = 1; j <= 3; j++) {
	  cerr << "defGrad_new(" << i << "," << j << ")= " 
	       << deformationGradient_new[idx](i,j)  << "\t";
	}
	cerr << endl;
      }
#endif
#endif
      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
	deformationGradient[idx];
      
      // get the volumetric part of the deformation
      double J = deformationGradient_new[idx].Determinant();
      
      fbar = deformationGradientInc * 
	pow(deformationGradientInc.Determinant(),-1./3.);
#if 1
      cerr << "J = " << J << " fbar = " << fbar << endl;
#endif
      bElBar_new[idx] = fbar*bElBar_old[idx]*fbar.Transpose();
#if 0
      cerr << "bElBar_old = " << bElBar_old[idx] << endl;
      
      cerr << "bElBar_new = " << bElBar_new[idx] << endl;
#endif
      // Shear is equal to the shear modulus times dev(bElBar)
      double mubar = 1./3. * bElBar_new[idx].Trace()*shear;
      Matrix3 shrTrl = (bElBar_new[idx]*shear - Identity*mubar);
#if 0
      cerr << "shear " << shear << endl;
      
      for (int i = 1; i<= 3; i++) {
	for (int j = 1; j <= 3; j++) {
	  cerr << "shrTrl(" << i << "," << j << ")= " << shrTrl(i,j) << "\t";
	}
	cerr << endl;
      }
#endif
      // get the hydrostatic part of the stress
      double p = bulk*log(J)/J;
      
      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*p + shrTrl/J;

            
      double coef1 = bulk;
      double coef2 = 2.*bulk*log(J);
#if 0
      cerr << "mubar = " << mubar << " coef1 = " << coef1 << " coef2 = " 
	   << coef2 << endl;
#endif
      FastMatrix D(6,6);
      
      D(0,0) = coef1 - coef2 + 2.*mubar*2./3. - 2./3.*(2.*shrTrl(1,1));
      D(0,1) = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(1,1) + shrTrl(2,2));
      D(0,2) = coef1 -2.*mubar*1./3. - 2./3.*(shrTrl(1,1) + shrTrl(3,3));
      D(0,3) =  - 2./3.*(shrTrl(1,2));
      D(0,4) =  - 2./3.*(shrTrl(1,3));
      D(0,5) =  - 2./3.*(shrTrl(2,3));
      D(1,1) = coef1 - coef2 + 2.*mubar*2./3. - 2./3.*(2.*shrTrl(2,2));
      D(1,2) = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(2,2) + shrTrl(3,3));
      D(1,3) =  - 2./3.*(shrTrl(1,2));
      D(1,4) =  - 2./3.*(shrTrl(1,3));
      D(1,5) =  - 2./3.*(shrTrl(2,3));
      D(2,2) = coef1 - coef2 + 2.*mubar*2./3. - 2./3.*(2.*shrTrl(3,3));
      D(2,3) =  - 2./3.*(shrTrl(1,2));
      D(2,4) =  - 2./3.*(shrTrl(1,3));
      D(2,5) =  - 2./3.*(shrTrl(2,3));
      D(3,3) =  -.5*coef2 + mubar;
      D(3,4) = 0.;
      D(3,5) = 0.;
      D(4,4) =  -.5*coef2 + mubar;
      D(4,5) = 0.;
      D(5,5) =  -.5*coef2 + mubar;
      
      D(1,0)=D(0,1);
      D(2,0)=D(0,2);
      D(2,1)=D(1,2);
      D(3,0)=D(0,3);
      D(3,1)=D(1,3);
      D(3,2)=D(2,3);
      D(4,0)=D(0,4);
      D(4,1)=D(1,4);
      D(4,2)=D(2,4);
      D(4,3)=D(3,4);
      D(5,0)=D(0,5);
      D(5,1)=D(1,5);
      D(5,2)=D(2,5);
      D(5,3)=D(3,5);
      D(5,4)=D(4,5);
      
      FastMatrix sig(3,3);
      for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++) {
	  sig(i,j)=pstress[idx](i+1,j+1);
	}
      }

      // Error it looks like the stress tensor is doubled for some reason.
#if 1
      cerr << "sig = " << "\t" << sig(0,0) << "\t" << sig(0,1) << "\t" 
	   << sig(0,2) << endl;
      cerr << "sig = " << "\t" << sig(1,0) << "\t" << sig(1,1) << "\t" 
	   << sig(1,2) << endl;
      cerr << "sig = " << "\t" << sig(2,0) << "\t" << sig(2,1) << "\t" 
	   << sig(2,2) << endl;
#endif
      
      double volold = pvolumeold[idx];
      double volnew = pvolumeold[idx]*J;
#if 1
      cerr << "volnew = " << volnew << " volold = " << volold << endl;
#endif
      pvolume_deformed[idx] = volnew;
#if 1
      for (int i = 0; i < 6; i++) {
	for (int j = 0; j < 6; j++) {
	  cerr << "D[" << i << "][" << j << "]= " << D(i,j) << "\t";
	}
	cerr << endl;
      }
      
      for (int i = 0; i < 6; i++) {
	for (int j = 0; j < 24; j++) {
	  cerr << "B[" << i << "][" << j << "]= " << B(i,j) << "\t";
	}
	cerr << endl;
      }
      
      for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 24; j++) {
	  cerr << "Bnl[" << i << "][" << j << "]= " << Bnl(i,j) << "\t";
	}
	cerr << endl;
      }
#endif
      // Perform kmat = B.transpose()*D*B*volold
      FastMatrix out(24,6);
      Btrans.transpose(B);
      out.multiply(Btrans, D);
      kmat.multiply(out, B);
      kmat.multiply(volold);
      
      // Perform kgeo = Bnl.transpose*sig*Bnl*volnew;
      FastMatrix out1(24,3);
      Bnltrans.transpose(Bnl);
      out1.multiply(Bnltrans, sig);
      kgeo.multiply(out1, Bnl);
      kgeo.multiply(volnew);
      cerr.precision(16);
#if 0
      for (int d_o_f = 0; d_o_f < (int) dof.size(); d_o_f++)
	cerr << "dof = " << dof[d_o_f] << endl;
#endif
      for (int I = 0; I < (int)dof.size();I++) {
	int dofi = dof[I];
	for (int J = 0; J < (int)dof.size(); J++) {
	  int dofj = dof[J];
#if 1
#ifdef OLD_SPARSE
	  cerr << "KK[" << dofi << "][" << dofj << "]= " << KK[dofi][dofj] 
	       << endl;
#endif
#if 0
	  cerr << "kmat[" << I << "][" << J << "]= " << kmat(I,J) << endl;
	  cerr << "kgeo[" << I << "][" << J << "]= " << kgeo(I,J) << endl;
#endif
#endif
#ifdef OLD_SPARSE
	  KK[dofi][dofj] = KK[dofi][dofj] + (kmat(I,J) + kgeo(I,J));
#endif
#ifdef HAVE_PETSC
	  PetscScalar v = kmat(I,J) + kgeo(I,J);
	  MatSetValues(A,1,&dofi,1,&dofj,&v,ADD_VALUES);
#endif
#ifdef OLD_SPARSE
	  cerr << "KK[" << dofi << "][" << dofj << "]= " << KK[dofi][dofj] 
	       << endl;
#endif
	  
	}
      }
      
    }
  }
#ifdef HAVE_PETSC
  MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);
#endif
  
}


void CompNeoHookImplicit::computeStressTensorImplicitOnly(const PatchSubset* patches,
						   const MPMMaterial* matl,
						   DataWarehouse* old_dw,
						   DataWarehouse* new_dw)


{
   for(int pp=0;pp<patches->size();pp++){
     const Patch* patch = patches->get(pp);
     Matrix3 velGrad,Shear,deformationGradientInc,dispGrad,fbar;

     Matrix3 Identity;

     Identity.Identity();

     Vector dx = patch->dCell();
     double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

     int dwi = matl->getDWIndex();
     ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
     constParticleVariable<Point> px;
     ParticleVariable<Matrix3> deformationGradient_new,bElBar_new;
     constParticleVariable<Matrix3> deformationGradient,bElBar_old;
     ParticleVariable<Matrix3> pstress;
     constParticleVariable<double> pvolume,pvolumeold;
     ParticleVariable<double> pvolume_deformed;
     constNCVariable<Vector> dispNew;
     delt_vartype delT;

     old_dw->get(delT,lb->delTLabel);
     old_dw->get(px,                  lb->pXLabel,                  pset);
     old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
     old_dw->get(pvolumeold,          lb->pVolumeOldLabel,             pset);

     new_dw->get(dispNew,lb->dispNewLabel,dwi,patch,Ghost::AroundCells,1);
     new_dw->getModifiable(pstress,        lb->pStressLabel_preReloc, pset);
     new_dw->getModifiable(pvolume_deformed, lb->pVolumeDeformedLabel, pset);
     new_dw->getModifiable(deformationGradient_new,
			   lb->pDeformationMeasureLabel_preReloc, pset);
     new_dw->getModifiable(bElBar_new,lb->bElBarLabel_preReloc, pset);
     

     cerr << "delT = " << delT << endl;

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

	for(int k = 0; k < 8; k++) {
	  const Vector& disp = dispNew[ni[k]];
	  for (int j = 0; j<3; j++){
	    for (int i = 0; i<3; i++) {
	      dispGrad(i+1,j+1) += disp[i] * d_S[k][j]* oodx[j];
	    }
	  }
	}

	// Find the stressTensor using the displacement gradient

	// Inputs: dispGrad
	// Outputs: D,sig (stress tensor), J


       // Compute the deformation gradient increment using the dispGrad

       deformationGradientInc = dispGrad + Identity;
#if 1
      for (int i = 1; i<= 3; i++) {
	for (int j = 1; j <= 3; j++) {
	  cerr << "dispGrad(" << i << "," << j << ")= " << dispGrad(i,j) 
	       << "\t";
	}
	cerr << endl;
      }
      
      for (int i = 1; i<= 3; i++) {
	for (int j = 1; j <= 3; j++) {
	  cerr << "defGradInc(" << i << "," << j << ")= " 
	       << deformationGradientInc(i,j)  << "\t";
	}
	cerr << endl;
      }
      
#endif
       // Update the deformation gradient tensor to its time n+1 value.
#if 0
       cerr << "Before the update . . ." << endl;
       cerr << "Old defGrad = " << deformationGradient[idx] << endl;
       cerr << "New defGrad = " << deformationGradient_new[idx] << endl;
#endif
       
       deformationGradient_new[idx] = deformationGradientInc *
				      deformationGradient_new[idx];

       // get the volumetric part of the deformation
       double J = deformationGradient_new[idx].Determinant();

       fbar = deformationGradientInc * 
	 pow(deformationGradientInc.Determinant(),-1./3.);

#if 1
      cerr << "J = " << J << " fbar = " << fbar << endl;
#endif
       bElBar_new[idx] = fbar*bElBar_new[idx]*fbar.Transpose();

       // Shear is equal to the shear modulus times dev(bElBar)
       double mubar = 1./3. * bElBar_new[idx].Trace()*shear;
       Matrix3 shrTrl = (bElBar_new[idx]*shear - Identity*mubar);

       // get the hydrostatic part of the stress
       double p = bulk*log(J)/J;

       // compute the total stress (volumetric + deviatoric)
       pstress[idx] = Identity*p + shrTrl/J;


       FastMatrix sig(3,3);
       for (int i = 0; i < 3; i++) {
	 for (int j = 0; j < 3; j++) {
	   sig(i,j)= pstress[idx](i+1,j+1);
	 }
       }

#if 1
      cerr << "sig = " << "\t" << sig(0,0) << "\t" << sig(0,1) << "\t" 
	   << sig(0,2) << endl;
      cerr << "sig = " << "\t" << sig(1,0) << "\t" << sig(1,1) << "\t" 
	   << sig(1,2) << endl;
      cerr << "sig = " << "\t" << sig(2,0) << "\t" << sig(2,1) << "\t" 
	   << sig(2,2) << endl;
#endif

       pvolume_deformed[idx] = pvolumeold[idx]*J;
#if 1
      cerr << "volnew = " << pvolumeold[idx]*J << " volold = " 
	   << pvolumeold[idx] << endl;
#endif

     }
     new_dw->put(delt_vartype(delT),lb->delTLabel);
   }
}

void CompNeoHookImplicit::addInitialComputesAndRequires(Task*,
                                                const MPMMaterial*,
                                                const PatchSet*) const
{

}

void CompNeoHookImplicit::addComputesAndRequires(Task* task,
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


    task->computes(lb->pStressLabel_preReloc,             matlset);
    task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
    task->computes(lb->pVolumeDeformedLabel,              matlset);


}


void CompNeoHookImplicit::addComputesAndRequiresImplicit(Task* task,
						  const MPMMaterial* matl,
						  const PatchSet* patches,
						  const bool recursion)
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
  // new version uses ParentOldDW

  if (recursion) {
    task->requires(Task::ParentOldDW, lb->pXLabel,      matlset, Ghost::None);
    task->requires(Task::ParentOldDW, lb->pVolumeLabel, matlset, Ghost::None);
    task->requires(Task::ParentOldDW,lb->pVolumeOldLabel,matlset,Ghost::None);
    task->requires(Task::ParentNewDW, lb->pDeformationMeasureLabel_preReloc,
		   matlset,Ghost::None);
    task->requires(Task::ParentNewDW,lb->bElBarLabel_preReloc,matlset,
		   Ghost::None);
    task->requires(Task::OldDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);
  }
  else {
    task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
    task->requires(Task::OldDW, lb->pVolumeLabel, matlset, Ghost::None);
    task->requires(Task::OldDW,lb->pVolumeOldLabel,matlset,Ghost::None);
    task->requires(Task::OldDW, lb->pDeformationMeasureLabel,
		   matlset, Ghost::None);
    task->requires(Task::OldDW,lb->bElBarLabel,matlset,Ghost::None);
    task->requires(Task::NewDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);
  }
  

  if(d_8or27==27){
    task->requires(Task::OldDW, lb->pSizeLabel,      matlset, Ghost::None);
  }

  task->computes(lb->pStressLabel_preReloc,matlset);  
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);
  task->computes(lb->bElBarLabel_preReloc,matlset);

  
}

void CompNeoHookImplicit::addComputesAndRequiresImplicitOnly(Task* task,
						     const MPMMaterial* matl,
						     const PatchSet*,
						     const bool)
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeOldLabel, matlset, Ghost::None);

  task->requires(Task::NewDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);
  
  task->requires(Task::OldDW, lb->delTLabel);
  
  if(d_8or27==27){
    task->requires(Task::OldDW, lb->pSizeLabel,      matlset, Ghost::None);
  }
  
  task->modifies(lb->pStressLabel_preReloc);
  task->modifies(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->modifies(lb->pVolumeDeformedLabel,              matlset);
  task->modifies(lb->bElBarLabel_preReloc,matlset);
  task->computes(lb->delTLabel);
  

}



// The "CM" versions use the pressure-volume relationship of the CNH model
double CompNeoHookImplicit::computeRhoMicroCM(double pressure, 
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

void CompNeoHookImplicit::computePressEOSCM(const double rho_cur,double& pressure, 
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

double CompNeoHookImplicit::getCompressibility()
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
    ASSERTEQ(sizeof(CompNeoHookImplicit::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(CompNeoHookImplicit::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "CompNeoHookImplicit::StateData", true, 
				  &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
