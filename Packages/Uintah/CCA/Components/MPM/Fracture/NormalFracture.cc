#include "NormalFracture.h"

#include "ParticlesNeighbor.h"
#include "Connectivity.h"
#include "CrackFace.h"
#include "CellsNeighbor.h"

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <stdlib.h>
#include <list>

namespace Uintah {
using namespace SCIRun;

void
NormalFracture::
initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw)
{
}

void NormalFracture::computeBoundaryContact(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  int matlindex = mpm_matl->getDWIndex();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
      patch, Ghost::AroundCells, 1, lb->pXLabel);

    //patchAndGhost data
    ParticleVariable<Point>  pX_pg;
    ParticleVariable<double> pVolume_pg;
    ParticleVariable<int>    pIsBroken_pg;
    ParticleVariable<Vector> pCrackNormal_pg;
    ParticleVariable<Vector> pVelocity_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
    old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
    old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
    old_dw->get(pVelocity_pg, lb->pVelocityLabel, pset_pg);

    //patchOnly data
    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);

    //cout<<"computeFracture:computeBoundaryContact: "<< pset_p->numParticles()<<endl;

    ParticleVariable<Point>  pX_p;
    new_dw->get(pX_p, lb->pXXLabel, pset_p);

    //particle index exchange from patch to patch+ghost
    vector<int> pIdxEx( pset_p->numParticles() );
    fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);

    Lattice lattice(pX_pg);

    //Allocate new data
    ParticleVariable<Vector> pTouchNormal_p_new;
    new_dw->allocate(pTouchNormal_p_new, lb->pTouchNormalLabel, pset_p);
  
    for(ParticleSubset::iterator iter = pset_p->begin();
      iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = pIdxEx[pIdx_p];

      const Point& particlePoint = pX_pg[pIdx_pg];
      double size0 = pow(pVolume_pg[pIdx_pg],1./3.);
    
      ParticlesNeighbor particles;
      lattice.getParticlesNeighbor(particlePoint, particles);
      int particlesNumber = particles.size();

      int touchFacetsNum = 0;
      pTouchNormal_p_new[pIdx_p] = Vector(0.,0.,0.);
      

      //other side
      for(int j=0; j<particlesNumber; j++) {
        int idx_pg = particles[j];
        if( pIdx_pg == idx_pg ) continue;
        if(pIsBroken_pg[idx_pg] > 0) {
          double size1 = pow(pVolume_pg[idx_pg],1./3.);
          const Vector& n1 = pCrackNormal_pg[idx_pg];
      
          Vector dis = particlePoint - pX_pg[idx_pg];
	  
	  if( Dot( (pVelocity_pg[pIdx_pg]-pVelocity_pg[idx_pg]),
	           dis ) >= 0 ) continue;

          double vDis = Dot( dis, n1 );
	  
          if( vDis>0 && vDis<(size0+size1)/2 ) {
            double hDis = (dis - n1 * vDis).length();
            if(hDis < size1/2) {
	      pTouchNormal_p_new[pIdx_p] -= n1;
  	      touchFacetsNum ++;
	    }
          }
	}
      }

      //self side
      if(pIsBroken_pg[pIdx_pg] > 0) {
        const Vector& n0 = pCrackNormal_pg[pIdx_pg];
        for(int j=0; j<particlesNumber; j++) {
          int idx_pg = particles[j];
          if( pIdx_pg == idx_pg ) continue;
 
          Vector dis = pX_pg[idx_pg] - particlePoint;
	  
	  if( Dot( (pVelocity_pg[idx_pg]-pVelocity_pg[pIdx_pg]),
	           dis ) >= 0 ) continue;

          double size1 = pow(pVolume_pg[idx_pg],1./3.);

          double vDis = Dot( dis, n0 );
          if( vDis>0 && vDis<(size0+size1)/2 ) {
            double hDis = (dis - n0 * vDis).length();
            if(hDis < size0/2) {
	      pTouchNormal_p_new[pIdx_p] += n0;
  	      touchFacetsNum ++;
            }
          }
        }
      }

      if(touchFacetsNum>0) {
        pTouchNormal_p_new[pIdx_p].normalize();
        //cout<<"HAVE crack contact!"<<endl;
      }
    }

    new_dw->put(pTouchNormal_p_new, lb->pTouchNormalLabel);
  }
}

void NormalFracture::computeConnectivity(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  static Vector zero(0.,0.,0.);

  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);

    const Vector dx = patch->dCell();
    double cellLength = dx.x();

    int matlindex = mpm_matl->getDWIndex();
    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
      patch, Ghost::AroundCells, 1, lb->pXLabel);

    ParticleVariable<Point>  pX_pg;
    ParticleVariable<double> pVolume_pg;
    ParticleVariable<int>    pIsBroken_pg;
    ParticleVariable<int>    pIsolated_pg;
    ParticleVariable<Vector> pCrackNormal_pg;
    ParticleVariable<Vector> pTouchNormal_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
    old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
    old_dw->get(pIsolated_pg, lb->pIsolatedLabel, pset_pg);
    old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
    new_dw->get(pTouchNormal_pg, lb->pTouchNormalLabel, pset_pg);

    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);

    //cout<<"computeConnectivity:computeBoundaryContact: "<< pset_p->numParticles()<<endl;

    ParticleVariable<Point>  pX_p;
    new_dw->get(pX_p, lb->pXXLabel, pset_p);

    vector<int> pIdxEx( pset_p->numParticles() );
    fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);

    ParticleVariable<int>       pConnectivity_p_new;
    ParticleVariable<int>       pIsolated_p_new;
    ParticleVariable<Vector>    pContactNormal_p_new;
    new_dw->allocate(pConnectivity_p_new, lb->pConnectivityLabel, pset_p);
    new_dw->allocate(pIsolated_p_new, lb->pIsolatedLabel, pset_p);
    new_dw->allocate(pContactNormal_p_new, lb->pContactNormalLabel, pset_p);

    Lattice lattice(pX_pg);
    ParticlesNeighbor particles;
    IntVector cellIdx;

    vector<BoundaryBand> pBoundaryBand_pg(pset_pg->numParticles());
    Array3<BoundaryBand> gBoundaryBand( 
      patch->getCellLowIndex()-IntVector(1,1,1),
      patch->getCellHighIndex()+IntVector(1,1,1) );

    int pnumber = pBoundaryBand_pg.size();
    for(int pidx=0;pidx<pnumber;++pidx) {
      pBoundaryBand_pg[pidx].setup(pidx,
                             pCrackNormal_pg,
			     pIsBroken_pg,
			     pVolume_pg,
			     lattice,
			     cellLength*0.75);
    }
    
    IntVector l(gBoundaryBand.getLowIndex());
    IntVector h(gBoundaryBand.getHighIndex());
    for (int ii = l.x(); ii < h.x(); ii++)
    for (int jj = l.y(); jj < h.y(); jj++)
    for (int kk = l.z(); kk < h.z(); kk++) {
      IntVector nidx(ii,jj,kk);
      gBoundaryBand[nidx].setup(patch->nodePosition(nidx),
                             pCrackNormal_pg,
			     pIsBroken_pg,
			     pVolume_pg,
			     lattice,
			     cellLength*0.75);
    }

    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p, pIdx_pg;
      pIdx_p = *iter;
      pIdx_pg = pIdxEx[pIdx_p];
      
      pIsolated_p_new[pIdx_p] = pIsolated_pg[pIdx_pg];
      
      if( !pIsolated_pg[pIdx_pg] ) {
    
        pContactNormal_p_new[pIdx_p] = zero;
    
        const Point& part = pX_pg[pIdx_pg];

        patch->findCell(pX_p[pIdx_p],cellIdx);
        particles.clear();
        particles.buildIn(cellIdx,lattice);
        int particlesNumber = particles.size();

        IntVector nodeIdx[8];
        patch->findNodesFromCell(cellIdx,nodeIdx);
    
        int conn[8];
        for(int k=0;k<8;++k) {
          Point node = patch->nodePosition(nodeIdx[k]);

  	  if(pBoundaryBand_pg[pIdx_pg].numCracks() == 0 &&
	     gBoundaryBand[nodeIdx[k]].numCracks() == 0)
	  {
	    conn[k] = 1;
	  }
	  else {
	    conn[k] = 1;
	    VisibilityConnection(
	        particles, pIdx_pg, node,
		pIsBroken_pg, pCrackNormal_pg, pVolume_pg, pX_pg, conn[k]);

	    if(conn[k] == 0) BoundaryBandConnection(
	        pBoundaryBand_pg, gBoundaryBand, 
		pIdx_pg, part, nodeIdx[k], node, conn[k]);
		
            if(conn[k] == 0) ContactConnection(
	        particles, pIdx_pg, node,
		pTouchNormal_pg, pVolume_pg, pX_pg,
		conn[k], pContactNormal_p_new[pIdx_p]);
          }
        } //loop over of k
      
/* 
        if(pIsBroken_pg[pIdx_pg]) {
          for(int k=0;k<8;++k) {
            Point node = patch->nodePosition(nodeIdx[k]);
	    if(Dot(node-part,pCrackNormal_pg[pIdx_pg]) > 0 &&
	       conn[k] == 1)
	    {
	      cout<<"particle: "<<part<<"   node "<<k<<" : "<<node<<endl;
	    }
          }
        }
*/

        Connectivity connectivity(conn);
        pConnectivity_p_new[pIdx_p] = connectivity.flag();

/*
        int numConnected = 0;
        for(int k=0;k<8;++k) {
          if(conn[k] == 1) numConnected++;
        }
        if(numConnected<1) {
          pIsolated_p_new[pIdx_p] = 1;
        }
*/
      } //if isolated
    }
  
    new_dw->put(pConnectivity_p_new, lb->pConnectivityLabel);
    new_dw->put(pIsolated_p_new, lb->pIsolatedLabel_preReloc);
    new_dw->put(pContactNormal_p_new, lb->pContactNormalLabel);
  }
}

void NormalFracture::computeFracture(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    //const Vector dx = patch->dCell();
    //double cellLength = dx.x();

    int matlindex = mpm_matl->getDWIndex();
    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
       patch, Ghost::AroundCells, 1, lb->pXLabel);

    ParticleVariable<Point>  pX_pg;
    ParticleVariable<Vector> pVelocity_pg;
    ParticleVariable<double> pVolume_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    new_dw->get(pVelocity_pg, lb->pVelocityLabel_afterUpdate, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);

    //patchOnly data
    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  
    ParticleVariable<Point>   pX_p;
    ParticleVariable<Vector>  pRotationRate_p;
    ParticleVariable<Matrix3> pStress_p;
    ParticleVariable<int>     pIsBroken_p;
    ParticleVariable<Vector>  pCrackNormal_p;
    ParticleVariable<Vector>  pTipNormal_p;
    ParticleVariable<Vector>  pExtensionDirection_p;
    ParticleVariable<double>  pToughness_p;
    ParticleVariable<double>  pCrackSurfacePressure_p;
    ParticleVariable<double>  pStrainEnergy_p;
    ParticleVariable<double>  pMass_p;

    new_dw->get(pX_p, lb->pXXLabel, pset_p);
    new_dw->get(pRotationRate_p, lb->pRotationRateLabel, pset_p);
    new_dw->get(pStress_p, lb->pStressLabel_afterStrainRate, pset_p);
    old_dw->get(pIsBroken_p, lb->pIsBrokenLabel, pset_p);
    old_dw->get(pCrackNormal_p, lb->pCrackNormalLabel, pset_p);
    old_dw->get(pTipNormal_p, lb->pTipNormalLabel, pset_p);
    old_dw->get(pExtensionDirection_p, lb->pExtensionDirectionLabel, pset_p);
    old_dw->get(pToughness_p, lb->pToughnessLabel, pset_p);
    old_dw->get(pCrackSurfacePressure_p, 
      lb->pCrackSurfacePressureLabel, pset_p);
    new_dw->get(pStrainEnergy_p, lb->pStrainEnergyLabel, pset_p);
    old_dw->get(pMass_p, lb->pMassLabel, pset_p);
    
    //particle index exchange from patch to patch+ghost
    vector<int> pIdxEx( pset_p->numParticles() );
    fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);
    Lattice lattice(pX_pg);

    ParticleVariable<int> pIsBroken_p_new;
    ParticleVariable<Vector> pCrackNormal_p_new;
    ParticleVariable<Matrix3> pStress_p_new;
    ParticleVariable<double> pCrackSurfacePressure_p_new;
    ParticleVariable<Vector> pVelocity_p_new;
  
    new_dw->allocate(pIsBroken_p_new, lb->pIsBrokenLabel, pset_p);
    new_dw->allocate(pCrackNormal_p_new, lb->pCrackNormalLabel, pset_p);
    new_dw->allocate(pStress_p_new, lb->pStressLabel, pset_p);
    new_dw->allocate(pCrackSurfacePressure_p_new,
      lb->pCrackSurfacePressureLabel, pset_p);
    new_dw->allocate(pVelocity_p_new, lb->pVelocityLabel, pset_p);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);
  
    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = pIdxEx[pIdx_p];
    
      pIsBroken_p_new[pIdx_p] = pIsBroken_p[pIdx_p];
      pStress_p_new[pIdx_p] = pStress_p[pIdx_p];
      
      //for explosive fracture
      pCrackSurfacePressure_p_new[pIdx_p] = pCrackSurfacePressure_p[pIdx_p];
      if(pCrackSurfacePressure_p[pIdx_p]>0) {
        pCrackSurfacePressure_p_new[pIdx_p] += 
	  mpm_matl->getPressureRate() * delT;
	//cout<<"pCrackSurfacePressure="<<pCrackSurfacePressure_p_new[pIdx_p]
	//<<endl;
      }

      pVelocity_p_new[pIdx_p] = pVelocity_pg[pIdx_pg];

      pCrackNormal_p_new[pIdx_p] = pCrackNormal_p[pIdx_p];
      if( pIsBroken_p_new[pIdx_p] > 0 ) {
        pCrackNormal_p_new[pIdx_p] += Cross( pRotationRate_p[pIdx_p] * delT, 
	                            pCrackNormal_p_new[pIdx_p] );
        pCrackNormal_p_new[pIdx_p].normalize();
	continue;
      }

      if(pTipNormal_p[pIdx_p].length2()<0.5) continue;

      const Vector& nx = pExtensionDirection_p[pIdx_p];
      const Vector& ny = pTipNormal_p[pIdx_p];

      static double Gmax = 0;

      double R = pow( pVolume_pg[pIdx_pg], 0.333333 ) /2.;
      Point pTip = pX_p[pIdx_p] + ny*R - nx*R;

      ParticlesNeighbor particles;
      lattice.getParticlesNeighbor(pTip, particles);

      //double sigma = Dot(pStress_p[pIdx_p]*ny,ny);
      //cout<<"stress: "<<pStress_p[pIdx_p]<<endl;
      //cout<<"sigma: "<<sigma<<" position: "<<pX_p[pIdx_p]<<endl;
      //cout<<"pTipNormal: "<<pTipNormal_p[pIdx_p]<<endl;
      //cout<<"pExtensionDirection: "<<pExtensionDirection_p[pIdx_p]<<endl;

      //cout<<"sigma*cellLength*2: "<<sigma*cellLength*2
      //    <<" pToughness: "<<pToughness_p[pIdx_p]<<endl;
      //if(sigma*cellLength*2 < pToughness_p[pIdx_p]) continue;

      double G = particles.computeCrackClosureIntegral(
        pTip,R,nx,ny,pStress_p[pIdx_p],pX_pg,pVelocity_pg,pVolume_pg,delT);

      if(G>Gmax) {
        Gmax=G;
        cout<<"Max energy release rate: "<<Gmax<<endl;
      }
    
      Vector& N = pCrackNormal_p_new[pIdx_p];
      if( G > pToughness_p[pIdx_p] ) {
        double sigmay = getMaxEigenvalue(pStress_p[pIdx_p], N);
	if( sigmay <= 0 ) continue;
	double rel = Dot(N,ny);
	if( fabs(rel) < 0.7 ) continue;
	if( rel < 0 ) N = -N;

        //stress release
        double I2 = 0;
        for(int i=1;i<=3;++i)
        for(int j=1;j<=3;++j) {
	  I2 += pStress_p[pIdx_p](i,j) * pStress_p[pIdx_p](i,j);
          pStress_p_new[pIdx_p](i,j) -= N[i] * sigmay * N[j];
        }
        double v = sqrt( 2. * pStrainEnergy_p[pIdx_p] * sigmay * sigmay / I2 /
          pMass_p[pIdx_p] );
	pVelocity_p_new[pIdx_p] -= N * v;

        pIsBroken_p_new[pIdx_p] = 1;
	pCrackSurfacePressure_p_new[pIdx_p] = mpm_matl->getExplosivePressure();
      
        cout<<"crack! "
	    <<"normal="<<pCrackNormal_p_new[pIdx_p]<<endl;
      }
    }
    
    new_dw->put(pToughness_p, lb->pToughnessLabel_preReloc);
    new_dw->put(pIsBroken_p_new, lb->pIsBrokenLabel_preReloc);
    new_dw->put(pCrackNormal_p_new, lb->pCrackNormalLabel_preReloc);
    new_dw->put(pStress_p_new, lb->pStressLabel_afterFracture);
    new_dw->put(pCrackSurfacePressure_p_new, lb->pCrackSurfacePressureLabel_preReloc);
    new_dw->put(pVelocity_p_new, lb->pVelocityLabel_afterFracture);
  }
}

void NormalFracture::computeCrackExtension(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int matlindex = mpm_matl->getDWIndex();

    //patchAndGhost data (_pg)
    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
      patch, Ghost::AroundCells, 1, lb->pXLabel);

    ParticleVariable<Point>  pX_pg;
    ParticleVariable<int>    pIsBroken_pg;
    ParticleVariable<Vector> pCrackNormal_pg;
    ParticleVariable<Vector> pExtensionDirection_pg;
    ParticleVariable<double> pVolume_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    new_dw->get(pIsBroken_pg, lb->pIsBrokenLabel_preReloc, pset_pg);
    new_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel_preReloc, pset_pg);
    old_dw->get(pExtensionDirection_pg, lb->pExtensionDirectionLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);

    //patchAndGhost data (_p)
    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  
    ParticleVariable<Point>   pX_p;
    ParticleVariable<Vector>  pTipNormal_p;

    new_dw->get(pX_p, lb->pXXLabel, pset_p);
    old_dw->get(pTipNormal_p, lb->pTipNormalLabel, pset_p);

    //Lattice for particle neighbor finding
    vector<int> pIdxEx( pset_p->numParticles() );
    fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);
    Lattice lattice(pX_pg);

    //variables to be computed
    ParticleVariable<Vector> pTipNormal_p_new;
    ParticleVariable<Vector> pExtensionDirection_p_new;

    new_dw->allocate(pTipNormal_p_new, lb->pTipNormalLabel, pset_p);
    new_dw->allocate(pExtensionDirection_p_new, 
      lb->pExtensionDirectionLabel, pset_p);
    
    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = pIdxEx[pIdx_p];
    
      if(pExtensionDirection_pg[pIdx_pg].length2()>0.5 && 
         pIsBroken_pg[pIdx_pg])
      {
        pExtensionDirection_p_new[pIdx_p] = Vector(0.,0.,0.);
        pTipNormal_p_new[pIdx_p] = Vector(0.,0.,0.);
      }
      else {
        pExtensionDirection_p_new[pIdx_p] = pExtensionDirection_pg[pIdx_pg];
        pTipNormal_p_new[pIdx_p] = pTipNormal_p[pIdx_p];
      }

      if(pIsBroken_pg[pIdx_pg]) continue;
      if(pExtensionDirection_pg[pIdx_pg].length2()>0.5) continue;
      
      ParticlesNeighbor particles;
      lattice.getParticlesNeighbor(pX_p[pIdx_p], particles);
      int particlesNumber = particles.size();
    
      int extensionIdx_pg = -1;
      double vDistance = DBL_MAX;
      for(int p=0; p<particlesNumber; p++) {
        int idx_pg = particles[p];
        if(pExtensionDirection_pg[idx_pg].length2()>0.5 && pIsBroken_pg[idx_pg])
        {
	  double r = pow(pVolume_pg[idx_pg],0.333333)/2;
	  Vector dis = pX_pg[pIdx_pg] - (pX_pg[idx_pg] + pCrackNormal_pg[idx_pg] * r);
	  if(dis.length()>r*3) continue;
	  if( Dot(dis,pExtensionDirection_pg[idx_pg]) > 0 ) {
            double vDis = fabs( Dot(dis, pCrackNormal_pg[idx_pg]) );
            if(vDis<vDistance) {
              vDistance = vDis;
	      extensionIdx_pg = idx_pg;
	    }
	  }
        }
      }
    
      if( extensionIdx_pg >= 0 && vDistance < pow(pVolume_pg[pIdx_pg],0.333333)*0.8 )
      {
        const Vector& ny = pCrackNormal_pg[extensionIdx_pg];
        const Vector& nx = pExtensionDirection_pg[extensionIdx_pg];
        double r = pow(pVolume_pg[extensionIdx_pg],0.333333)/2;
        Point pTip = pX_pg[extensionIdx_pg] + ny * r - nx * r;
      
        Vector& Ny = pTipNormal_p_new[pIdx_p];
	Vector& Nx = pExtensionDirection_p_new[pIdx_p];
	
	Ny = pCrackNormal_pg[extensionIdx_pg];

        Vector dis = pX_p[pIdx_p] - pTip;
        if(Dot(dis,ny) * Dot(Ny,ny) > 0) Ny = -Ny;
        Nx = dis - Ny * Dot(dis,Ny);
        Nx.normalize();
      }
    }

    new_dw->put(pTipNormal_p_new, lb->pTipNormalLabel_preReloc);
    new_dw->put(pExtensionDirection_p_new, lb->pExtensionDirectionLabel_preReloc);
  }
}

NormalFracture::
NormalFracture(ProblemSpecP& ps) : Fracture(ps)
{
}

NormalFracture::~NormalFracture()
{
}

void NormalFracture::BoundaryBandConnection(
   const vector<BoundaryBand>& pBoundaryBand_pg,
   const Array3<BoundaryBand>& gBoundaryBand,
   particleIndex pIdx_pg,
   const Point& part,
   const IntVector& nodeIdx,
   const Point& node,
   int& conn) const
{
  int part_pBB = pBoundaryBand_pg[pIdx_pg].inside(pIdx_pg);
  int part_gBB = gBoundaryBand[nodeIdx].inside(part);
  int node_pBB = pBoundaryBand_pg[pIdx_pg].inside(node);
  int node_gBB = gBoundaryBand[nodeIdx].inside(node);
	
  if(pBoundaryBand_pg[pIdx_pg].numCracks() == 0) {
    if(node_gBB) {
      if(part_gBB) {
        conn = 1;
	return;
      }
      else {
        conn = 0;
        return;
      }
    }
    else {
      if(part_gBB) {
        conn = 0;
        return;
      }
      else {
        conn = 1;
        return;
      }
    }
  }
  else {
    if(gBoundaryBand[nodeIdx].numCracks() == 0) {
      if(part_pBB) {
	if(node_pBB) {
	  conn = 1;
	  return;
	}
	else {
	  conn = 0;
	  return;
	}
      }
      else {
        if(node_pBB) {
	  conn = 0;
	  return;
	}
        else {
	  conn = 1;
	  return;
	}
      }
    }
    else {
      if(part_pBB) {
        if(node_pBB) {
	  conn = 1;
	  return;
	}
        else {
	  conn = 0;
	  return;
	}
      }
      else {
        if(node_pBB) {
	  conn = 0;
	  return;
	}
        else {
          if(node_gBB) {
            if(part_gBB) {
	      conn = 1;
	      return;
	    }
            else {
	      conn = 0;
	      return;
	    }
	  }
          else {
            if(part_gBB) {
	      conn = 0;
	      return;
	    }
            else {
	      conn = 1;
	      return;
	    }
          }
        }
      }
    }
  }
}


void NormalFracture::VisibilityConnection(
   const ParticlesNeighbor& particles,
   particleIndex pIdx_pg,
   const Point& node,
   const ParticleVariable<int>& pIsBroken_pg,
   const ParticleVariable<Vector>& pCrackNormal_pg,
   const ParticleVariable<double>& pVolume_pg,
   const ParticleVariable<Point>& pX_pg,
   int& conn) const
{
  const Point& part = pX_pg[pIdx_pg];

  if(pIsBroken_pg[pIdx_pg]) {
    Vector dis = node - part;
    double vdis = Dot( pCrackNormal_pg[pIdx_pg], dis );
    if( vdis > 0 ) {
      conn = 0;
      return;
    }
  }
  
  int particlesNumber = particles.size();
  for(int i=0; i<particlesNumber; i++) {
    int pidx_pg = particles[i];
    if(pIsBroken_pg[pidx_pg]) {
      if(pidx_pg != pIdx_pg) {
        double r = connectionRadius(pVolume_pg[pidx_pg]);
	double r2 = r*r;
	const Point& O = pX_pg[pidx_pg];
	if( !particles.visible(part,node,O,pCrackNormal_pg[pidx_pg],r2) ) {
	  conn = 0;
	  return;
	}
      }
    }
  }
}


void NormalFracture::ContactConnection(
   const ParticlesNeighbor& particles,
   particleIndex pIdx_pg,
   const Point& node,
   const ParticleVariable<Vector>& pTouchNormal_pg,
   const ParticleVariable<double>& pVolume_pg,
   const ParticleVariable<Point>& pX_pg,
   int& conn,
   Vector& pContactNormal) const
{
  const Point& part = pX_pg[pIdx_pg];

  if( pTouchNormal_pg[pIdx_pg].length2() > 0.5 ) {
    Vector dis = node-part;
    double vdis = Dot(pTouchNormal_pg[pIdx_pg], dis);
    if( vdis > 0 ) {
      conn = 2;
      pContactNormal = pTouchNormal_pg[pIdx_pg];
      return;
    }
  }

  int particlesNumber = particles.size();
  for(int i=0; i<particlesNumber; i++) {
    int pidx_pg = particles[i];
    if(pidx_pg != pIdx_pg) {
      double r = connectionRadius(pVolume_pg[pidx_pg]);
      double r2 = r*r;
      if( pTouchNormal_pg[pidx_pg].length2() > 0.5 ) {
        const Point& O = pX_pg[pidx_pg];
        if( !particles.visible(part,node,O,pTouchNormal_pg[pidx_pg],r2) )
	{
	  conn = 2;
	  pContactNormal = pTouchNormal_pg[pidx_pg];
	  return;
        }
      }
    }
  }
}


double NormalFracture::connectionRadius(double volume)
{
  return pow(volume,0.333333)/1.414;
}


//for debugging
bool
NormalFracture::
isDebugParticle(const Point& p)
{
  double cellsize=0.1;
  double particlesize = cellsize/2;
  Point p0 = Point(cellsize/4,cellsize/4,cellsize/4);
  Vector d = p - p0;
  
  if( fabs(d.x()) < particlesize/2 &&
      fabs(d.y()) < particlesize/2 &&
      fabs(d.z()) < particlesize/2 ) return true;
  else return false;
}

} //namespace Uintah
