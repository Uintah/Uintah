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
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  //patchAndGhost data
  ParticleVariable<Point>  pX_pg;
  ParticleVariable<double> pVolume_pg;
  ParticleVariable<int>    pIsBroken_pg;
  ParticleVariable<Vector> pCrackNormal_pg[3];

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
  old_dw->get(pCrackNormal_pg[0], lb->pCrackNormal1Label, pset_pg);
  old_dw->get(pCrackNormal_pg[1], lb->pCrackNormal2Label, pset_pg);
  old_dw->get(pCrackNormal_pg[2], lb->pCrackNormal3Label, pset_pg);

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
    for(int facetIdx=0;facetIdx<3;facetIdx++)
    {
      for(int j=0; j<particlesNumber; j++) {
        int idx_pg = particles[j];
        if( pIdx_pg == idx_pg ) continue;
        if(pIsBroken_pg[idx_pg] > facetIdx) {
          double size1 = pow(pVolume_pg[idx_pg],1./3.);
	  const Vector& n1 = pCrackNormal_pg[facetIdx][idx_pg];
      
          Vector dis = particlePoint - pX_pg[idx_pg];
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
    }

    //self side
    for(int facetIdx=0;facetIdx<3;facetIdx++)
    {
      if(pIsBroken_pg[pIdx_pg] > facetIdx) {
        const Vector& n0 = pCrackNormal_pg[facetIdx][pIdx_pg];
        for(int j=0; j<particlesNumber; j++) {
          int idx_pg = particles[j];
          if( pIdx_pg == idx_pg ) continue;
 
          double size1 = pow(pVolume_pg[idx_pg],1./3.);
          Vector dis = pX_pg[idx_pg] - particlePoint;

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
    }

    if(touchFacetsNum>0) pTouchNormal_p_new[pIdx_p].normalize();
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
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);

  static Vector zero(0.,0.,0.);
  
  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  ParticleVariable<Point>  pX_pg;
  ParticleVariable<double> pVolume_pg;
  ParticleVariable<int>    pIsBroken_pg;
  ParticleVariable<Vector> pCrackNormal_pg[3];
  ParticleVariable<Vector> pTouchNormal_pg;

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
  old_dw->get(pCrackNormal_pg[0], lb->pCrackNormal1Label, pset_pg);
  old_dw->get(pCrackNormal_pg[1], lb->pCrackNormal2Label, pset_pg);
  old_dw->get(pCrackNormal_pg[2], lb->pCrackNormal3Label, pset_pg);
  new_dw->get(pTouchNormal_pg, lb->pTouchNormalLabel, pset_pg);

  ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);

  //cout<<"computeConnectivity:computeBoundaryContact: "<< pset_p->numParticles()<<endl;

  ParticleVariable<Point>  pX_p;
  new_dw->get(pX_p, lb->pXXLabel, pset_p);

  vector<int> pIdxEx( pset_p->numParticles() );
  fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);

  ParticleVariable<int>       pConnectivity_p_new;
  ParticleVariable<Vector>    pContactNormal_p_new;
  new_dw->allocate(pConnectivity_p_new, lb->pConnectivityLabel, pset_p);
  new_dw->allocate(pContactNormal_p_new, lb->pContactNormalLabel, pset_p);

  Lattice lattice(pX_pg);
  ParticlesNeighbor particles;
  IntVector cellIdx;
  IntVector nodeIdx[8];

  for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
  {
    particleIndex pIdx_p = *iter;
    particleIndex pIdx_pg = pIdxEx[pIdx_p];
    
    pContactNormal_p_new[pIdx_p] = Vector(0.,0.,0.);
    
    patch->findCell(pX_p[pIdx_p],cellIdx);
    particles.clear();
    particles.buildIn(cellIdx,lattice);
    int particlesNumber = particles.size();

    //ConnectivityInfo
    patch->findNodesFromCell(cellIdx,nodeIdx);
    
    int conn[8];
    for(int k=0;k<8;++k) {
      conn[k] = 1;
      const Point& A = pX_pg[pIdx_pg];
      Point B = patch->nodePosition(nodeIdx[k]);
    
      for(int i=0; i<particlesNumber; i++) {
        int pidx_pg = particles[i];
	
        double r = pow(pVolume_pg[pidx_pg] *0.75/M_PI,0.3333333333);
        double r2 = r*r;
	
	if( pTouchNormal_pg[pidx_pg].length2() > 0.5 )
	{
	  Point O = pX_pg[pidx_pg] + pTouchNormal_pg[pidx_pg] * r;
          if( !particles.visible(A,B,O,pTouchNormal_pg[pidx_pg],r2) ) {
	    conn[k] = 2;
	    pContactNormal_p_new[pIdx_p] = pTouchNormal_pg[pidx_pg];
	    break;
	  }
	}
	
	if(conn[k]==1) for(int facetIdx=0;facetIdx<3;facetIdx++)
	{
	  if(pIsBroken_pg[pidx_pg]>facetIdx) {
	    Point O = pX_pg[pidx_pg] + pCrackNormal_pg[facetIdx][pidx_pg] * r;
	    if( !particles.visible(A,B,O,pCrackNormal_pg[facetIdx][pidx_pg],r2) ) {
	      conn[k] = 0;
	    }
	  }
	}
      }
    }

    Connectivity connectivity(conn);
    pConnectivity_p_new[pIdx_p] = connectivity.flag();
  }
  
  new_dw->put(pConnectivity_p_new, lb->pConnectivityLabel);
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

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  ParticleVariable<Point>  pX_pg;
  ParticleVariable<int>    pIsBroken_pg;
  ParticleVariable<Vector> pCrackNormal_pg[3];

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
  old_dw->get(pCrackNormal_pg[0], lb->pCrackNormal1Label, pset_pg);
  old_dw->get(pCrackNormal_pg[1], lb->pCrackNormal2Label, pset_pg);
  old_dw->get(pCrackNormal_pg[2], lb->pCrackNormal3Label, pset_pg);

  //patchOnly data
  ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  //cout<<"computeFracture:numParticles: "<< pset_p->numParticles()<<endl;
  
  ParticleVariable<Point>  pX_p;
  ParticleVariable<double> pVolume_p;
  ParticleVariable<Matrix3> pStress_p;
  ParticleVariable<double> pStrainEnergy_p;
  ParticleVariable<double> pToughness_p;
  ParticleVariable<Vector> pRotationRate_p;
  ParticleVariable<int> pConnectivity_p;

  new_dw->get(pX_p, lb->pXXLabel, pset_p);
  old_dw->get(pVolume_p, lb->pVolumeLabel, pset_p);
  new_dw->get(pStress_p, lb->pStressAfterStrainRateLabel, pset_p);
  new_dw->get(pStrainEnergy_p, lb->pStrainEnergyLabel, pset_p);
  old_dw->get(pToughness_p, lb->pToughnessLabel, pset_p);
  new_dw->get(pRotationRate_p, lb->pRotationRateLabel, pset_p);
  new_dw->get(pConnectivity_p, lb->pConnectivityLabel, pset_p);

  //particle index exchange from patch to patch+ghost
  vector<int> pIdxEx( pset_p->numParticles() );
  fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);
  Lattice lattice(pX_pg);

  NCVariable<Matrix3> gStress;
  new_dw->get(gStress, lb->gStressForSavingLabel,
    matlindex, patch, Ghost::AroundCells, 1);

  ParticleVariable<int> pIsBroken_p_new;
  ParticleVariable<Vector> pCrackNormal_p_new[3];
  ParticleVariable<Matrix3> pStress_p_new;
  
  new_dw->allocate(pIsBroken_p_new, lb->pIsBrokenLabel, pset_p);
  new_dw->allocate(pCrackNormal_p_new[0], lb->pCrackNormal1Label, pset_p);
  new_dw->allocate(pCrackNormal_p_new[1], lb->pCrackNormal2Label, pset_p);
  new_dw->allocate(pCrackNormal_p_new[2], lb->pCrackNormal3Label, pset_p);
  new_dw->allocate(pStress_p_new, lb->pStressLabel, pset_p);

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  const Vector dx = patch->dCell();
  
  for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
  {
    particleIndex pIdx_p = *iter;
    particleIndex pIdx_pg = pIdxEx[pIdx_p];
    
    pIsBroken_p_new[pIdx_p] = pIsBroken_pg[pIdx_pg];
    pCrackNormal_p_new[0][pIdx_p] = pCrackNormal_pg[0][pIdx_pg];
    pCrackNormal_p_new[1][pIdx_p] = pCrackNormal_pg[1][pIdx_pg];
    pCrackNormal_p_new[2][pIdx_p] = pCrackNormal_pg[2][pIdx_pg];
    
    for(int facetIdx=0;facetIdx<3;facetIdx++) {
      if( pIsBroken_pg[pIdx_pg] > facetIdx ) {
        pCrackNormal_p_new[facetIdx][pIdx_p] += Cross( pRotationRate_p[pIdx_p] * delT, 
	                                    pCrackNormal_pg[facetIdx][pIdx_pg] );
        pCrackNormal_p_new[facetIdx][pIdx_p].normalize();
      }
    }
    
    pStress_p_new[pIdx_p] = pStress_p[pIdx_p];

    if(pIsBroken_pg[pIdx_pg] >= 3) continue;
    
    double area = pow(pVolume_p[pIdx_p],2./3.);
    double resistEnergy = pToughness_p[pIdx_p] * area;
    
    //check toughness
    /*
    cout<<"pStrainEnergy/resistEnergy"
        <<pStrainEnergy[pIdx]/resistEnergy<<endl;
    */
    
    if( resistEnergy > pStrainEnergy_p[pIdx_p] ) continue;

    Vector N;
    double maxStress = getMaxEigenvalue(pStress_p[pIdx_p], N);

    if(maxStress < 0) continue;

    if(pIsBroken_pg[pIdx_pg]>0)
      if(fabs(Dot(N,pCrackNormal_pg[0][pIdx_pg])) > 0.9 ) continue;
    if(pIsBroken_pg[pIdx_pg]>1)
      if(fabs(Dot(N,pCrackNormal_pg[1][pIdx_pg])) > 0.9 ) continue;
    if(pIsBroken_pg[pIdx_pg]>2)
      if(fabs(Dot(N,pCrackNormal_pg[2][pIdx_pg])) > 0.9 ) continue;
    
    const Matrix3& stress = pStress_p[pIdx_p];

    double I2 = 0;
    for(int i=1;i<=3;++i)
    for(int j=1;j<=3;++j) {
      double s = stress(i,j);
      I2 +=  s * s;
    }
      
    double driveEnergy = pStrainEnergy_p[pIdx_p] * maxStress * maxStress / I2;
    //cout<<"driveEnergy/resistEnergy: "<<driveEnergy/resistEnergy<<endl;
    if( driveEnergy < resistEnergy ) continue;
    
    IntVector ni[8];
    Vector d_S[8];
    patch->findCellAndShapeDerivatives(pX_p[pIdx_p], ni, d_S);

    Connectivity connectivity(pConnectivity_p[pIdx_p]);
    int conn[8];
    connectivity.getInfo(conn);
    Connectivity::modifyShapeDerivatives(conn,d_S,Connectivity::connect);

    Vector dSn (0.,0.,0.);
    for(int k = 0; k < 8; k++) {
      if(conn[k]==Connectivity::connect) {
	double gStressNN = Dot(N, gStress[ni[k]]*N);
	for (int i = 1; i<=3; i++) {
	  dSn[i] += gStressNN * d_S[k][i] / dx[i];
	}
      }
    }
    
    if( Dot(dSn,N)<0 ) N=-N;

    //double r = pow(0.75/M_PI*pVolume_p[pIdx_p],1./3.);
    double size = pow(pVolume_p[pIdx_p],1./3.);
    ParticlesNeighbor particles;
    lattice.getParticlesNeighbor(pX_p[pIdx_p], particles);
    int particlesNumber = particles.size();
    
    bool accept = false;
    for(int p=0; p<particlesNumber; p++) {
      int idx_pg = particles[p];

      Vector dis = pX_pg[idx_pg] - pX_pg[pIdx_pg];
      double vdis = Dot(N,dis);
      if( fabs(vdis) > size/2 ) continue;
      Vector Vdis = N * vdis;
      if( (dis-Vdis).length() > size*1.5 ) continue;
      
      if( pIsBroken_pg[idx_pg]>0) {
        if( Dot(pCrackNormal_pg[0][idx_pg],N) > 0.9 ) {
          if( fabs(Dot(dis,pCrackNormal_pg[0][idx_pg])) < size/2 ) {
	    accept = true;
	    break;
	  }
	}
      }
      if(pIsBroken_pg[idx_pg]>1) {
        if( Dot(pCrackNormal_pg[1][idx_pg],N) > 0.9 ) {
          if( fabs(Dot(dis,pCrackNormal_pg[1][idx_pg])) < size/2 ) {
	    accept = true;
	    break;
	  }
	}
      }
      if(pIsBroken_pg[idx_pg]>2) {
        if( Dot(pCrackNormal_pg[2][idx_pg],N) > 0.9 ) {
          if( fabs(Dot(dis,pCrackNormal_pg[2][idx_pg])) < size/2 ) {
	    accept = true;
	    break;
	  }
	}
      }
    }
    if(!accept) continue;

    for(int i=1;i<=3;++i)
    for(int j=1;j<=3;++j) {
      pStress_p_new[pIdx_p](i,j) -= N[i] * maxStress * N[j];
    }
    
    cout<<"crack!"<<endl;

    pCrackNormal_p_new[pIsBroken_p_new[pIdx_p]][pIdx_p] = N;
    pIsBroken_p_new[pIdx_p]++;
  }

  new_dw->put(pIsBroken_p_new, lb->pIsBrokenLabel_preReloc);
  new_dw->put(pCrackNormal_p_new[0], lb->pCrackNormal1Label_preReloc);
  new_dw->put(pCrackNormal_p_new[1], lb->pCrackNormal2Label_preReloc);
  new_dw->put(pCrackNormal_p_new[2], lb->pCrackNormal3Label_preReloc);
  new_dw->put(pStress_p_new, lb->pStressAfterFractureReleaseLabel);
  new_dw->put(pToughness_p, lb->pToughnessLabel_preReloc);
  }
}

NormalFracture::
NormalFracture(ProblemSpecP& ps) : Fracture(ps)
{
}

NormalFracture::~NormalFracture()
{
}

} //namespace Uintah
