#include "NormalFracture.h"

#include "ParticlesNeighbor.h"
#include "Visibility.h"
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
                            DataWarehouseP& new_dw)
{
}

void NormalFracture::computeBoundaryContact(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  //patchAndGhost data
  ParticleVariable<Point>  pX_pg;
  ParticleVariable<Vector> pCrackNormal_pg;
  ParticleVariable<int>    pIsBroken_pg;
  ParticleVariable<double> pVolume_pg;
  ParticleVariable<Vector> pVelocity_pg;

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
  old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
  old_dw->get(pVelocity_pg, lb->pVelocityLabel, pset_pg);

  //patchOnly data
  ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<Point>  pX_p;
  new_dw->get(pX_p, lb->pXXLabel, pset_p);

  //particle index exchange from patch to patch+ghost
  vector<int> pIdxEx( pset_p->numParticles() );
  fit(pset_p,pX_p,pset_pg,pX_pg,pIdxEx);

  Lattice lattice(pX_pg);
  ParticlesNeighbor particles;
  IntVector cellIdx;

  //Allocate new data
  ParticleVariable<Vector> pVelocity_p_new;
  ParticleVariable<Vector> pCrackSurfaceContactForce_p_new;
  
  new_dw->allocate(pVelocity_p_new,
    lb->pVelocityAfterBoundaryContactLabel, pset_p);
  new_dw->allocate(pCrackSurfaceContactForce_p_new, 
    lb->pCrackSurfaceContactForceLabel, pset_p);
  
  for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
  {
    particleIndex pIdx_p = *iter;
    particleIndex pIdx_pg = pIdxEx[pIdx_p];

    pVelocity_p_new[pIdx_p] = pVelocity_pg[pIdx_pg];
    pCrackSurfaceContactForce_p_new[pIdx_p] = Vector(0.,0.,0.);
  }

  new_dw->put(pVelocity_p_new,
    lb->pVelocityAfterBoundaryContactLabel);
  new_dw->put(pCrackSurfaceContactForce_p_new, 
    lb->pCrackSurfaceContactForceLabel);
}

void NormalFracture::computeVisibility(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  ParticleVariable<Point>  pX_pg;
  ParticleVariable<double> pVolume_pg;
  ParticleVariable<Vector> pCrackNormal_pg;
  ParticleVariable<int>    pIsBroken_pg;

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
  old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);

  ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<int>    pVisibility_p;
  new_dw->allocate(pVisibility_p, lb->pVisibilityLabel, pset_p);
  ParticleVariable<Point>  pX_p;
  new_dw->get(pX_p, lb->pXXLabel, pset_p);

  vector<int> pIdxEx( pset_p->numParticles() );
  fit(pset_p,pX_p,
      pset_pg,pX_pg,
      pIdxEx);

  Lattice lattice(pX_pg);
  ParticlesNeighbor particles;
  IntVector cellIdx;
  IntVector nodeIdx[8];

  for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
  {
    particleIndex pIdx = *iter;
    patch->findCell(pX_p[pIdx],cellIdx);
    particles.clear();
    particles.buildIn(cellIdx,lattice);
    
    //visibility
    patch->findNodesFromCell(cellIdx,nodeIdx);
    
    Visibility vis;
    for(int i=0;i<8;++i) {
      if(particles.visible( pIdxEx[pIdx],
                            patch->nodePosition(nodeIdx[i]),
		            pX_pg,
		            pIsBroken_pg,
		            pCrackNormal_pg,
		            pVolume_pg) ) vis.setVisible(i);
      else {
        vis.setUnvisible(i);
      }
    }
    pVisibility_p[pIdx] = vis.flag();
  }
  
  new_dw->put(pVisibility_p, lb->pVisibilityLabel);
}

void NormalFracture::computeFracture(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  // Create arrays for the particle data
  ParticleVariable<Point>  pX_pg;
  ParticleVariable<double> pVolume_pg;
  ParticleVariable<Vector> pCrackNormal_pg;
  ParticleVariable<int>    pIsBroken_pg;

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
  old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, 
     pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);

  ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<Point>  pX_p;
  new_dw->get(pX_p, lb->pXXLabel, pset_p);

  vector<int> pIdxEx( pset_p->numParticles() );
  fit(pset_p,pX_p,
      pset_pg,pX_pg,
      pIdxEx);

  Lattice lattice(pX_pg);
  ParticlesNeighbor particles;
  IntVector cellIdx;

  //finding new crackFaces;
  ParticleVariable<Matrix3> pStress_p;
  ParticleVariable<double> pStrainEnergy_p;
  ParticleVariable<double> pToughness_p;
  
  new_dw->get(pStress_p, lb->pStressAfterStrainRateLabel, pset_p);
  new_dw->get(pStrainEnergy_p, lb->pStrainEnergyLabel, pset_p);
  old_dw->get(pToughness_p, lb->pToughnessLabel, pset_p);

  for(ParticleSubset::iterator iter = pset_pg->begin();
          iter != pset_pg->end(); iter++)
  {
    particleIndex pIdx = *iter;
    if(pIsBroken_pg[pIdx] == 0) continue;

    double pHalfSize = pow(pVolume_pg[pIdx],1./3.)/2;
    CrackFace face(pCrackNormal_pg[pIdx],
        pX_pg[pIdx] + pCrackNormal_pg[pIdx] * pHalfSize,
	pHalfSize);
    if(face.isTip(pCrackNormal_pg,
                  pIsBroken_pg,
		  lattice)) lattice.insert(face);
  }
  
  ParticleVariable<Vector> pNewCrackNormal_p_new;
  ParticleVariable<int> pNewIsBroken_p_new;
  
  new_dw->allocate(pNewCrackNormal_p_new, lb->pNewCrackNormalLabel, pset_p);
  new_dw->allocate(pNewIsBroken_p_new, lb->pNewIsBrokenLabel, pset_p);
  
  //break
  for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
  {
    particleIndex pIdx_p = *iter;
    particleIndex pIdx_pg = pIdxEx[pIdx_p];
    
    pNewCrackNormal_p_new[pIdx_p] = 
        pCrackNormal_pg[pIdx_pg];
    pNewIsBroken_p_new[pIdx_p] = 0;
    
    if(pIsBroken_pg[pIdx_pg] == 1) continue;
    
    double area = pow(pVolume_pg[pIdx_pg],2./3.);
    double resistEnergy = pToughness_p[pIdx_p] * area;
    

    //check toughness
    /*
    cout<<"pStrainEnergy/resistEnergy"
        <<pStrainEnergy_p[pIdx_p]/resistEnergy<<endl;
    */
    
    if( resistEnergy > pStrainEnergy_p[pIdx_p] ) continue;

    Point& particlePoint = pX_pg[pIdx_pg];
    
    IntVector thisCellIndex;
    patch->findCell(particlePoint,thisCellIndex);
    CellsNeighbor cells;
    cells.buildIncluding(thisCellIndex,lattice);
        
    CrackFace* crackFace = NULL;
    
    int cellsNum = cells.size();
    for(int i=0;i<cellsNum;++i) {
      cellIdx = cells[i];
      for(list<CrackFace>::iterator fIter=lattice[cellIdx].crackFaces.begin();
         fIter!=lattice[cellIdx].crackFaces.end();
 	 ++fIter)
      {
        if( fIter->atTip(particlePoint) )
	{
	  crackFace = &(*fIter);
	  break;
	}
      }
      if(crackFace) break;
    }
        
    if(crackFace) {
      //cout<<"particle at crack tip"<<endl;
      
      Vector maxDirection;
      double maxStress;
      
/*
      if( crackFace->closeToBoundary(particlePoint,lattice) )
      {
        maxDirection = crackFace->getNormal();
	maxStress = Dot(maxDirection,
	                pStress_p[pIdx_p]*maxDirection);
      }
      else
      {
*/
        double sig[3];
        pStress_p[pIdx_p].getEigenValues(sig[0], sig[1], sig[2]);
        maxStress = sig[0];

        vector<Vector> eigenVectors = pStress_p[pIdx_p].
          getEigenVectors( maxStress,Max(fabs(sig[0]),fabs(sig[1]),fabs(sig[2])) );

        for(int i=0;i<eigenVectors.size();++i) eigenVectors[i].normalize();

        if(eigenVectors.size() == 1) maxDirection = eigenVectors[0];

        else if(eigenVectors.size() == 2) {
          cout<<"eigenVectors.size = 2"<<endl;
          double theta = drand48() * M_PI * 2;
          maxDirection = eigenVectors[0] * cos(theta) + 
                       eigenVectors[1] * sin(theta);
        }

        else if(eigenVectors.size() == 3) {
          cout<<"eigenVectors.size = 3"<<endl;
          double theta = drand48() * M_PI * 2;
          double beta = drand48() * M_PI;
          double cos_beta = cos(beta);
          double sin_beta = sin(beta);
          Vector xy = eigenVectors[2] * sin_beta;
          maxDirection = 
	     eigenVectors[0] * (sin_beta * cos(theta)) +
	     eigenVectors[1] * (sin_beta * sin(theta)) +
	     eigenVectors[2] * cos_beta;
        }
//      }
      
      Vector N = maxDirection;
      if( Dot(crackFace->getTip() - particlePoint, N) < 0 ) N = -N;
      
      double I2 = 0;
      for(int i=1;i<=3;++i)
      for(int j=1;j<=3;++j) {
        double s = pStress_p[pIdx_p](i,j);
        I2 +=  s * s;
      }
      
      double driveEnergy = pStrainEnergy_p[pIdx_p] * 
             maxStress * maxStress / I2;
      //cout<<"driveEnergy/resistEnergy: "<<driveEnergy/resistEnergy<<endl;
      if( driveEnergy < resistEnergy ) continue;
      
      if( lattice.checkPossible(N,
                                 pIdx_pg,
                                 pX_pg,
		  		 pVolume_pg,
				 pCrackNormal_pg,
				 pIsBroken_pg) )
      {
        pNewCrackNormal_p_new[pIdx_p] = N;
        pNewIsBroken_p_new[pIdx_p] = 1;
      }
    }
  }

  new_dw->put(pNewCrackNormal_p_new, lb->pNewCrackNormalLabel);
  new_dw->put(pNewIsBroken_p_new, lb->pNewIsBrokenLabel);
}

void NormalFracture::stressRelease(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  static Vector zero(0.,0.,0.);
  
  //patchAndGhost data
  ParticleVariable<Point>  pX_pg;
  ParticleVariable<Vector> pNewCrackNormal_pg;
  ParticleVariable<int>    pIsBroken_pg;
  ParticleVariable<int>    pNewIsBroken_pg;

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  old_dw->get(pX_pg, lb->pXLabel, pset_pg);
  new_dw->get(pNewCrackNormal_pg, lb->pNewCrackNormalLabel, 
     pset_pg);
  old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, 
     pset_pg);
  new_dw->get(pNewIsBroken_pg, lb->pNewIsBrokenLabel, 
     pset_pg);

  //patchOnly data
  ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<Point>  pX_p;
  ParticleVariable<Matrix3> pStress_p;
  ParticleVariable<Vector> pRotationRate_p;
  ParticleVariable<double> pVolume_p;
  
  old_dw->get(pVolume_p, lb->pVolumeLabel, pset_p);
  new_dw->get(pStress_p, lb->pStressAfterStrainRateLabel, pset_p);
  new_dw->get(pRotationRate_p, lb->pRotationRateLabel, pset_p);
  new_dw->get(pX_p, lb->pXXLabel, pset_p);

  vector<int> pIdxEx( pset_p->numParticles() );
  fit(pset_p,pX_p,
      pset_pg,pX_pg,
      pIdxEx);

  Lattice lattice(pX_pg);
  ParticlesNeighbor particles;
  IntVector cellIdx;

  //Allocate new data
  ParticleVariable<Vector> pCrackNormal_p_new;
  ParticleVariable<int> pIsBroken_p_new;
  ParticleVariable<Matrix3> pStress_p_new;
  
  new_dw->allocate(pCrackNormal_p_new, lb->pCrackNormalLabel, pset_p);
  new_dw->allocate(pIsBroken_p_new, lb->pIsBrokenLabel, pset_p);
  new_dw->allocate(pStress_p_new, lb->pStressLabel, pset_p);
  
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
  {
    particleIndex pIdx_p = *iter;
    particleIndex pIdx_pg = pIdxEx[pIdx_p];

    if( pIsBroken_pg[pIdx_pg] != 0 ) {
      pCrackNormal_p_new[pIdx_p] = 
        pNewCrackNormal_pg[pIdx_pg] +
        Cross( pRotationRate_p[pIdx_p] * delT, 
	       pNewCrackNormal_pg[pIdx_pg] );

      /*
      cout<<"===="<<endl;
      cout<<pIdx_p<<endl;
      cout<<pIdx_pg<<endl;
      cout<<pNewCrackNormal_pg[pIdx_pg]<<endl;
      cout<<pRotationRate_p[pIdx_p] * delT<<endl;
      cout<<pCrackNormal_p_new[pIdx_p]<<endl;
      */
      
      pCrackNormal_p_new[pIdx_p].normalize();

      pIsBroken_p_new[pIdx_p] = 
        pIsBroken_pg[pIdx_pg];
      pStress_p_new[pIdx_p] = 
        pStress_p[pIdx_p];
      continue;
    }
    
    if( pNewIsBroken_pg[pIdx_pg] == 0 ) {
      pCrackNormal_p_new[pIdx_p] = zero;
      pIsBroken_p_new[pIdx_p] = 0;
      pStress_p_new[pIdx_p] = 
        pStress_p[pIdx_p];
      continue;
    }

    double r = pow(pVolume_p[pIdx_p],1./3.)/2;
    if(lattice.checkPossible(
                   pIdx_pg,
		   r,
                   pX_pg,
                   pNewCrackNormal_pg,
                   pIsBroken_pg,
                   pNewIsBroken_pg ) )
    {
      cout<<"crack grow!"<<endl;
      const Vector& N = pNewCrackNormal_pg[pIdx_pg];
      double maxStress = Dot(N,pStress_p[pIdx_p] * N);
      Matrix3 stress;
      for(int i=1;i<=3;++i)
      for(int j=1;j<=3;++j) {
        stress(i,j) = N(i-1) * maxStress * N(j-1);
      }
      pStress_p_new[pIdx_p] = 
        pStress_p[pIdx_p] - stress;
      pIsBroken_p_new[pIdx_p] = 1;
      pCrackNormal_p_new[pIdx_p] = N;
    }
    else {
      pCrackNormal_p_new[pIdx_p] = zero;
      pIsBroken_p_new[pIdx_p] = 0;
      pStress_p_new[pIdx_p] = 
        pStress_p[pIdx_p];
    }
  }

  new_dw->put(pStress_p_new, lb->pStressAfterFractureReleaseLabel);
  new_dw->put(pCrackNormal_p_new, lb->pCrackNormalLabel_preReloc);
  new_dw->put(pIsBroken_p_new, lb->pIsBrokenLabel_preReloc);
}

NormalFracture::
NormalFracture(ProblemSpecP& ps) : Fracture(ps)
{
}

NormalFracture::~NormalFracture()
{
}

} //namespace Uintah
