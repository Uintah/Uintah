#include "SimpleFracture.h"

#include "ParticlesNeighbor.h"
#include "Visibility.h"

#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/VarTypes.h>

#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/CellIterator.h>

#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <stdlib.h>
#include <list>

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

void
SimpleFracture::
initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouseP& new_dw)
{
}

void SimpleFracture::computeNodeVisibility(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  // Create arrays for the particle data
  ParticleVariable<Point>  pX;
  ParticleVariable<double> pVolume;
  ParticleVariable<Vector> pCrackSurfaceNormal;
  ParticleVariable<int>    pIsBroken;

  int matlindex = mpm_matl->getDWIndex();

  ParticleSubset* pset_patchAndGhost = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundNodes, 1, lb->pXLabel);

  old_dw->get(pX, lb->pXLabel, pset_patchAndGhost);
  old_dw->get(pVolume, lb->pVolumeLabel, pset_patchAndGhost);
  old_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, 
     pset_patchAndGhost);
  old_dw->get(pIsBroken, lb->pIsBrokenLabel, pset_patchAndGhost);

  ParticleSubset* pset_patchOnly = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<int>    pVisibility;
  new_dw->allocate(pVisibility, lb->pVisibilityLabel, pset_patchOnly);

  Lattice lattice(pX);
  ParticlesNeighbor particles;
  IntVector cellIdx;
  IntVector nodeIdx[8];

  for(ParticleSubset::iterator iter = pset_patchOnly->begin();
          iter != pset_patchOnly->end(); iter++)
  {
    particleIndex pIdx = *iter;
    patch->findCell(pX[pIdx],cellIdx);
    particles.clear();
    particles.buildIn(cellIdx,lattice);
    
    //visibility
    patch->findNodesFromCell(cellIdx,nodeIdx);
    
    Visibility vis;
    //cout<<"point:"<<pX[pIdx]<<endl;
    for(int i=0;i<8;++i) {
      //cout<<"node "<<i<<":"<<patch->nodePosition(nodeIdx[i])<<endl;
      if(particles.visible( pIdx,
                            patch->nodePosition(nodeIdx[i]),
		            pX,
		            pIsBroken,
		            pCrackSurfaceNormal,
		            pVolume) ) vis.setVisible(i);
      else vis.setUnvisible(i);
    }
    pVisibility[pIdx] = vis.flag();
  }
  
  new_dw->put(pVisibility, lb->pVisibilityLabel);
}

void
SimpleFracture::
crackGrow(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
   ParticleVariable<Matrix3> pStress;
   ParticleVariable<int> pIsBroken;
   ParticleVariable<double> pTensileStrength;

   ParticleVariable<int> pIsNewlyBroken;
   ParticleVariable<Vector> pNewlyBrokenSurfaceNormal;

   int matlindex = mpm_matl->getDWIndex();
   ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

   new_dw->get(pStress, lb->pStressAfterStrainRateLabel, pset);
   old_dw->get(pIsBroken, lb->pIsBrokenLabel, pset);
   old_dw->get(pTensileStrength, lb->pTensileStrengthLabel, pset);
   
   new_dw->allocate(pIsNewlyBroken, lb->pIsNewlyBrokenLabel, pset);
   new_dw->allocate(pNewlyBrokenSurfaceNormal, 
     lb->pNewlyBrokenSurfaceNormalLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin(); 
       iter != pset->end(); iter++)
   {
      particleIndex idx = *iter;
      pIsNewlyBroken[idx] = 0;
      pNewlyBrokenSurfaceNormal[idx] = Vector(0.,0.,0.);
      
      //label out the broken particles
      if(pIsBroken[idx]) continue;

      //get the max stress
      double sig[3];
      Matrix3 stre = pStress[idx];
      stre(1,3) = 0;
      stre(3,1) = 0;
      stre(2,3) = 0;
      stre(3,2) = 0;
      stre(3,3) = 0;
      
      //pStress[idx].getEigenValues(sig[0], sig[1], sig[2]);
      stre.getEigenValues(sig[0], sig[1], sig[2]);
      double maxStress = sig[0];
	
      //compare with the tensile strength
      if(maxStress < pTensileStrength[idx]) continue;
      
      //cout<<pStress[idx]<<endl;
      
      //vector<Vector> eigenVectors = pStress[idx].getEigenVectors(maxStress,
	// fabs(maxStress));
      vector<Vector> eigenVectors = stre.getEigenVectors(maxStress,
	 fabs(maxStress));

      //cout<<"eigenVectorsSize: "<<eigenVectors.size()<<endl;
      for(int i=0;i<eigenVectors.size();++i) {
        //cout<<"eigenVectors: "<<eigenVectors[i]<<endl;
        eigenVectors[i].normalize();
      }
	
      Vector maxDirection;
      if(eigenVectors.size() == 1) maxDirection = eigenVectors[0];

      if(eigenVectors.size() == 2) {
	cout<<"eigenVectors.size = 2"<<endl;
	double theta = drand48() * M_PI * 2;
	maxDirection = ( eigenVectors[0] * cos(theta) + 
	                 eigenVectors[1] * sin(theta));
      }
	
      if(eigenVectors.size() == 3) {
	cout<<"eigenVectors.size = 3"<<endl;
	double theta = drand48() * M_PI * 2;
	double beta = drand48() * M_PI;
 	double cos_beta = cos(beta);
	double sin_beta = sin(beta);
	Vector xy = eigenVectors[2] * sin_beta;
	maxDirection = eigenVectors[0] * (sin_beta * cos(theta)) +
	               eigenVectors[1] * (sin_beta * sin(theta)) +
	               eigenVectors[2] * cos_beta;
      }

      //cout<<"Crack nucleated in direction: "<<maxDirection<<"."<<endl;
      //if(drand48()>0.5) maxDirection = -maxDirection;
      pNewlyBrokenSurfaceNormal[idx] = maxDirection;
      pIsNewlyBroken[idx] = 1;

      cout<<"Crack nucleated in direction: "<<maxDirection<<"."<<endl;
   }
      
   new_dw->put(pIsNewlyBroken, lb->pIsNewlyBrokenLabel);
   new_dw->put(pNewlyBrokenSurfaceNormal, lb->pNewlyBrokenSurfaceNormalLabel);
   new_dw->put(pTensileStrength, lb->pTensileStrengthLabel_preReloc);
}

void
SimpleFracture::
stressRelease(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  //patch + ghost variables
  ParticleVariable<Point> pX;
  ParticleVariable<int> pIsNewlyBroken;
  ParticleVariable<Vector> pNewlyBrokenSurfaceNormal;
  ParticleVariable<Matrix3> pStress;

  int matlindex = mpm_matl->getDWIndex();

  ParticleSubset* pset_patchAndGhost = old_dw->getParticleSubset(
     matlindex, patch, Ghost::AroundNodes, 1, lb->pXLabel);

  old_dw->get(pX, lb->pXLabel, pset_patchAndGhost);
  new_dw->get(pIsNewlyBroken, lb->pIsNewlyBrokenLabel, pset_patchAndGhost);
  new_dw->get(pNewlyBrokenSurfaceNormal, lb->pNewlyBrokenSurfaceNormalLabel, 
     pset_patchAndGhost);
  new_dw->get(pStress, lb->pStressAfterStrainRateLabel, pset_patchAndGhost);

  Lattice lattice(pX);

  //patch variables
  ParticleSubset* pset_patchOnly = old_dw->getParticleSubset(
     matlindex, patch);

  ParticleVariable<int> pIsBroken;
  ParticleVariable<Vector> pCrackSurfaceNormal;
  ParticleVariable<Vector> pImageVelocity;
  ParticleVariable<Vector> pVelocity;

  ParticleVariable<double> pMass;
  ParticleVariable<double> pStrainEnergy;
  ParticleVariable<Vector> pRotationRate;

  old_dw->get(pIsBroken, lb->pIsBrokenLabel, pset_patchOnly);
  old_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, pset_patchOnly);
  old_dw->get(pImageVelocity, lb->pImageVelocityLabel, pset_patchOnly);
  new_dw->get(pVelocity, lb->pVelocityAfterUpdateLabel, pset_patchOnly);

  old_dw->get(pMass, lb->pMassLabel, pset_patchOnly);
  new_dw->get(pStrainEnergy, lb->pStrainEnergyLabel, pset_patchOnly);
  new_dw->get(pRotationRate, lb->pRotationRateLabel, pset_patchOnly);
  
  ParticleVariable<int> pStressReleased;
  new_dw->allocate(pStressReleased, lb->pStressReleasedLabel, pset_patchOnly);
  
  ParticleVariable<Matrix3> pStress_new;
  ParticleVariable<int> pIsBroken_new;
  ParticleVariable<Vector> pCrackSurfaceNormal_new;
  ParticleVariable<Vector> pImageVelocity_new;
  ParticleVariable<Vector> pVelocity_new;

  new_dw->allocate(pStress_new, lb->pStressLabel, pset_patchOnly);
  new_dw->allocate(pIsBroken_new, lb->pIsBrokenLabel, pset_patchOnly);
  new_dw->allocate(pCrackSurfaceNormal_new, lb->pCrackSurfaceNormalLabel, 
     pset_patchOnly);
  new_dw->allocate(pImageVelocity_new, lb->pImageVelocityLabel, pset_patchOnly);
  new_dw->allocate(pVelocity_new, lb->pVelocityLabel, pset_patchOnly);
  
  
  double range = ( patch->dCell().x() + 
	           patch->dCell().y() + 
		   patch->dCell().z() )/3;

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  IntVector cellIdx;

  //update for broken particles,crack surface rotation
  for(ParticleSubset::iterator iter = pset_patchOnly->begin(); 
     iter != pset_patchOnly->end(); iter++)
  {
    particleIndex idx = *iter;

    pStressReleased[idx] = 0;

    pStress_new[idx] = pStress[idx];
    pIsBroken_new[idx] = pIsBroken[idx];
    pImageVelocity_new[idx] = pImageVelocity[idx];
    pVelocity_new[idx] = pVelocity[idx];

    if(pIsBroken[idx]) {
      pCrackSurfaceNormal_new[idx] = pCrackSurfaceNormal[idx] +
                                  Cross( pRotationRate[idx] * delT, 
	                                 pCrackSurfaceNormal[idx] );
      pCrackSurfaceNormal_new[idx].normalize();
    }
  }

  delt_vartype delTAfterConstitutiveModel;
  new_dw->get(delTAfterConstitutiveModel, lb->delTAfterConstitutiveModelLabel);
  double delTAfterFracture = delTAfterConstitutiveModel;
  
  for(ParticleSubset::iterator iter = pset_patchAndGhost->begin(); 
     iter != pset_patchAndGhost->end(); iter++)
  {
    particleIndex idx = *iter;
        
    if( pStressReleased[idx] == 1 ) continue;
    if(!pIsNewlyBroken[idx]) continue;
      
    bool insidePatch,pairInsidePatch;
    insidePatch = patch->findCell(pX[idx],cellIdx);
    ParticlesNeighbor particlesNeighbor;
    particlesNeighbor.buildIn(cellIdx,lattice);
      
    Vector N = pNewlyBrokenSurfaceNormal[idx];

    particleIndex pairIdx = -1;
    double pairRatio = 0;
    double maxStress = Dot(pStress[idx]*N,N);
    for(std::vector<particleIndex>::const_iterator 
       ip = particlesNeighbor.begin();
       ip != particlesNeighbor.end(); ++ip)
    {
      particleIndex pNeighbor = *ip;	

      if( pStressReleased[pNeighbor] == 1 ) continue;
      if( pIsBroken[pNeighbor] == 1 ) continue;
      if(pNeighbor == idx) continue;
      if( !patch->findCell(pX[pNeighbor],cellIdx) ) continue;	
      Vector dis = (pX[idx] - pX[pNeighbor]);
      double d = dis.length();
      if( d > range ) continue;
      if( fabs(Dot(dis,N))/d < sqrt(2.)/2 ) continue;
        
      double ratio = Dot(pStress[pNeighbor]*N,N) / maxStress;
      if(ratio > pairRatio) {
        pairIdx = pNeighbor;
        pairRatio = ratio;
      }
    }
    if(pairIdx == -1) continue;
    pairInsidePatch = patch->findCell(pX[pairIdx],cellIdx);

    if( Dot( (pX[pairIdx]-pX[idx]), N ) < 0 ) N = -N;
    cout<<"pairRatio:"<<pairRatio<<endl;

    Matrix3 stress;
    double I2,sRelease;
    double v;
      
    if(insidePatch) {
      I2 = 0;
      sRelease = maxStress;
      for(int i=1;i<=3;++i)
      for(int j=1;j<=3;++j) {
        I2 += pStress[idx](i,j) * pStress[idx](i,j);
        stress(i,j) = N(i-1) * sRelease * N(j-1);
      }
      v = sqrt( pStrainEnergy[idx] * sRelease * sRelease / I2 / pMass[idx] );
      pVelocity_new[idx] -= (N * v);
      pStress_new[idx] -= stress;
      pCrackSurfaceNormal_new[idx] = N;
      pIsBroken_new[idx] = 1;
      if(v>0) delTAfterFracture = Min(delTAfterFracture, range/v/2);
      pStressReleased[idx] = 1;
    }

    if(pairInsidePatch) {
      I2 = 0;
      sRelease = maxStress * pairRatio;
      for(int i=1;i<=3;++i)
      for(int j=1;j<=3;++j) {
        I2 += pStress_new[pairIdx](i,j) * pStress_new[pairIdx](i,j);
        stress(i,j) = N(i-1) * sRelease * N(j-1);
      }
      v = sqrt( pStrainEnergy[pairIdx] * sRelease * sRelease / I2 / 
         pMass[pairIdx] );
      pVelocity_new[pairIdx] += (N * v);
      pStress_new[pairIdx] -= stress;
      pCrackSurfaceNormal_new[pairIdx] = -N;
      pIsBroken_new[pairIdx] = 1;
      if(v>0) delTAfterFracture = Min(delTAfterFracture, range/v/2);
      pStressReleased[pairIdx] = 1;
    }
  }

  /*
  for(ParticleSubset::iterator iter = pset->begin(); 
     iter != pset->end(); iter++)
  {
    pVelocity_new[*iter].z(0);
  }
  */
  
  new_dw->put(pCrackSurfaceNormal_new, lb->pCrackSurfaceNormalLabel_preReloc);
  new_dw->put(pIsBroken_new, lb->pIsBrokenLabel_preReloc);
  new_dw->put(pStress_new, lb->pStressAfterFractureReleaseLabel);
  new_dw->put(pImageVelocity_new, lb->pImageVelocityLabel_preReloc);
  new_dw->put(pVelocity_new, lb->pVelocityAfterFractureLabel);

  new_dw->put(delt_vartype(delTAfterFracture), lb->delTAfterFractureLabel);
}

SimpleFracture::
SimpleFracture(ProblemSpecP& ps)
: Fracture(ps)
{
}

} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.1  2000/11/21 20:46:28  tan
// Implemented different models for fracture simulations.  SimpleFracture model
// is for the simulation where the resolution focus only on macroscopic major
// cracks. NormalFracture and ExplosionFracture models are more sophiscated
// and specific fracture models that are currently underconstruction.
//
