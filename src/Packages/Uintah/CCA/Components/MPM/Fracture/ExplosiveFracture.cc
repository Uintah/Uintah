#include "ExplosiveFracture.h"

#include "ParticlesNeighbor.h"
#include "Visibility.h"

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <stdlib.h>
#include <list>

namespace Uintah {
using namespace SCIRun;

void
ExplosiveFracture::
initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw)
{
}

void ExplosiveFracture::computeNodeVisibility(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* insidePset = old_dw->getParticleSubset(matlindex, patch);

  ParticleVariable<int>    pVisibility;
  new_dw->allocate(pVisibility, lb->pVisibilityLabel, insidePset);

  for(ParticleSubset::iterator iter = insidePset->begin();
          iter != insidePset->end(); iter++)
  {
    particleIndex pIdx = *iter;
    Visibility vis;
    pVisibility[pIdx] = vis.flag();
  }
  
  new_dw->put(pVisibility, lb->pVisibilityLabel);
}

void
ExplosiveFracture::
crackGrow(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
   ParticleVariable<Matrix3> pStress;
   ParticleVariable<double> pTensileStrength;
   ParticleVariable<int> pIsNewlyBroken;
   ParticleVariable<Vector> pNewlyBrokenSurfaceNormal;

   int matlindex = mpm_matl->getDWIndex();
   ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

   old_dw->get(pTensileStrength, lb->pTensileStrengthLabel, pset);
   new_dw->get(pStress, lb->pStressAfterStrainRateLabel, pset);   
   new_dw->allocate(pIsNewlyBroken, lb->pIsNewlyBrokenLabel, pset);
   new_dw->allocate(pNewlyBrokenSurfaceNormal, 
     lb->pNewlyBrokenSurfaceNormalLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin(); 
       iter != pset->end(); iter++)
   {
     pIsNewlyBroken[*iter] = 0;
     pNewlyBrokenSurfaceNormal[*iter] = Vector(0.,0.,0.);
   }

   for(ParticleSubset::iterator iter = pset->begin(); 
       iter != pset->end(); iter++)
   {
     particleIndex idx = *iter;

     //get the max stress
     double sig[3];
     pStress[idx].getEigenValues(sig[0], sig[1], sig[2]);
     double maxStress = sig[0];
               
     //compare with the tensile strength
     if(maxStress <= pTensileStrength[idx]) continue;

     vector<Vector> eigenVectors = pStress[idx].
        getEigenVectors(maxStress,maxStress);

     for(int i=0;i<(int)eigenVectors.size();++i) {
         eigenVectors[i].normalize();
     }
	
     Vector maxDirection;
     if(eigenVectors.size() == 1) {
         maxDirection = eigenVectors[0];
     }
       
     if(eigenVectors.size() == 2) {
         cout<<"eigenVectors.size = 2"<<endl;
         double theta = drand48() * M_PI * 2;
         maxDirection = (eigenVectors[0] * cos(theta) + eigenVectors[1] * sin(theta));
     }
       
     if(eigenVectors.size() == 3) {
         cout<<"eigenVectors.size = 3"<<endl;
         double theta = drand48() * M_PI * 2;
         double beta = drand48() * M_PI;
         double cos_beta = cos(beta);
         double sin_beta = sin(beta);
	 // Unused variable - steve
         // Vector xy = eigenVectors[2] * sin_beta;
         maxDirection = eigenVectors[0] * (sin_beta * cos(theta)) +
	                eigenVectors[1] * (sin_beta * sin(theta)) +
		  	eigenVectors[2] * cos_beta;
     }

     pNewlyBrokenSurfaceNormal[idx] = maxDirection;
     pIsNewlyBroken[idx] = 1;

     cout<<"Crack nucleated."<<endl;
   }
      
   new_dw->put(pIsNewlyBroken, lb->pIsNewlyBrokenLabel);
   new_dw->put(pNewlyBrokenSurfaceNormal, lb->pNewlyBrokenSurfaceNormalLabel);
   new_dw->put(pTensileStrength, lb->pTensileStrengthLabel_preReloc);
}

void
ExplosiveFracture::
stressRelease(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  //patch + ghost variables
  ParticleVariable<Point> pX;
  ParticleVariable<int> pIsNewlyBroken;
  ParticleVariable<Vector> pNewlyBrokenSurfaceNormal;

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* outsidePset = old_dw->getParticleSubset(matlindex, patch,
	Ghost::AroundNodes, 1, lb->pXLabel);

  old_dw->get(pX, lb->pXLabel, outsidePset);
  new_dw->get(pIsNewlyBroken, lb->pIsNewlyBrokenLabel, outsidePset);
  new_dw->get(pNewlyBrokenSurfaceNormal, lb->pNewlyBrokenSurfaceNormalLabel, outsidePset);

  Lattice lattice(pX);

  //patch variables
  ParticleVariable<Matrix3> pStress;
  ParticleVariable<int> pIsBroken;
  ParticleVariable<Vector> pCrackNormal;
  ParticleVariable<double> pMass;
  ParticleVariable<double> pStrainEnergy;
  ParticleVariable<Vector> pImageVelocity;

  ParticleSubset* insidePset = old_dw->getParticleSubset(matlindex, patch);
  new_dw->get(pStress, lb->pStressAfterStrainRateLabel, insidePset);
  old_dw->get(pIsBroken, lb->pIsBrokenLabel, insidePset);
  old_dw->get(pCrackNormal, lb->pCrackNormalLabel, insidePset);

  old_dw->get(pMass, lb->pMassLabel, insidePset);
  new_dw->get(pStrainEnergy, lb->pStrainEnergyLabel, insidePset);
  old_dw->get(pImageVelocity, lb->pImageVelocityLabel, insidePset);

  ParticleVariable<int> pStressReleased;
  new_dw->allocate(pStressReleased, lb->pStressReleasedLabel, insidePset);

  double delTAfterFracture = 1.e12;

  for(ParticleSubset::iterator iter = insidePset->begin(); 
     iter != insidePset->end(); iter++)
  {
    pStressReleased[*iter] = 0;
    pImageVelocity[*iter] = Vector(0,0,0);
  }
  
  double range = ( patch->dCell().x() + 
	           patch->dCell().y() + 
		   patch->dCell().z() )/3;

  for(ParticleSubset::iterator iter = outsidePset->begin(); 
     iter != outsidePset->end(); iter++)
  {
    particleIndex idx = *iter;
        
    if( pStressReleased[idx] == 1 ) continue;
    if(pIsNewlyBroken[idx]) {
      IntVector cellIdx;
      patch->findCell(pX[idx],cellIdx);
      ParticlesNeighbor particlesNeighbor;
      particlesNeighbor.buildIn(cellIdx,lattice);
      
      Vector N = pCrackNormal[idx];
      particleIndex pairIdx = -1;
      double pairRatio = 0;
      double maxStress = Dot(pStress[idx]*N,N);
      for(std::vector<particleIndex>::const_iterator ip = particlesNeighbor.begin();
	       ip != particlesNeighbor.end(); ++ip)
      {
        particleIndex pNeighbor = *ip;	

	if( pStressReleased[pNeighbor] == 1 ) continue;
        if(pNeighbor == idx) continue;
        if( !patch->findCell(pX[pNeighbor],cellIdx) ) continue;	
        Vector dis = (pX[idx] - pX[pNeighbor]);
        double d = dis.length();
        if( d > range ) continue;
        if( fabs(Dot(dis,N))/d < 0.5 ) continue;
        
        double ratio = Dot(pStress[pNeighbor]*N,N) / maxStress;
        if(ratio > pairRatio) {
          pairIdx = pNeighbor;
          pairRatio = ratio;
        }
      }
      if(pairIdx == -1) continue;
      cout<<"pairRatio:"<<pairRatio<<endl;
      if( Dot( (pX[pairIdx]-pX[idx]), N ) < 0 ) N = -N;
		
      //Matrix3 stress;
      double I2,sRelease;
      double v;
      double alpha = 1;
      double timeScale = 0.01;
      
      I2 = 0;
      sRelease = maxStress;
      for(int i=1;i<=3;++i)
      for(int j=1;j<=3;++j) {
        I2 += pStress[idx](i,j) * pStress[idx](i,j);
	//stress(i,j) = N(i-1) * sRelease * N(j-1);
      }
      v = sqrt( pStrainEnergy[idx] * sRelease * sRelease / I2 * 2 / pMass[idx] ) 
         * alpha;
      pImageVelocity[idx] -= (N * v);
      //pStress[idx] -= stress;
      delTAfterFracture = Min(delTAfterFracture, range/v/2*timeScale);
      pStressReleased[idx] = 1;
      
      I2 = 0;
      sRelease = maxStress * pairRatio;
      for(int i=1;i<=3;++i)
      for(int j=1;j<=3;++j) {
        I2 += pStress[pairIdx](i,j) * pStress[pairIdx](i,j);
	//stress(i,j) = N(i-1) * sRelease * N(j-1);
      }
      v = sqrt( pStrainEnergy[pairIdx] * sRelease * sRelease / I2 * 2 / pMass[pairIdx] ) 
         * alpha;
      pImageVelocity[pairIdx] += (N * v);
      //pStress[pairIdx] -= stress;
      delTAfterFracture = Min(delTAfterFracture, range/v/2*timeScale);
      pStressReleased[pairIdx] = 1;      
    }
  }

/*
  for(ParticleSubset::iterator iter = outsidePset->begin(); 
     iter != outsidePset->end(); iter++)
  {
    pImageVelocity[*iter].z(0);
  }
*/
  
  new_dw->put(pCrackNormal, lb->pCrackNormalLabel_preReloc);
  new_dw->put(pIsBroken, lb->pIsBrokenLabel_preReloc);
  new_dw->put(pStress, lb->pStressAfterFractureReleaseLabel);
  new_dw->put(pImageVelocity, lb->pImageVelocityLabel_preReloc);

  new_dw->put(delt_vartype(delTAfterFracture), lb->delTLabel);
}

ExplosiveFracture::
ExplosiveFracture(ProblemSpecP& ps)
: Fracture(ps)
{
}

} // End namespace Uintah
