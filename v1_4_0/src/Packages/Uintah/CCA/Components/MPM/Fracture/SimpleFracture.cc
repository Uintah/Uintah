#include "SimpleFracture.h"

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
SimpleFracture::
initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw)
{
}

void SimpleFracture::computeNodeVisibility(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  // Create arrays for the particle data
  ParticleVariable<Point>  pX_patchAndGhost;
  ParticleVariable<double> pVolume;
  ParticleVariable<Vector> pCrackNormal;
  ParticleVariable<int>    pIsBroken;

  int matlindex = mpm_matl->getDWIndex();

  ParticleSubset* pset_patchAndGhost = old_dw->getParticleSubset(matlindex, 
     patch, Ghost::AroundCells, 1, lb->pXLabel);

  old_dw->get(pX_patchAndGhost, lb->pXLabel, pset_patchAndGhost);
  old_dw->get(pVolume, lb->pVolumeLabel, pset_patchAndGhost);
  old_dw->get(pCrackNormal, lb->pCrackNormalLabel, 
     pset_patchAndGhost);
  old_dw->get(pIsBroken, lb->pIsBrokenLabel, pset_patchAndGhost);

  ParticleSubset* pset_patchOnly = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<int>    pVisibility;
  new_dw->allocate(pVisibility, lb->pVisibilityLabel, pset_patchOnly);
  ParticleVariable<Point>  pX_patchOnly;
  new_dw->get(pX_patchOnly, lb->pXXLabel, pset_patchOnly);

  vector<int> particleIndexExchange( pset_patchOnly->numParticles() );
  fit(pset_patchOnly,pX_patchOnly,
      pset_patchAndGhost,pX_patchAndGhost,
      particleIndexExchange);

  Lattice lattice(pX_patchAndGhost);
  ParticlesNeighbor particles;
  IntVector cellIdx;
  IntVector nodeIdx[8];
  
  /*
  cout<<"number of particles: "<<pset_patchOnly->numParticles()<<endl;
  cout<<"begin: "<<*pset_patchOnly->begin()<<endl;
  cout<<"end: "<<*(pset_patchOnly->end()-1)<<endl;
  */

  for(ParticleSubset::iterator iter = pset_patchOnly->begin();
          iter != pset_patchOnly->end(); iter++)
  {
    particleIndex pIdx = *iter;
    patch->findCell(pX_patchOnly[pIdx],cellIdx);
    particles.clear();
    particles.buildIn(cellIdx,lattice);
    
    //visibility
    patch->findNodesFromCell(cellIdx,nodeIdx);
    
    Visibility vis;
    for(int i=0;i<8;++i) {
      if(particles.visible( particleIndexExchange[pIdx],
                            patch->nodePosition(nodeIdx[i]),
		            pX_patchAndGhost,
		            pIsBroken,
		            pCrackNormal,
		            pVolume) ) vis.setVisible(i);
      else {
        vis.setUnvisible(i);
	/*
	if(patch->nodePosition(nodeIdx[i]).x()!=0) {
        cout<<"point:"<<pX[pIdx]
	    <<"node:"<<i<<":"<<patch->nodePosition(nodeIdx[i])<<endl;
	    }
	*/
      }
    }
    pVisibility[pIdx] = vis.flag();
  }
  
  new_dw->put(pVisibility, lb->pVisibilityLabel);

#if 0
  for(ParticleSubset::iterator iter = pset_patchAndGhost->begin();
          iter != pset_patchAndGhost->end(); iter++)
  {
    particleIndex pIdx = *iter;
    if(!pIsBroken[idx]) continue;
  }
#endif

}

void
SimpleFracture::
crackGrow(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
    int matlindex = mpm_matl->getDWIndex();

    NCVariable<double> gMass;
    NCVariable<Matrix3> gStress;
    NCVariable<double> gTensileStrength;
    new_dw->get(gMass,  lb->gMassLabel,
       matlindex, patch, Ghost::None, 0);
    new_dw->get(gStress, lb->gStressForSavingLabel,
       matlindex, patch, Ghost::None, 0);
    new_dw->get(gTensileStrength, lb->gTensileStrengthLabel, 
       matlindex, patch, Ghost::None, 0);

    NCVariable<Vector> gCrackNormal;
    new_dw->allocate(gCrackNormal, lb->gCrackNormalLabel,
       matlindex, patch);
    
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++)
    {
      if(gMass[*iter]>0.0) {
        //get the max stress
        double sig[3];
	gStress[*iter].getEigenValues(sig[0], sig[1], sig[2]);
        double maxStress = sig[0];
	
        //compare with the tensile strength
        if(maxStress < gTensileStrength[*iter]) {
          gCrackNormal[*iter] = Vector(0.,0.,0.);
	  continue;
	}
      
        vector<Vector> eigenVectors = gStress[*iter].getEigenVectors(maxStress,
	   fabs(maxStress));

        for(int i=0;i<(int)eigenVectors.size();++i) eigenVectors[i].normalize();

        if(eigenVectors.size() == 1) 
	  gCrackNormal[*iter] = eigenVectors[0];

        else if(eigenVectors.size() == 2) {
	  cout<<"eigenVectors.size = 2"<<endl;
  	  double theta = drand48() * M_PI * 2;
	  gCrackNormal[*iter] = 
	    ( eigenVectors[0] * cos(theta) + 
	      eigenVectors[1] * sin(theta));
        }
	
        else if(eigenVectors.size() == 3) {
	  cout<<"eigenVectors.size = 3"<<endl;
  	  double theta = drand48() * M_PI * 2;
	  double beta = drand48() * M_PI;
 	  double cos_beta = cos(beta);
	  double sin_beta = sin(beta);
	  // Unused - Steve
	  //Vector xy = eigenVectors[2] * sin_beta;
	  gCrackNormal[*iter] = 
	     eigenVectors[0] * (sin_beta * cos(theta)) +
	     eigenVectors[1] * (sin_beta * sin(theta)) +
	     eigenVectors[2] * cos_beta;
        }
	//cout<<"crack open in direction "<<gCrackNormal[*iter]<<endl;
      }
    }
    new_dw->put(gCrackNormal, lb->gCrackNormalLabel, 
       matlindex, patch);
}

void
SimpleFracture::
stressRelease(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  int matlindex = mpm_matl->getDWIndex();
  
  NCVariable<Vector> gCrackNormal;
  new_dw->get(gCrackNormal, lb->gCrackNormalLabel,
	matlindex, patch, Ghost::AroundCells, 1);

  ParticleSubset* pset = old_dw->getParticleSubset(
     matlindex, patch);

  ParticleVariable<Point> pX;
  ParticleVariable<double> pMass;
  ParticleVariable<double> pStrainEnergy;
  ParticleVariable<Vector> pRotationRate;
  ParticleVariable<int> pVisibility;

  ParticleVariable<Vector> pVelocity;
  ParticleVariable<Matrix3> pStress;

  ParticleVariable<int> pIsBroken;
  ParticleVariable<int> pIsBroken_new;
  ParticleVariable<Vector> pCrackNormal;
  ParticleVariable<Vector> pCrackNormal_new;

  old_dw->get(pX, lb->pXLabel, pset);
  old_dw->get(pMass, lb->pMassLabel, pset);
  new_dw->get(pStrainEnergy, lb->pStrainEnergyLabel, pset);
  new_dw->get(pRotationRate, lb->pRotationRateLabel, pset);
  new_dw->get(pVisibility, lb->pVisibilityLabel, pset);

  new_dw->get(pVelocity, lb->pVelocityAfterUpdateLabel, pset);
  new_dw->get(pStress, lb->pStressAfterStrainRateLabel, pset);

  old_dw->get(pIsBroken, lb->pIsBrokenLabel, pset);
  old_dw->get(pCrackNormal, lb->pCrackNormalLabel, pset);
  new_dw->allocate(pIsBroken_new, lb->pIsBrokenLabel, pset);
  new_dw->allocate(pCrackNormal_new, lb->pCrackNormalLabel, pset);

  IntVector ni[8];
  Vector zero(0.,0.,0.);

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++)
  {
    particleIndex pIdx = *iter;
    
    pIsBroken_new[pIdx] = pIsBroken[pIdx];
    pCrackNormal_new[pIdx] = pCrackNormal[pIdx];
    
    if(pIsBroken[pIdx]) {
      pCrackNormal_new[pIdx] += Cross( pRotationRate[pIdx] * delT, 
	                                     pCrackNormal[pIdx] );
      pCrackNormal_new[pIdx].normalize();
      continue;
    }

    Visibility vis;
    vis = pVisibility[pIdx];

    IntVector nodeIdx;
    if( !vis.visible( patch->findClosestNode(pX[pIdx], nodeIdx) ) ) continue;
    /*
    cout<<"particle: "<<pX[pIdx]
        <<" node: "<<patch->nodePosition(nodeIdx)<<endl;
    */
    
    const Vector& N = gCrackNormal[nodeIdx];
    if(N == zero) continue;
    
    pIsBroken_new[pIdx] = 1;
    Vector d = patch->nodePosition(nodeIdx) - pX[pIdx];
    if( Dot(d,N) > 0 ) pCrackNormal_new[pIdx] = N;
    else pCrackNormal_new[pIdx] = -N;
  }

  /*
  double delT_new = delT;
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
  */

  new_dw->put(pCrackNormal_new, lb->pCrackNormalLabel_preReloc);
  new_dw->put(pIsBroken_new, lb->pIsBrokenLabel_preReloc);

  new_dw->put(pStress, lb->pStressAfterFractureReleaseLabel);
  new_dw->put(pVelocity, lb->pVelocityAfterFractureLabel);
}

SimpleFracture::
SimpleFracture(ProblemSpecP& ps)
: Fracture(ps)
{
}

} //namespace Uintah
