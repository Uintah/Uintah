#include "NormalFracture.h"

#include "ParticlesNeighbor.h"
#include "Connectivity.h"
#include "CellsNeighbor.h"
#include "IndexExchange.h"
#include "SurfaceCouples.h"

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
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  int matlindex = mpm_matl->getDWIndex();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
      patch, Ghost::AroundCells, 1, lb->pXLabel);

    //patchAndGhost data
    constParticleVariable<Point>  pX_pg;
    constParticleVariable<double> pVolume_pg;
    constParticleVariable<int>    pIsBroken_pg;
    constParticleVariable<Vector> pCrackNormal_pg;
    constParticleVariable<Vector> pVelocity_pg;
    constParticleVariable<double> pMass_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
    old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
    old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
    old_dw->get(pVelocity_pg, lb->pVelocityLabel, pset_pg);
    old_dw->get(pMass_pg, lb->pMassLabel, pset_pg);

    //patchOnly data
    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);

    constParticleVariable<Point>  pX_p;
    new_dw->get(pX_p, lb->pXXLabel, pset_p);

    //particle index exchange from patch to patch+ghost
    IndexExchange indexExchange(pset_p,pX_p,pset_pg,pX_pg);

    Lattice lattice(pX_pg);

    //Allocate new data
    ParticleVariable<Vector> pContactForce_p_new;
    ParticleVariable<int>    pCrackEffective_p_new;

    new_dw->allocate(pContactForce_p_new, lb->pContactForceLabel, pset_p);
    new_dw->allocate(pCrackEffective_p_new, lb->pCrackEffectiveLabel, pset_p);
  
    for(ParticleSubset::iterator iter = pset_p->begin();
      iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = indexExchange.getPatchAndGhostIndex(pIdx_p);

      const Point& particlePoint = pX_pg[pIdx_pg];
      double size0 = pow(pVolume_pg[pIdx_pg],0.333);
    
      ParticlesNeighbor particles;
      lattice.getParticlesNeighbor(particlePoint, particles);
      int particlesNumber = particles.size();

      //crack effective
      pCrackEffective_p_new[pIdx_p] = 0;
      if(pIsBroken_pg[pIdx_pg]) {
        for(int j=0; j<particlesNumber; j++) {
          int idx_pg = particles[j];
          if( pIdx_pg != idx_pg ) {
            if( pIsBroken_pg[idx_pg]) {
              if(Dot(pCrackNormal_pg[pIdx_pg],pCrackNormal_pg[idx_pg]) < 0) {
                pCrackEffective_p_new[pIdx_p] = 1;
                break;
              }
            }
            else {
              if(Dot(pCrackNormal_pg[pIdx_pg],
                     pX_pg[idx_pg]-pX_pg[pIdx_pg]) > 0) {
                pCrackEffective_p_new[pIdx_p] = 1;
                break;
              }
            }
          }
        }
      }

      //contact force
      double mass = 0;
      Vector momentum(0.,0.,0.);
      if(pCrackEffective_p_new[pIdx_p]) {
        const Vector& n0 = pCrackNormal_pg[pIdx_pg];
        for(int j=0; j<particlesNumber; j++) {
          int idx_pg = particles[j];
          if( pIdx_pg != idx_pg && pIsBroken_pg[idx_pg] ) {
            Vector dis = pX_pg[idx_pg] - particlePoint;
            if( Dot( (pVelocity_pg[idx_pg]-pVelocity_pg[pIdx_pg]), n0 ) < 0 &&
	        Dot( pCrackNormal_pg[idx_pg], n0 ) < -0.707 )
	    {
              double size1 = pow(pVolume_pg[idx_pg],1./3.);
              double vDis = Dot( dis, n0 );
              if( vDis>-size0 && vDis<(size0+size1)/2 ) {
                double hDis = (dis - n0 * vDis).length();
                if(hDis < size0*2.0) {
                  mass += pMass_pg[idx_pg];
                  momentum += (pVelocity_pg[idx_pg] * pMass_pg[idx_pg]);
                }
              }
            }
          }
        }
      }
      
      if( mass > 0 ) {
        pCrackEffective_p_new[pIdx_p] = -1;
        pContactForce_p_new[pIdx_p] = 
	  pCrackNormal_pg[pIdx_pg] * Dot( pCrackNormal_pg[pIdx_pg],
	  ( momentum/mass - pVelocity_pg[pIdx_pg] )*
          ( pMass_pg[pIdx_pg] /2 /delT ) );
      }
      else {
        pContactForce_p_new[pIdx_p] = Vector(0.,0.,0.);
      }
    }

    new_dw->put(pContactForce_p_new, lb->pContactForceLabel);
    new_dw->put(pCrackEffective_p_new, lb->pCrackEffectiveLabel);
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

    constParticleVariable<Point>  pX_pg;
    constParticleVariable<double> pVolume_pg;
    constParticleVariable<Vector> pCrackNormal_pg;
    constParticleVariable<int>    pCrackEffective_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
    old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
    new_dw->get(pCrackEffective_pg, lb->pCrackEffectiveLabel, pset_pg);

    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);

    /*
    cout<<"computeConnectivity:computeBoundaryContact: "
      <<pset_p->numParticles()<<endl;
      */

    constParticleVariable<Point>  pX_p;
    new_dw->get(pX_p, lb->pXXLabel, pset_p);

    IndexExchange indexExchange(pset_p,pX_p,pset_pg,pX_pg);

    ParticleVariable<int>       pConnectivity_p_new;
    new_dw->allocate(pConnectivity_p_new, lb->pConnectivityLabel, pset_p);

    Lattice lattice(pX_pg);
    ParticlesNeighbor particles;
    IntVector cellIdx;

    vector<BoundaryBand> pBoundaryBand_pg(pset_pg->numParticles());
    int pnumber = pBoundaryBand_pg.size();
    for(int pidx=0;pidx<pnumber;++pidx) {
      pBoundaryBand_pg[pidx].setup(pidx,
                             pCrackNormal_pg,
			     pCrackEffective_pg,
			     pVolume_pg,
			     lattice,
			     cellLength*0.75);
    }
    
    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = indexExchange.getPatchAndGhostIndex(pIdx_p);
      
      const Point& part = pX_pg[pIdx_pg];

      patch->findCell(pX_p[pIdx_p],cellIdx);
      particles.clear();
      particles.buildIn(cellIdx,lattice);

      IntVector nodeIdx[8];
      patch->findNodesFromCell(cellIdx,nodeIdx);
    
      int conn[8];
      for(int k=0;k<8;++k) {
        Point node = patch->nodePosition(nodeIdx[k]);

	conn[k] = 1;
	VisibilityConnection(
	        particles, pIdx_pg, node,
		pCrackEffective_pg, pCrackNormal_pg, pVolume_pg,
		pX_pg, conn[k]);

	if(conn[k] == 0) BoundaryBandConnection(
	        pBoundaryBand_pg, pIdx_pg, part, node, conn[k]);

      } //loop over of k
      
      Connectivity connectivity(conn);
      pConnectivity_p_new[pIdx_p] = connectivity.flag();
    }
  
    new_dw->put(pConnectivity_p_new, lb->pConnectivityLabel);
  }
}

void NormalFracture::computeFracture(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  //cout<<"patch size: "<<patches->size()<<endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int matlindex = mpm_matl->getDWIndex();
    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
       patch, Ghost::AroundCells, 1, lb->pXLabel);

    constParticleVariable<Point>   pX_pg;
    constParticleVariable<double>  pVolume_pg;
    constParticleVariable<Vector>  pTipNormal_pg;
    constParticleVariable<Vector>  pExtensionDirection_pg;
    constParticleVariable<Matrix3> pStress_pg;
    constParticleVariable<double>  pToughness_pg;
    constParticleVariable<double>  pStrainEnergy_pg;
    constParticleVariable<double>  pMass_pg;
    constParticleVariable<int>     pIsBroken_pg;
    constParticleVariable<Vector>  pCrackNormal_pg;
    constParticleVariable<Vector>  pDisplacement_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);
    old_dw->get(pTipNormal_pg, lb->pTipNormalLabel, pset_pg);
    old_dw->get(pExtensionDirection_pg, lb->pExtensionDirectionLabel, pset_pg);
    new_dw->get(pStress_pg, lb->pStressLabel_afterStrainRate, pset_pg);
    old_dw->get(pToughness_pg, lb->pToughnessLabel, pset_pg);
    new_dw->get(pStrainEnergy_pg, lb->pStrainEnergyLabel, pset_pg);
    old_dw->get(pMass_pg, lb->pMassLabel, pset_pg);
    old_dw->get(pIsBroken_pg, lb->pIsBrokenLabel, pset_pg);
    old_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel, pset_pg);
    old_dw->get(pDisplacement_pg, lb->pDisplacementLabel, pset_pg);

    //patchOnly data
    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  
    constParticleVariable<Point>   pX_p;
    constParticleVariable<Vector>  pRotationRate_p;
    constParticleVariable<double>  pCrackSurfacePressure_p;
    constParticleVariable<Vector>  pVelocity_p;

    new_dw->get(pX_p, lb->pXXLabel, pset_p);
    new_dw->get(pRotationRate_p, lb->pRotationRateLabel, pset_p);
    old_dw->get(pCrackSurfacePressure_p,lb->pCrackSurfacePressureLabel,pset_p);
    new_dw->get(pVelocity_p, lb->pVelocityLabel_afterUpdate, pset_p);
    
    //particle index exchange from patch to patch+ghost
    IndexExchange indexExchange(pset_p,pX_p,pset_pg,pX_pg);

    Lattice lattice(pX_pg);

    ParticleVariable<int> pIsBroken_p_new;
    ParticleVariable<Vector> pCrackNormal_p_new;
    ParticleVariable<Matrix3> pStress_p_new;
    ParticleVariable<double> pCrackSurfacePressure_p_new;
    ParticleVariable<Vector> pVelocity_p_new;
    ParticleVariable<double> pToughness_p_new;
  
    new_dw->allocate(pIsBroken_p_new, lb->pIsBrokenLabel, pset_p);
    new_dw->allocate(pCrackNormal_p_new, lb->pCrackNormalLabel, pset_p);
    new_dw->allocate(pStress_p_new, lb->pStressLabel, pset_p);
    new_dw->allocate(pCrackSurfacePressure_p_new,
      lb->pCrackSurfacePressureLabel, pset_p);
    new_dw->allocate(pVelocity_p_new, lb->pVelocityLabel, pset_p);
    new_dw->allocate(pToughness_p_new, lb->pToughnessLabel, pset_p);

    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = indexExchange.getPatchAndGhostIndex(pIdx_p);
    
      pIsBroken_p_new[pIdx_p] = pIsBroken_pg[pIdx_pg];
      pStress_p_new[pIdx_p] = pStress_pg[pIdx_pg];
      pToughness_p_new[pIdx_p] = pToughness_pg[pIdx_pg];
      
      //for explosive fracture
      pCrackSurfacePressure_p_new[pIdx_p] = pCrackSurfacePressure_p[pIdx_p];
      if(pCrackSurfacePressure_p[pIdx_p]>0) {
        pCrackSurfacePressure_p_new[pIdx_p] += 
	  mpm_matl->getPressureRate() * delT;
      }

      pVelocity_p_new[pIdx_p] = pVelocity_p[pIdx_p];

      pCrackNormal_p_new[pIdx_p] = pCrackNormal_pg[pIdx_pg];
      if( pIsBroken_p_new[pIdx_p] > 0 ) {
        pCrackNormal_p_new[pIdx_p] += Cross( pRotationRate_p[pIdx_p] * delT, 
	                            pCrackNormal_p_new[pIdx_p] );
        pCrackNormal_p_new[pIdx_p].normalize();
      }
    }
    
    //build surface couples
    vector<SurfaceCouple> couples;
    {
      double relate_cosine = 0.5;
      for(ParticleSubset::iterator iterA = pset_pg->begin();
          iterA != pset_pg->end(); iterA++)
      {
        particleIndex pIdxA_pg = *iterA;
        const Vector& tipNormalA = pTipNormal_pg[pIdxA_pg];
        if(tipNormalA.length2()>0.5) {
          particleIndex match = -1;
          double distance = pow(pVolume_pg[pIdxA_pg],0.333)*2; //max crack gap
	  Vector normal;
          ParticlesNeighbor particles;
          lattice.getParticlesNeighbor(pX_pg[pIdxA_pg], particles);
          int particlesNumber = particles.size();
          for(int j=0; j<particlesNumber; j++)
          {
            particleIndex pIdxB_pg = particles[j];
            if(pIdxA_pg != pIdxB_pg) {
	      if(pIsBroken_pg[pIdxB_pg]) {
                const Vector& crackNormalB = pCrackNormal_pg[pIdxB_pg];
                if( Dot(tipNormalA,crackNormalB) < -relate_cosine ) {
                  Vector dis = pX_pg[pIdxB_pg] - pX_pg[pIdxA_pg];
	          double AB = dis.length();
	          double vAB = Dot(tipNormalA,dis);
                  if(vAB/AB>relate_cosine) {
	            if(AB<distance) {
	              //cout<<"crack couple"<<endl;
	              distance = AB;
		      match = pIdxB_pg;
		      normal = tipNormalA - crackNormalB;
 	              normal.normalize();
	            }
	          }
	        }
	      }
	      else {
	        const Vector& tipNormalB = pTipNormal_pg[pIdxB_pg];
	        if(tipNormalB.length2() > 0.5) {
                  if( Dot(tipNormalA,tipNormalB) < -relate_cosine ) {
                    Vector dis = pX_pg[pIdxB_pg] - pX_pg[pIdxA_pg];
	            double AB = dis.length();
	            double vAB = Dot(tipNormalA,dis);
                    if(vAB/AB>relate_cosine) {
	              if(AB<distance) {
	                //cout<<"tip couple"<<endl;
	                distance = AB;
		        match = pIdxB_pg;
		        normal = tipNormalA - tipNormalB;
 	                normal.normalize();
	              }
	            }
	          }
	        }
	      }
	    }
          }
          if(match >= 0) {
            bool newMatch = true;
            int coupleNum = couples.size();
            for(int i=0;i<coupleNum;++i) {
	      if( couples[i].getIdxA() == match && 
	          couples[i].getIdxB() == pIdxA_pg )
	      {
                newMatch = false;
	        break;
	      }
            }
            if(newMatch) {
	      SurfaceCouple couple(pIdxA_pg,match,normal);
	      couples.push_back(couple);
	    }
	  }
        } //if A
      }
    }
    
    //cout<<"tip couples number: "<<couples.size()<<endl;
    
    //calculate energy release rate
    
    int coupleNum = couples.size();
    for(int k=0;k<coupleNum;++k) {
      const SurfaceCouple& couple = couples[k];
      Vector nx = pExtensionDirection_pg[couple.getIdxA()] +
                  pExtensionDirection_pg[couple.getIdxB()];
      nx.normalize();

      double toughness = ( pToughness_pg[couple.getIdxA()] +
                           pToughness_pg[couple.getIdxB()] )/2;
      double GI,GII,GIII;
      Vector N;

      if( couple.computeCrackClosureIntegralAndCrackNormalFromEnergyReleaseRate(
        nx,lattice,pStress_pg,pDisplacement_pg,pVolume_pg,pIsBroken_pg,toughness,
	GI,GII,GIII,N) )
      {
	int constraint = mpm_matl->getFractureModel()->getConstraint();
	if( constraint > 0) N[constraint] = 0.;

	//pIdxA
	{
          particleIndex pIdx_pg = couple.getIdxA();
	  int pIdx_p = indexExchange.getPatchOnlyIndex(pIdx_pg);
	  if( pIdx_p >= 0 && !pIsBroken_pg[pIdx_pg] ) {
	    pCrackNormal_p_new[pIdx_p] = N;
            pIsBroken_p_new[pIdx_p] = 1;
            pCrackSurfacePressure_p_new[pIdx_p] = 
	        mpm_matl->getExplosivePressure();
            cout<<"crack! "<<"normal="<<pCrackNormal_p_new[pIdx_p]<<endl;
	  }
        }

	//pIdxB
	{
	  particleIndex pIdx_pg = couple.getIdxB();
  	  int pIdx_p = indexExchange.getPatchOnlyIndex(pIdx_pg);
	  if( pIdx_p >= 0 && !pIsBroken_pg[pIdx_pg] ) {
  	    pCrackNormal_p_new[pIdx_p] = -N;
	    pIsBroken_p_new[pIdx_p] = 1;
	    pCrackSurfacePressure_p_new[pIdx_p] = 
	        mpm_matl->getExplosivePressure();
            cout<<"crack! "<<"normal="<<pCrackNormal_p_new[pIdx_p]<<endl;
	  }
	}
      }
    }
    
    new_dw->put(pToughness_p_new, lb->pToughnessLabel_preReloc);
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
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int matlindex = mpm_matl->getDWIndex();

    //patchAndGhost data (_pg)
    ParticleSubset* pset_pg = old_dw->getParticleSubset(matlindex, 
      patch, Ghost::AroundCells, 1, lb->pXLabel);

    constParticleVariable<Point>  pX_pg;
    constParticleVariable<int>    pIsBroken_pg;
    constParticleVariable<Vector> pCrackNormal_pg;
    constParticleVariable<Vector> pExtensionDirection_pg;
    constParticleVariable<double> pVolume_pg;

    old_dw->get(pX_pg, lb->pXLabel, pset_pg);
    new_dw->get(pIsBroken_pg, lb->pIsBrokenLabel_preReloc, pset_pg);
    new_dw->get(pCrackNormal_pg, lb->pCrackNormalLabel_preReloc, pset_pg);
    old_dw->get(pExtensionDirection_pg, lb->pExtensionDirectionLabel, pset_pg);
    old_dw->get(pVolume_pg, lb->pVolumeLabel, pset_pg);

    //patchAndGhost data (_p)
    ParticleSubset* pset_p = old_dw->getParticleSubset(matlindex, patch);
  
    constParticleVariable<Point>   pX_p;
    constParticleVariable<Vector>  pTipNormal_p;
    constParticleVariable<Vector>  pRotationRate_p;

    new_dw->get(pX_p, lb->pXXLabel, pset_p);
    old_dw->get(pTipNormal_p, lb->pTipNormalLabel, pset_p);
    new_dw->get(pRotationRate_p, lb->pRotationRateLabel, pset_p);

    //Lattice for particle neighbor finding
    IndexExchange indexExchange(pset_p,pX_p,pset_pg,pX_pg);
    Lattice lattice(pX_pg);

    //variables to be computed
    ParticleVariable<Vector> pTipNormal_p_new;
    ParticleVariable<Vector> pExtensionDirection_p_new;

    new_dw->allocate(pTipNormal_p_new, lb->pTipNormalLabel, pset_p);
    new_dw->allocate(pExtensionDirection_p_new, 
      lb->pExtensionDirectionLabel, pset_p);
    
    //build newly cracked surface couples
    vector<SurfaceCouple> couples;
    {
      double relate_cosine = 0.7;
      for(ParticleSubset::iterator iterA = pset_pg->begin();
          iterA != pset_pg->end(); iterA++)
      {
        particleIndex pIdxA_pg = *iterA;

        if(pIsBroken_pg[pIdxA_pg] && 
	   pExtensionDirection_pg[pIdxA_pg].length2()>0.5)
        //if(pIsBroken_pg[pIdxA_pg])
	{
          const Vector& crackNormalA = pCrackNormal_pg[pIdxA_pg];
          particleIndex match = -1;
          double distance = pow(pVolume_pg[pIdxA_pg],0.333)*2;
	  Vector normal = crackNormalA;
          ParticlesNeighbor particles;
          lattice.getParticlesNeighbor(pX_pg[pIdxA_pg], particles);
          int particlesNumber = particles.size();
          for(int j=0; j<particlesNumber; j++)
          {
            particleIndex pIdxB_pg = particles[j];
            if(pIdxA_pg != pIdxB_pg) {
	      if(pIsBroken_pg[pIdxB_pg]) {
                const Vector& crackNormalB = pCrackNormal_pg[pIdxB_pg];
                if( Dot(crackNormalA,crackNormalB) < -relate_cosine ) {
		  normal -= crackNormalB;
	          normal.normalize();
                  Vector dis = pX_pg[pIdxB_pg] - pX_pg[pIdxA_pg];
	          double AB = dis.length();
	          double vAB = Dot(normal,dis);
                  if(vAB/AB>relate_cosine) {
	            if(AB<distance) {
	              distance = AB;
		      match = pIdxB_pg;
	            }
	          }
	        }
	      }
/*
	      else {
                Vector dis = pX_pg[pIdxB_pg] - pX_pg[pIdxA_pg];
	        double AB = dis.length();
	        double vAB = Dot(normal,dis);
                if(vAB/AB>relate_cosine) {
	          if(AB<distance) {
	            distance = AB;
	            match = pIdxB_pg;
		  }
	        }
	      }
*/
	    }
          }
          if(match >= 0) {
            bool newMatch = true;
            int coupleNum = couples.size();
            for(int i=0;i<coupleNum;++i) {
	      if( couples[i].getIdxA() == match && 
	          couples[i].getIdxB() == pIdxA_pg )
	      {
                newMatch = false;
	        break;
	      }
            }
            if(newMatch) {
	      SurfaceCouple couple(pIdxA_pg,match,normal);
	      couples.push_back(couple);
	    }
	  }
        } //if A
      }
    }
    int couplesNum = couples.size();
    //cout<<"before extension crack couplesNum "<<couplesNum<<endl;

    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = indexExchange.getPatchAndGhostIndex(pIdx_p);
    
      pExtensionDirection_p_new[pIdx_p] = pExtensionDirection_pg[pIdx_pg];
      pTipNormal_p_new[pIdx_p] = pTipNormal_p[pIdx_p];
    }

    //clean the newly crack tip area
    for(int k=0;k<couplesNum;++k) {
      Point tip(couples[k].crackTip(pX_pg));
      ParticlesNeighbor particles;
      lattice.getParticlesNeighbor(tip, particles);
      double d = pow( (pVolume_pg[couples[k].getIdxA()]+
                       pVolume_pg[couples[k].getIdxB()])/2, 0.333 ) * 3;
      int particlesNumber = particles.size();
      for(int j=0; j<particlesNumber; j++)
      {
        particleIndex pidx_pg = particles[j];
        if( (pX_pg[pidx_pg]-tip).length() < d ) {
	  particleIndex pIdx_p = indexExchange.getPatchOnlyIndex(pidx_pg);
	  if(pIdx_p >= 0) {
	    Vector dis = pX_pg[pidx_pg]-tip;
	    if(Dot(pExtensionDirection_pg[couples[k].getIdxA()],dis)>0 ||
	       Dot(pExtensionDirection_pg[couples[k].getIdxB()],dis)>0)
	    {
              pExtensionDirection_p_new[pIdx_p] = Vector(0.,0.,0.);
              pTipNormal_p_new[pIdx_p] = Vector(0.,0.,0.);
	    }
	  }
	}
      }
    }

    for(ParticleSubset::iterator iter = pset_p->begin();
          iter != pset_p->end(); iter++)
    {
      particleIndex pIdx_p = *iter;
      particleIndex pIdx_pg = indexExchange.getPatchAndGhostIndex(pIdx_p);
      
      if( pExtensionDirection_p_new[pIdx_p].length2()>0.5 && 
         !pIsBroken_pg[pIdx_pg] )
      {
        pExtensionDirection_p_new[pIdx_p] += 
	    Cross( pRotationRate_p[pIdx_p] * delT, 
	    pExtensionDirection_p_new[pIdx_p] );
        pExtensionDirection_p_new[pIdx_p].normalize();
        pTipNormal_p_new[pIdx_p] += 
	    Cross( pRotationRate_p[pIdx_p] * delT, 
	    pTipNormal_p_new[pIdx_p] );
        pTipNormal_p_new[pIdx_p].normalize();
      }

      if(!pIsBroken_pg[pIdx_pg])
      {
        double volume = pVolume_pg[pIdx_pg];
        int extension = -1;
        double distanceToCrack = pow(volume,0.333)*0.866*3;
        for(int k=0;k<couplesNum;++k) {
	  if( couples[k].extensible(pIdx_pg,
	    pX_pg,pExtensionDirection_pg,pCrackNormal_pg,
	    volume,distanceToCrack) ) extension = k;
	}
        
        if( extension >= 0 )
        {
	  //cout<<"A extension" <<endl;
	  const SurfaceCouple& couple = couples[extension];
          Point pTip = couple.crackTip(pX_pg);
          Vector& Ny = pTipNormal_p_new[pIdx_p];
	  Vector& Nx = pExtensionDirection_p_new[pIdx_p];
	  Ny = couple.getNormal();
          Vector dis = pX_p[pIdx_p] - pTip;
          if(Dot(dis,Ny) > 0) Ny = -Ny;
          Nx = dis - Ny * Dot(dis,Ny);
          Nx.normalize();

          int constraint = mpm_matl->getFractureModel()->getConstraint();
	  if( constraint > 0) {
	    Nx[constraint] = 0.;
	    Ny[constraint] = 0.;
	  }
        }
      }
    }
    
    //cout<<"NumExtension "<<NumExtension<<endl;

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
   particleIndex pIdx_pg,
   const Point& part,
   const Point& node,
   int& conn) const
{
  int part_pBB = pBoundaryBand_pg[pIdx_pg].inside(pIdx_pg);
  int node_pBB = pBoundaryBand_pg[pIdx_pg].inside(node);
	
  if(pBoundaryBand_pg[pIdx_pg].numCracks() != 0) {
         if( part_pBB &&  node_pBB) conn = 1;
    else if(!part_pBB && !node_pBB) conn = 1;
    else if(!part_pBB &&  node_pBB) conn = 0;
    else conn = 0;
  }
}

void NormalFracture::VisibilityConnection(
   const ParticlesNeighbor& particles,
   particleIndex pIdx_pg,
   const Point& node,
   const ParticleVariable<int>& pCrackEffective_pg,
   const ParticleVariable<Vector>& pCrackNormal_pg,
   const ParticleVariable<double>& pVolume_pg,
   const ParticleVariable<Point>& pX_pg,
   int& conn) const
{
  const Point& part = pX_pg[pIdx_pg];

  if(pCrackEffective_pg[pIdx_pg]) {
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
    if(pCrackEffective_pg[pidx_pg]) {
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
      if( (dis-pTouchNormal_pg[pIdx_pg]*vdis).length() < 
          connectionRadius(pVolume_pg[pIdx_pg]) )
      {
        conn = 2;
        pContactNormal = pTouchNormal_pg[pIdx_pg];
        return;
      }
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
  return pow(volume,0.333333)*0.87;
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
