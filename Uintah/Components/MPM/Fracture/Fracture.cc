#include "Fracture.h"

#include "ParticlesNeighbor.h"
#include "Visibility.h"

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

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

void
Fracture::
initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouseP& new_dw)
{
   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   
   ParticleVariable<int> pIsBroken;
   new_dw->allocate(pIsBroken, lb->pIsBrokenLabel, pset);
   ParticleVariable<Vector> pCrackSurfaceNormal;
   new_dw->allocate(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, pset);   
   ParticleVariable<double> pMicrocrackSize;
   new_dw->allocate(pMicrocrackSize, lb->pMicrocrackSizeLabel, pset);
   ParticleVariable<double> pMicrocrackPosition;
   new_dw->allocate(pMicrocrackPosition, lb->pMicrocrackPositionLabel, pset);
   ParticleVariable<double> pCrackingSpeed;
   new_dw->allocate(pCrackingSpeed, lb->pCrackingSpeedLabel, pset);
   
   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
        particleIndex idx = *iter;
	pIsBroken[idx] = 0;
	pCrackSurfaceNormal[idx] = Vector(0.,0.,0.);
	pMicrocrackSize[idx] = 0;
	pMicrocrackPosition[idx] = 0;
	pCrackingSpeed[idx] = 0;
   }

   new_dw->put(pIsBroken, lb->pIsBrokenLabel);
   new_dw->put(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel);
   new_dw->put(pMicrocrackSize, lb->pMicrocrackSizeLabel);
   new_dw->put(pMicrocrackPosition, lb->pMicrocrackPositionLabel);
   new_dw->put(pCrackingSpeed, lb->pCrackingSpeedLabel);
}

void Fracture::computerNodesVisibilityAndCrackSurfaceContactForce(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  int matlindex = mpm_matl->getDWIndex();
  int vfindex = mpm_matl->getVFIndex();

  // Create arrays for the particle data
  ParticleVariable<Point>  pX;
  ParticleVariable<Vector> pCrackSurfaceNormal;
  ParticleVariable<double> pMicrocrackSize;
  ParticleVariable<double> pMicrocrackPosition;
  ParticleVariable<int>    pIsBroken;

  //for crack surface contact computation
  ParticleVariable<Vector> pVelocity;
  ParticleVariable<double> pVolume;
  ParticleVariable<double> pMass;

  ParticleSubset* outsidePset = old_dw->getParticleSubset(matlindex, patch,
	Ghost::AroundNodes, 1, lb->pXLabel);

  old_dw->get(pX,             lb->pXLabel, outsidePset);
  old_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, outsidePset);
  old_dw->get(pMicrocrackSize, lb->pMicrocrackSizeLabel, outsidePset);
  old_dw->get(pMicrocrackPosition, lb->pMicrocrackPositionLabel, outsidePset);
  old_dw->get(pIsBroken, lb->pIsBrokenLabel, outsidePset);

  old_dw->get(pVelocity, lb->pVelocityLabel, outsidePset);
  old_dw->get(pVolume, lb->pVolumeLabel, outsidePset);
  old_dw->get(pMass, lb->pMassLabel, outsidePset);
  
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  ParticleSubset* insidePset = old_dw->getParticleSubset(matlindex, patch);

  ParticleVariable<int>    pVisibility;
  new_dw->allocate(pVisibility, lb->pVisibilityLabel, insidePset);
  ParticleVariable<Vector> pCrackSurfaceContactForce;
  new_dw->allocate(pCrackSurfaceContactForce, lb->pCrackSurfaceContactForceLabel,
     insidePset);

  Lattice lattice(pX);
  ParticlesNeighbor particles( pX,
			       pIsBroken,
			       pCrackSurfaceNormal,
			       pMicrocrackSize,
			       pMicrocrackPosition );
  IntVector cellIdx;
  IntVector nodeIdx[8];

  for(ParticleSubset::iterator iter = insidePset->begin();
          iter != insidePset->end(); iter++)
  {
    pCrackSurfaceContactForce[*iter] = Vector(0.,0.,0.);
  }
  
  for(ParticleSubset::iterator iter = insidePset->begin();
          iter != insidePset->end(); iter++)
  {
    particleIndex pIdx = *iter;
    patch->findCell(pX[pIdx],cellIdx);
    particles.clear();
    particles.buildIn(cellIdx,lattice);
    
    //visibility
    patch->findNodesFromCell(cellIdx,nodeIdx);
    Visibility vis;
    for(int i=0;i<8;++i) {
      if( particles.visible( pX[pIdx],patch->nodePosition(nodeIdx[i]) ) )
         vis.setVisible(i);
      else vis.setUnvisible(i);
    }
    pVisibility[pIdx] = vis.flag();
    
    Point& X1 = pX[pIdx];
    double size1 = pow(pVolume[pIdx],0.3333);
    double mass1 = pMass[pIdx];
    Vector& V1 = pVelocity[pIdx];

    //crack surface contact force
    for(int pNeighbor=0; pNeighbor<particles.size();++pNeighbor)
    {
      if( particles[pNeighbor] > pIdx ) {
        particleIndex pContact = particles[pNeighbor];
	if(pIsBroken[pContact] || pIsBroken[pIdx]) {
	
 	  Point& X2 = pX[pContact];
	  double size2 = pow(pVolume[pContact],0.3333);
	  
	  if( (X1-X2).length() < (size1+size2)/2 * 0.9 ) {
            double mass2 = pMass[pContact];
	    Vector& V2 = pVelocity[pContact];
	    Vector F = (V2-V1)*mass1*mass2/(mass1+mass2)/delT;
	    pCrackSurfaceContactForce[pIdx] += F;
	    pCrackSurfaceContactForce[pContact] -= F;
	  }	  
	}
      }
    }
  }
  
  new_dw->put(pVisibility, lb->pVisibilityLabel);
  new_dw->put(pCrackSurfaceContactForce, lb->pCrackSurfaceContactForceLabel);
}

void
Fracture::
crackGrow(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
   ParticleSubset* pset = old_dw->getParticleSubset(mpm_matl->getDWIndex(), 
      patch);
   
   ParticleVariable<Matrix3> pStress;
   ParticleVariable<int> pIsBroken;
   ParticleVariable<Vector> pCrackSurfaceNormal;
   ParticleVariable<double> pMicrocrackSize;
   ParticleVariable<double> pMicrocrackPosition;
   ParticleVariable<double> pCrackingSpeed;
   ParticleVariable<double> pDilationalWaveSpeed;
   ParticleVariable<double> pVolume;
   ParticleVariable<Vector> pRotationRate;

   new_dw->get(pStress, lb->pStressLabel_preReloc, pset);
   old_dw->get(pIsBroken, lb->pIsBrokenLabel, pset);
   old_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, pset);
   old_dw->get(pMicrocrackSize, lb->pMicrocrackSizeLabel, pset);
   old_dw->get(pMicrocrackPosition, lb->pMicrocrackPositionLabel, pset);
   old_dw->get(pCrackingSpeed, lb->pCrackingSpeedLabel, pset);  
   old_dw->get(pVolume, lb->pVolumeLabel, pset);
   new_dw->get(pDilationalWaveSpeed, lb->pDilationalWaveSpeedLabel, pset);
   new_dw->get(pRotationRate, lb->pRotationRateLabel, pset);

   delt_vartype delT;
   old_dw->get(delT, lb->delTLabel);

   double newSpeed;
   
   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++)
   {
      particleIndex idx = *iter;
      
      if(!pIsBroken[idx]) {
        //crack initiation

	//get the max stress and the direction
	double sig[3];
        int eigenValueNum = pStress[idx].getEigenValues(sig[0], sig[1], sig[2]);
	double maxStress = sig[eigenValueNum-1];
        vector<Vector> eigenVectors = pStress[idx].getEigenVectors(maxStress);	
	
	for(int i=0;i<eigenVectors.size();++i) {
          eigenVectors[i].normalize();
	}
	
	Vector maxDirection;
	if(eigenVectors.size() == 1) {
          maxDirection = eigenVectors[0];
	}

	if(eigenVectors.size() == 2) {
	  //cout<<"eigenVectors.size = 2"<<endl;
	  double theta = drand48() * M_PI * 2;
	  maxDirection = (eigenVectors[0] * cos(theta) + eigenVectors[1] * sin(theta));
	}
	
	if(eigenVectors.size() == 3) {
	  //cout<<"eigenVectors.size = 3"<<endl;
	  double theta = drand48() * M_PI * 2;
	  double beta = drand48() * M_PI;
	  double cos_beta = cos(beta);
	  double sin_beta = sin(beta);
	  Vector xy = eigenVectors[2] * sin_beta;
	  maxDirection = xy * cos(theta) +
	                 xy * sin(theta) +
			 eigenVectors[2] * cos_beta;
	}

	//compare with the tensile strength
	if(maxStress > d_tensileStrength) {
	  pIsBroken[idx] = 1;
	  pCrackSurfaceNormal[idx] = maxDirection;
	  pMicrocrackSize[idx] = 0;
	  pMicrocrackPosition[idx] = pow(pVolume[idx],0.33333) * (drand48() - 0.5);
	  pCrackingSpeed[idx] = 0;
	  cout<<"Microcrack initiated in direction "<<maxDirection<<"."<<endl;
	}
      }
      else {
        //crack surface rotation
	pCrackSurfaceNormal[idx] += Cross( pRotationRate[idx] * delT, 
	                                   pCrackSurfaceNormal[idx] );
	pCrackSurfaceNormal[idx].normalize();
	
        //crack propagation
        double sizeLimit = pow(pVolume[idx],0.333) /2;
	
	double tensilStress = Dot(pStress[idx] * pCrackSurfaceNormal[idx],
	   pCrackSurfaceNormal[idx]);
	
	if(tensilStress > d_tensileStrength)
	  newSpeed = pDilationalWaveSpeed[idx] * 
	  ( 1 - exp(tensilStress/d_tensileStrength - 1) );
	else newSpeed = 0;
	
	pMicrocrackSize[idx] += ( pCrackingSpeed[idx] + newSpeed )/2 * delT;
	if(pMicrocrackSize[idx] > sizeLimit) pMicrocrackSize[idx] = sizeLimit;
	
	pCrackingSpeed[idx] = newSpeed;
      }
   }

   new_dw->put(pIsBroken, lb->pIsBrokenLabel_preReloc);
   new_dw->put(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel_preReloc);
   new_dw->put(pMicrocrackSize, lb->pMicrocrackSizeLabel_preReloc);
   new_dw->put(pMicrocrackPosition, lb->pMicrocrackPositionLabel_preReloc);
   new_dw->put(pCrackingSpeed, lb->pCrackingSpeedLabel_preReloc);
}

Fracture::
Fracture(ProblemSpecP& ps)
{
  ps->require("tensile_strength",d_tensileStrength);

  lb = scinew MPMLabel();
}

Fracture::~Fracture()
{
}
  
} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.45  2000/09/11 20:23:26  tan
// Fixed a mistake in crack surface contact force algorithm.
//
// Revision 1.44  2000/09/11 19:45:43  tan
// Implemented crack surface contact force calculation algorithm.
//
// Revision 1.43  2000/09/11 18:55:51  tan
// Crack surface contact force is now considered in the simulation.
//
// Revision 1.42  2000/09/11 01:08:44  tan
// Modified time step calculation (in constitutive model computeStressTensor(...))
// when fracture cracking speed involved.
//
// Revision 1.41  2000/09/11 00:15:00  tan
// Added calculations on random distributed microcracks in broken particles.
//
// Revision 1.40  2000/09/10 23:09:59  tan
// Added calculations on the rotation of crack surafce during fracture.
//
// Revision 1.39  2000/09/10 22:51:13  tan
// Added particle rotationRate computation in computeStressTensor functions
// in each constitutive model classes.  The particle rotationRate will be used
// for fracture.
//
// Revision 1.38  2000/09/09 19:34:16  tan
// Added MPMLabel::pVisibilityLabel and SerialMPM::computerNodesVisibility().
//
// Revision 1.37  2000/09/08 20:28:02  tan
// Added visibility calculation to fracture broken cell shape function
// interpolation.
//
// Revision 1.36  2000/09/08 02:21:55  tan
// Crack initiation works now!
//
// Revision 1.35  2000/09/08 01:47:02  tan
// Added pDilationalWaveSpeedLabel for fracture and is saved as a
// side-effect of computeStressTensor in each constitutive model class.
//
// Revision 1.34  2000/09/07 22:32:02  tan
// Added code to compute crack initiation in crackGrow function.
//
// Revision 1.33  2000/09/07 21:11:10  tan
// Added particle variable pMicrocrackSize for fracture.
//
// Revision 1.32  2000/09/07 00:39:25  tan
// Fixed a bug in ForceBC.
//
// Revision 1.30  2000/09/05 19:38:19  tan
// Fracture starts to run in Uintah/MPM!
//
// Revision 1.29  2000/09/05 05:13:30  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.28  2000/08/09 03:18:02  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.27  2000/07/06 16:58:54  tan
// Least square interpolation added for particle velocities and stresses
// updating.
//
// Revision 1.26  2000/07/05 23:43:37  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.25  2000/07/05 21:37:52  tan
// Filled in the function of updateParticleInformationInContactCells.
//
// Revision 1.24  2000/06/23 16:49:32  tan
// Added LeastSquare Approximation and Lattice for neighboring algorithm.
//
// Revision 1.23  2000/06/23 01:38:07  tan
// Moved material property toughness to Fracture class.
//
// Revision 1.22  2000/06/17 07:06:40  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.21  2000/06/15 21:57:09  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.20  2000/06/05 02:07:59  tan
// Finished labelSelfContactNodesAndCells(...).
//
// Revision 1.19  2000/06/04 23:55:39  tan
// Added labelSelfContactCells(...) to label the self-contact cells
// according to the nodes self-contact information.
//
// Revision 1.18  2000/06/03 05:25:47  sparker
// Added a new for pSurfLabel (was uninitialized)
// Uncommented pleaseSaveIntegrated
// Minor cleanups of reduction variable use
// Removed a few warnings
//
// Revision 1.17  2000/06/02 21:54:22  tan
// Finished function labelSelfContactNodes(...) to label the gSalfContact
// according to the cSurfaceNormal information.
//
// Revision 1.16  2000/06/02 21:12:24  tan
// Added function isSelfContactNode(...) to determine if a node is a
// self-contact node.
//
// Revision 1.15  2000/06/02 19:13:39  tan
// Finished function labelCellSurfaceNormal() to label the cell surface normal
// according to the boundary particles surface normal information.
//
// Revision 1.14  2000/06/02 00:13:13  tan
// Added ParticleStatus to determine if a particle is a BOUNDARY_PARTICLE
// or a INTERIOR_PARTICLE.
//
// Revision 1.13  2000/06/01 23:56:00  tan
// Added CellStatus to determine if a cell HAS_ONE_BOUNDARY_SURFACE,
// HAS_SEVERAL_BOUNDARY_SURFACE or is INTERIOR cell.
//
// Revision 1.12  2000/05/30 20:19:12  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.11  2000/05/30 04:37:00  tan
// Using MPMLabel instead of VarLabel.
//
// Revision 1.10  2000/05/25 00:29:00  tan
// Put all velocity-field independent variables on material index of 0.
//
// Revision 1.9  2000/05/20 08:09:09  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.8  2000/05/15 18:59:10  tan
// Initialized NCVariables and CCVaribles for Fracture.
//
// Revision 1.7  2000/05/12 18:13:07  sparker
// Added an empty function for Fracture::updateSurfaceNormalOfBoundaryParticle
//
// Revision 1.6  2000/05/12 01:46:07  tan
// Added initializeFracture linked to SerialMPM's actuallyInitailize.
//
// Revision 1.5  2000/05/11 20:10:18  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.4  2000/05/10 18:32:11  tan
// Added member funtion to label self-contact cells.
//
// Revision 1.3  2000/05/10 05:06:40  tan
// Basic structure of fracture class.
//
