#include "Fracture.h"

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
Fracture::
initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouseP& new_dw)
{
}

void Fracture::computeNodeVisibility(
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
  ParticleSubset* outsidePset = old_dw->getParticleSubset(matlindex, patch,
	Ghost::AroundNodes, 1, lb->pXLabel);

  old_dw->get(pX, lb->pXLabel, outsidePset);
  old_dw->get(pVolume, lb->pVolumeLabel, outsidePset);
  old_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, outsidePset);
  old_dw->get(pIsBroken, lb->pIsBrokenLabel, outsidePset);

  ParticleSubset* insidePset = old_dw->getParticleSubset(matlindex, patch);

  ParticleVariable<int>    pVisibility;
  new_dw->allocate(pVisibility, lb->pVisibilityLabel, insidePset);

  Lattice lattice(pX);
  ParticlesNeighbor particles;
  IntVector cellIdx;
  IntVector nodeIdx[8];

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
    
    /*
    //output for broken particle information
    if(pVisibility[pIdx] != 255) {
      cout<<"this particle"<<pX[pIdx]<<" "
          <<"is broken "<<pIsBroken[pIdx]<<endl;
      for(int i=0;i<8;++i) {
        cout<<"node "<<i<<": "
	    <<"position "<<patch->nodePosition(nodeIdx[i])<<" "
	    <<"visibility "<<vis.visible(i)<<endl;
      }
      for(vector<particleIndex>::const_iterator 
        ip = lattice[cellIdx].particles.begin();
        ip != lattice[cellIdx].particles.end();
	++ip )
      {
        cout<<"particle position "<<pX[*ip]<<" "
	    <<"crack surface normal "<<pCrackSurfaceNormal[*ip]<<endl;
      }
    }
    */
    
  }
  
  new_dw->put(pVisibility, lb->pVisibilityLabel);
}

void
Fracture::
crackGrow(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
   ParticleVariable<Matrix3> pStress;
   ParticleVariable<int> pIsBroken;
   ParticleVariable<Vector> pCrackSurfaceNormal;
   ParticleVariable<Vector> pRotationRate;
   ParticleVariable<double> pTensileStrength;
   ParticleVariable<int> pIsNewlyBroken;

   int matlindex = mpm_matl->getDWIndex();
   ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

   old_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, pset);
   old_dw->get(pIsBroken, lb->pIsBrokenLabel, pset);
   old_dw->get(pTensileStrength, lb->pTensileStrengthLabel, pset);
   new_dw->get(pRotationRate, lb->pRotationRateLabel, pset);
   new_dw->get(pStress, lb->pStressAfterStrainRateLabel, pset);
   
   new_dw->allocate(pIsNewlyBroken, lb->pIsNewlyBrokenLabel, pset);

   delt_vartype delT;
   old_dw->get(delT, lb->delTLabel);
   
   bool fractured = false;;

   for(ParticleSubset::iterator iter = pset->begin(); 
       iter != pset->end(); iter++)
   {
      particleIndex idx = *iter;
      pIsNewlyBroken[idx] = 0;
      
      //update for broken particles,crack surface rotation
      if(pIsBroken[idx] == 1) {
	//cout<<"CrackSurfaceNormal before: "<<pCrackSurfaceNormal[idx]<<endl;
	pCrackSurfaceNormal[idx] += Cross( pRotationRate[idx] * delT, 
	                                   pCrackSurfaceNormal[idx] );
	pCrackSurfaceNormal[idx].normalize();
	//cout<<"CrackSurfaceNormal afetr: "<<pCrackSurfaceNormal[idx]<<endl;
      }

      //label out the broken particles
      if(pIsBroken[idx] == 0) {

        //get the max stress
        /*
        double sig[3];
        int eigenValueNum = pStress[idx].getEigenValues(sig[0], sig[1], sig[2]);
        double maxStress = sig[eigenValueNum-1];
        */
	
        Vector maxDirection(1,0,0);
        double maxStress = Dot( pStress[idx]*maxDirection, maxDirection );
      
        if(maxStress > pTensileStrength[idx]) {

          //get the max stress direction
/*	
          vector<Vector> eigenVectors = pStress[idx].getEigenVectors(maxStress);	

  	  if(eigenVectors.size() < 1) {
	    cout<<pStress[idx]<<endl;
	    exit(1);
  	  }
	  
	//get the max stress and the direction
	double sig[3];
        pStress[idx].getEigenValues(sig[0], sig[1], sig[2]);
	double maxStress = sig[0];

	//compare with the tensile strength
	if(maxStress > pTensileStrength[idx]) {
          vector<Vector> eigenVectors = pStress[idx].getEigenVectors(maxStress,
							      fabs(maxStress));

	  for(int i=0;i<eigenVectors.size();++i) {
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
	    Vector xy = eigenVectors[2] * sin_beta;
	    maxDirection = eigenVectors[0] * (sin_beta * cos(theta)) +
	                   eigenVectors[1] * (sin_beta * sin(theta)) +
		  	   eigenVectors[2] * cos_beta;
	  }
*/
          //if(drand48()>0.5) maxDirection = -maxDirection;
	  pCrackSurfaceNormal[idx] = maxDirection;
	  pIsBroken[idx] = 1;
	  pIsNewlyBroken[idx] = 1;
	  fractured = true;

	  cout<<"Crack nucleated in direction: "<<pCrackSurfaceNormal[idx]<<"."<<endl;
	}
      }
   }
      
   new_dw->put(pIsBroken, lb->pIsBrokenLabel_preReloc);
   new_dw->put(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel_preReloc);
   new_dw->put(pTensileStrength, lb->pTensileStrengthLabel_preReloc);
   new_dw->put(pIsNewlyBroken, lb->pIsNewlyBrokenLabel);

   delt_vartype delTAfterConstitutiveModel;
   new_dw->get(delTAfterConstitutiveModel, lb->delTAfterConstitutiveModelLabel);
   double delTAfterFracture = delTAfterConstitutiveModel;
   if(fractured) delTAfterFracture /= 3000;
   else {
     double delT_tolerant = delT * 1.5;
     if(delTAfterFracture > delT_tolerant) delTAfterFracture = delT_tolerant;
   }
   new_dw->put(delt_vartype(delTAfterFracture), lb->delTAfterFractureLabel);
}

void
Fracture::
stressRelease(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw)
{
  //patch + ghost variables
  ParticleVariable<Point> pX;
  ParticleVariable<int> pIsNewlyBroken;
  ParticleVariable<Vector> pCrackSurfaceNormal;

  int matlindex = mpm_matl->getDWIndex();
  ParticleSubset* outsidePset = old_dw->getParticleSubset(matlindex, patch,
	Ghost::AroundNodes, 1, lb->pXLabel);

  old_dw->get(pX, lb->pXLabel, outsidePset);
  new_dw->get(pIsNewlyBroken, lb->pIsNewlyBrokenLabel, outsidePset);
  new_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel_preReloc, outsidePset);

  Lattice lattice(pX);

  //patch variables
  ParticleVariable<Matrix3> pStress;
  ParticleVariable<int> pStressReleased;

  ParticleSubset* insidePset = old_dw->getParticleSubset(matlindex, patch);
  new_dw->get(pStress, lb->pStressAfterStrainRateLabel, insidePset);
  new_dw->allocate(pStressReleased, lb->pStressReleasedLabel, insidePset);
  
  for(ParticleSubset::iterator iter = insidePset->begin(); 
     iter != insidePset->end(); iter++)
  {
    pStressReleased[*iter] = 0;
  }
  
  double range = ( patch->dCell().x() + 
	           patch->dCell().y() + 
		   patch->dCell().z() )/3;

  for(ParticleSubset::iterator iter = outsidePset->begin(); 
     iter != outsidePset->end(); iter++)
  {
    particleIndex idx = *iter;
    
    if(pIsNewlyBroken[idx]) {
      IntVector cellIdx;
      patch->findCell(pX[idx],cellIdx);
      ParticlesNeighbor particlesNeighbor;
      particlesNeighbor.buildIn(cellIdx,lattice);
      
      const Vector& N = pCrackSurfaceNormal[idx];
      
      for(std::vector<particleIndex>::const_iterator ip = particlesNeighbor.begin();
	       ip != particlesNeighbor.end(); ++ip)
      {
        particleIndex pNeighbor = *ip;	
	
	if( !patch->findCell(pX[pNeighbor],cellIdx) ) continue;
	if( pStressReleased[pNeighbor] == 1 ) continue;
	
	if( (pX[idx] - pX[pNeighbor]).length() < range ) {
	    Matrix3 stress;
	    double s = Dot(pStress[pNeighbor]*N,N);
	    for(int i=1;i<=3;++i)
	    for(int j=1;j<=3;++j) {
	      stress(i,j) = 0;
	      for(int k=1;k<=3;++k)
	        if(i==j)
	          stress(i,j) += N(i-1) * pStress[pNeighbor](i,k) * N(k-1);
		else
	          stress(i,j) += N(j-1) * pStress[pNeighbor](i,k) * N(k-1)+
		                 N(i-1) * pStress[pNeighbor](j,k) * N(k-1);
	    }
	    //cout<<"stress before: "<<endl<<pStress[pNeighbor]<<endl;
	    pStress[pNeighbor] -= stress;
	    pStressReleased[pNeighbor] = 1;
	    //cout<<"stress after: "<<endl<<pStress[pNeighbor]<<endl;
	}
      }
    }
  }
  
  new_dw->put(pStress, lb->pStressAfterFractureReleaseLabel);
}

Fracture::
Fracture(ProblemSpecP& ps)
{
  lb = scinew MPMLabel();
}

Fracture::~Fracture()
{
}
  
} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.50  2000/09/22 07:18:57  tan
// MPM code works with fracture in three point bending.
//
// Revision 1.49  2000/09/20 18:28:36  witzel
// Oops.  Needed to take fabs for Matrix3::getEigenVectors relative_scale
// parameter.
//
// Revision 1.48  2000/09/20 18:10:09  witzel
// Eigen-value indices were changed such that sig[0] is the largest
// (because the order was changed in Matrix3::getEigenValues).  Also,
// gave relative_scale parameter for Matrix3::getEigenVectors().
//
// Revision 1.47  2000/09/16 04:18:04  tan
// Modifications to make fracture works well.
//
// Revision 1.46  2000/09/12 16:52:11  tan
// Reorganized crack surface contact force algorithm.
//
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
