//
// $Id$
//

#include "FrictionContact.h"
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace Uintah::MPM;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Dot;
using SCICore::Geometry::IntVector;
using std::vector;
using std::string;

using namespace std;

FrictionContact::FrictionContact(ProblemSpecP& ps,
				 SimulationStateP& d_sS)
{
  // Constructor
  IntVector v_f;

  ps->require("vel_fields",v_f);
  ps->require("mu",d_mu);

  d_sharedState = d_sS;
}

FrictionContact::~FrictionContact()
{
  // Destructor
}

void FrictionContact::initializeContact(const Patch* patch,
                                        int vfindex,
                                        DataWarehouseP& new_dw)
{
  NCVariable<double> normtraction;
  NCVariable<Vector> surfnorm;

  new_dw->allocate(normtraction,lb->gNormTractionLabel,vfindex , patch);
  new_dw->allocate(surfnorm,lb->gSurfNormLabel,vfindex , patch);

  normtraction.initialize(0.0);
  surfnorm.initialize(Vector(0.0,0.0,0.0));

  new_dw->put(normtraction,lb->gNormTractionLabel,vfindex , patch);
  new_dw->put(surfnorm,lb->gSurfNormLabel,vfindex , patch);

}

void FrictionContact::exMomInterpolated(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt;
  double centerOfMassMass;
  Vector dx = patch->dCell();

  int numMatls = d_sharedState->getNumMatls();
  int NVFs = d_sharedState->getNumVelFields();

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  vector<NCVariable<double> > gmass(NVFs);
  vector<NCVariable<Vector> > gvelocity(NVFs);
  vector<NCVariable<double> > normtraction(NVFs);
  vector<NCVariable<Vector> > surfnorm(NVFs);
  
  //  const MPMLabel *lb = MPMLabel::getLabels();

  // Retrieve necessary data from DataWarehouse
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], lb->gMassLabel,vfindex , patch,
		  Ghost::None, 0);
      new_dw->get(gvelocity[vfindex], lb->gVelocityLabel, vfindex, patch,
		  Ghost::None, 0);
      old_dw->get(normtraction[vfindex],lb->gNormTractionLabel,vfindex , patch,
		  Ghost::None, 0);
      old_dw->get(surfnorm[vfindex],lb->gSurfNormLabel,vfindex , patch,
		  Ghost::None, 0);
    }
  }
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int n = 0; n < NVFs; n++){
      centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
      centerOfMassMass+=gmass[n][*iter]; 
    }

    // Apply Coulomb friction contact
    // For grid points with mass calculate velocity
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

      // Loop over velocity fields.  Only proceed if velocity field mass
      // is nonzero (not numerical noise) and the difference from
      // the centerOfMassVelocity is nonzero (More than one velocity
      // field is contributing to grid vertex).
      for(int n = 0; n < NVFs; n++){
        Vector deltaVelocity=gvelocity[n][*iter]-centerOfMassVelocity;
        if(!compare(gmass[n][*iter]/centerOfMassMass,0.0)
	   //           && !compare(deltaVelocity.length(),0.0)){
           && !compare(gmass[n][*iter]-centerOfMassMass,0.0)){

          // Apply frictional contact if the surface is in compression
          // or the surface is stress free and surface is approaching.
          // Otherwise apply free surface conditions (do nothing).
          double normalDeltaVelocity=Dot(deltaVelocity,surfnorm[n][*iter]);
	  Dvdt=zero;
          if((normtraction[n][*iter] < 0.0) ||
             (compare(fabs(normtraction[n][*iter]),0.0) &&
              normalDeltaVelocity>0.0)){

              // Specialize algorithm in case where approach velocity
              // is in direction of surface normal.
              if(compare( (deltaVelocity
                        -surfnorm[n][*iter]*normalDeltaVelocity).length(),0.0)){
                Dvdt=-surfnorm[n][*iter]*normalDeltaVelocity;
              }
	      else if(!compare(fabs(normalDeltaVelocity),0.0)){
                Vector surfaceTangent=
		(deltaVelocity-surfnorm[n][*iter]*normalDeltaVelocity)/
                (deltaVelocity-surfnorm[n][*iter]*normalDeltaVelocity).length();
                double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                double frictionCoefficient=
                  Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVelocity));
                Dvdt=
                  -surfnorm[n][*iter]*normalDeltaVelocity
		  -surfaceTangent*frictionCoefficient*fabs(normalDeltaVelocity);
	      }
	      Vector epsilon=(Dvdt/dx)*delT;
	      double epsilon_max=
		Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
	      if(!compare(epsilon_max,0.0)){
		epsilon_max=epsilon_max*Max(1.0,
			  gmass[n][*iter]/(centerOfMassMass-gmass[n][*iter]));
		double ff=Min(epsilon_max,.5)/epsilon_max;
		Dvdt=Dvdt*ff;
	      }
	      gvelocity[n][*iter]+=Dvdt;

          }
	}
      }
    }
  }

  // Store new velocities in DataWarehouse
  for(int n=0; n< NVFs; n++){
    new_dw->put(gvelocity[n], lb->gMomExedVelocityLabel, n, patch);
  }
}

void FrictionContact::exMomIntegrated(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt;
  double centerOfMassMass;
  IntVector onex(1,0,0), oney(0,1,0), onez(0,0,1);
  typedef IntVector IV;
  Vector dx = patch->dCell();

  int numMatls = d_sharedState->getNumMatls();
  int NVFs = d_sharedState->getNumVelFields();

  //  const MPMLabel* lb = MPMLabel::getLabels();

  // This model requires getting the normal component of the
  // surface traction.  The first step is to calculate the
  // surface normals of each object.  Next, interpolate the
  // stress to the grid.  The quantity we want is n^T*stress*n
  // at each node.

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  vector<NCVariable<double> > gmass(NVFs);
  vector<NCVariable<Vector> > gvelocity_star(NVFs);
  vector<NCVariable<Vector> > gacceleration(NVFs);
  vector<NCVariable<double> > normtraction(NVFs);
  vector<NCVariable<Vector> > gsurfnorm(NVFs);


  Vector surnor;

  // First, calculate the gradient of the mass everywhere
  // normalize it, and stick it in surfNorm
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfi = matl->getVFIndex();
      new_dw->get(gmass[vfi], lb->gMassLabel, vfi, patch, Ghost::None, 0);
      new_dw->allocate(gsurfnorm[vfi],lb->gSurfNormLabel, vfi, patch);

      gsurfnorm[vfi].initialize(Vector(0.0,0.0,0.0));

      IntVector lowi(gsurfnorm[vfi].getLowIndex());
      IntVector highi(gsurfnorm[vfi].getHighIndex());

//      cout << "Low" << lowi << endl;
//      cout << "High" << highi << endl;

      // Compute the normals for all of the interior nodes
      for(int i = lowi.x()+1; i < highi.x()-1; i++){
        for(int j = lowi.y()+1; j < highi.y()-1; j++){
          for(int k = lowi.z()+1; k < highi.z()-1; k++){
	     surnor = Vector(
	        -(gmass[vfi][IV(i+1,j,k)] - gmass[vfi][IV(i-1,j,k)])/dx.x(),
         	-(gmass[vfi][IV(i,j+1,k)] - gmass[vfi][IV(i,j-1,k)])/dx.y(), 
	        -(gmass[vfi][IV(i,j,k+1)] - gmass[vfi][IV(i,j,k-1)])/dx.z()); 
	     double length = surnor.length();
	     if(length>0.0){
	    	 gsurfnorm[vfi][IntVector(i,j,k)] = surnor/length;;
	     }
          }
        }
      }

     // Compute normals on the surface nodes assuming a single patch
     // with reflective boundaries.  This needs to be generalized for
     // running in parallel.

      // Compute the normals for the x-surface nodes
      for(int j = lowi.y()+1; j < highi.y()-1; j++){
        for(int k = lowi.z()+1; k < highi.z()-1; k++){
           int i=lowi.x();
	   surnor = Vector(
	      0.0,
	      -(gmass[vfi][IV(i,j+1,k)] - gmass[vfi][IV(i,j-1,k)])/dx.y(), 
	      -(gmass[vfi][IV(i,j,k+1)] - gmass[vfi][IV(i,j,k-1)])/dx.z()); 
	   double length = surnor.length();
	   if(length>0.0){
	  	gsurfnorm[vfi][IntVector(i,j,k)] = surnor/length;;
	   }
           i=highi.x()-1;
	   surnor = Vector(
	      0.0,
	      -(gmass[vfi][IV(i,j+1,k)] - gmass[vfi][IV(i,j-1,k)])/dx.y(), 
	      -(gmass[vfi][IV(i,j,k+1)] - gmass[vfi][IV(i,j,k-1)])/dx.z()); 
	   length = surnor.length();
	   if(length>0.0){
	  	gsurfnorm[vfi][IntVector(i,j,k)] = surnor/length;;
	   }
        }
      }

      // Compute the normals for the y-surface nodes
      for(int i = lowi.x()+1; i < highi.x()-1; i++){
        for(int k = lowi.z()+1; k < highi.z()-1; k++){
           int j=lowi.y();
	   surnor = Vector(
	      -(gmass[vfi][IV(i+1,j,k)] - gmass[vfi][IV(i-1,j,k)])/dx.x(),
	      0.0,
	      -(gmass[vfi][IV(i,j,k+1)] - gmass[vfi][IV(i,j,k-1)])/dx.z()); 
	   double length = surnor.length();
	   if(length>0.0){
	  	gsurfnorm[vfi][IntVector(i,j,k)] = surnor/length;;
	   }
           j=highi.y()-1;
	   surnor = Vector(
	      -(gmass[vfi][IV(i+1,j,k)] - gmass[vfi][IV(i-1,j,k)])/dx.x(),
	      0.0,
	      -(gmass[vfi][IV(i,j,k+1)] - gmass[vfi][IV(i,j,k-1)])/dx.z()); 
	   length = surnor.length();
	   if(length>0.0){
	  	gsurfnorm[vfi][IntVector(i,j,k)] = surnor/length;;
	   }
        }
      }

      // Compute the normals for the z-surface nodes
      for(int i = lowi.x()+1; i < highi.x()-1; i++){
        for(int j = lowi.y()+1; j < highi.y()-1; j++){
           int k=lowi.z();
	   surnor = Vector(
	      -(gmass[vfi][IV(i+1,j,k)] - gmass[vfi][IV(i-1,j,k)])/dx.x(),
	      -(gmass[vfi][IV(i,j+1,k)] - gmass[vfi][IV(i,j-1,k)])/dx.y(), 
	      0.0);
	   double length = surnor.length();
	   if(length>0.0){
	  	gsurfnorm[vfi][IntVector(i,j,k)] = surnor/length;;
	   }
           k=highi.z()-1;
	   surnor = Vector(
	      -(gmass[vfi][IV(i+1,j,k)] - gmass[vfi][IV(i-1,j,k)])/dx.x(),
	      -(gmass[vfi][IV(i,j+1,k)] - gmass[vfi][IV(i,j-1,k)])/dx.y(), 
	      0.0);
	   length = surnor.length();
	   if(length>0.0){
	  	gsurfnorm[vfi][IntVector(i,j,k)] = surnor/length;;
	   }
        }
      }

      new_dw->put(gsurfnorm[vfi],lb->gSurfNormLabel, vfi, patch);

    }
  }

  // Next, interpolate the stress to the grid
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Create arrays for the particle stress and grid stress
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
      ParticleVariable<Matrix3> pstress;
      NCVariable<Matrix3>       gstress;
      new_dw->get(pstress, lb->pStressLabel_preReloc, pset);
      new_dw->allocate(gstress, lb->gStressLabel, vfindex, patch);
      gstress.initialize(Matrix3(0.0));
      
      ParticleVariable<Point> px;
      old_dw->get(px, lb->pXLabel, pset);

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;

         // Get the node indices that surround the cell
         IntVector ni[8];
	 double S[8];
         if(!patch->findCellAndWeights(px[idx], ni, S))
            continue;
         // Add each particles contribution to the local mass & velocity
         // Must use the node indices
         for(int k = 0; k < 8; k++) {
	   if (patch->containsNode(ni[k]))
	     gstress[ni[k]] += pstress[idx] * S[k];
         }
      }
      new_dw->put(gstress, lb->gStressLabel, vfindex, patch);

    }
  }

  // Compute the normal component of the traction
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      NCVariable<Matrix3>      gstress;
      NCVariable<double>       gnormtraction;
      NCVariable<Vector>       surfnorm;
      new_dw->get(gstress,lb->gStressLabel, vfindex, patch, Ghost::None, 0);
      new_dw->get(surfnorm,lb->gSurfNormLabel, vfindex, patch, Ghost::None, 0);
      new_dw->allocate(gnormtraction,lb->gNormTractionLabel, vfindex, patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	gnormtraction[*iter]=
			Dot(surfnorm[*iter]*gstress[*iter],surfnorm[*iter]);
      }

      new_dw->put(gnormtraction,lb->gNormTractionLabel, vfindex, patch);
    }
  }


  // FINALLY, we have all the pieces in place, compute the proper
  // interaction

  // Retrieve necessary data from DataWarehouse
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], lb->gMassLabel,vfindex , patch, Ghost::None, 0);
      new_dw->get(gvelocity_star[vfindex], lb->gVelocityStarLabel,
		  vfindex, patch, Ghost::None, 0);
      new_dw->get(gacceleration[vfindex],lb->gAccelerationLabel,vfindex,patch,
		  Ghost::None, 0);
      new_dw->get(normtraction[vfindex],lb->gNormTractionLabel,vfindex , patch,
		  Ghost::None, 0);
      new_dw->get(gsurfnorm[vfindex],lb->gSurfNormLabel,vfindex , patch,
		  Ghost::None, 0);
    }
  }
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);
  double epsilon_max_max=0.0;

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int  n = 0; n < NVFs; n++){
       centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
       centerOfMassMass+=gmass[n][*iter]; 
    }

    // Apply Coulomb friction contact
    // For grid points with mass calculate velocity
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

      // Loop over velocity fields.  Only proceed if velocity field mass
      // is nonzero (not numerical noise) and the difference from
      // the centerOfMassVelocity is nonzero (More than one velocity
      // field is contributing to grid vertex).
      for(int n = 0; n < NVFs; n++){
        Vector deltaVelocity=gvelocity_star[n][*iter]-centerOfMassVelocity;
        if(!compare(gmass[n][*iter]/centerOfMassMass,0.0)
	   //           && !compare(deltaVelocity.length(),0.0)){
           && !compare(gmass[n][*iter]-centerOfMassMass,0.0)){

          // Apply frictional contact if the surface is in compression
          // or the surface is stress free and surface is approaching.
          // Otherwise apply free surface conditions (do nothing).
          double normalDeltaVelocity=Dot(deltaVelocity,gsurfnorm[n][*iter]);
	  Dvdt=zero;
          if((normtraction[n][*iter] < 0.0) ||
	     (compare(fabs(normtraction[n][*iter]),0.0) &&
              normalDeltaVelocity>0.0)){

	    // Specialize algorithm in case where approach velocity
	    // is in direction of surface normal.
	    if(compare( (deltaVelocity
		 -gsurfnorm[n][*iter]*normalDeltaVelocity).length(),0.0)){
	      Dvdt=-gsurfnorm[n][*iter]*normalDeltaVelocity;
	    }
	    else if(!compare(fabs(normalDeltaVelocity),0.0)){
	      Vector surfaceTangent=
	       (deltaVelocity-gsurfnorm[n][*iter]*normalDeltaVelocity)/
               (deltaVelocity-gsurfnorm[n][*iter]*normalDeltaVelocity).length();
	      double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
	      double frictionCoefficient=
		Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVelocity));
	      Dvdt=
		-gsurfnorm[n][*iter]*normalDeltaVelocity
		-surfaceTangent*frictionCoefficient*fabs(normalDeltaVelocity);
	    }
	    Vector epsilon=(Dvdt/dx)*delT;
	    double epsilon_max=
	      Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
	    epsilon_max_max=max(epsilon_max,epsilon_max_max);
	    if(!compare(epsilon_max,0.0)){
	      epsilon_max=epsilon_max*Max(1.0,
			  gmass[n][*iter]/(centerOfMassMass-gmass[n][*iter]));
	      double ff=Min(epsilon_max,.5)/epsilon_max;
	      Dvdt=Dvdt*ff;
	    }
	    gvelocity_star[n][*iter]+=Dvdt;
	    Dvdt=Dvdt/delT;
	    gacceleration[n][*iter]+=Dvdt;
          }
	}
      }
    }
  }

  //  print out epsilon_max_max
  static int ts=0;
  static ofstream tmpout("max_strain.dat");

  tmpout << ts << " " << epsilon_max_max << endl;
  ts++;

  // Store new velocities and accelerations in DataWarehouse
  for(int n = 0; n < NVFs; n++){
    new_dw->put(gvelocity_star[n], lb->gMomExedVelocityStarLabel, n, patch);
    new_dw->put(gacceleration[n], lb->gMomExedAccelerationLabel, n, patch);
  }
}

void FrictionContact::addComputesAndRequiresInterpolated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{

  int idx = matl->getDWIndex();
  //  const MPMLabel* lb = MPMLabel::getLabels();
  t->requires( old_dw, lb->gNormTractionLabel,idx , patch, Ghost::None, 0);
  t->requires( old_dw, lb->gSurfNormLabel,idx , patch, Ghost::None, 0);
  t->requires( new_dw, lb->gMassLabel, idx, patch, Ghost::None);
  t->requires( new_dw, lb->gVelocityLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityLabel, idx, patch );


}

void FrictionContact::addComputesAndRequiresIntegrated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{

  int idx = matl->getDWIndex();
  //  const MPMLabel* lb = MPMLabel::getLabels();
  t->requires( new_dw, lb->pStressLabel_preReloc, idx, patch,
                         Ghost::AroundNodes, 1);
  t->requires(new_dw, lb->gMassLabel, idx, patch, Ghost::None);
  t->requires(new_dw, lb->gVelocityStarLabel, idx, patch, Ghost::None);
  t->requires(new_dw, lb->gAccelerationLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gNormTractionLabel,idx , patch);
  t->computes( new_dw, lb->gSurfNormLabel,idx , patch);
  t->computes( new_dw, lb->gMomExedVelocityStarLabel, idx, patch);
  t->computes( new_dw, lb->gMomExedAccelerationLabel, idx, patch);
  t->computes( new_dw, lb->gStressLabel, idx, patch);


}

// $Log$
// Revision 1.32  2000/09/25 20:23:20  sparker
// Quiet g++ warnings
//
// Revision 1.31  2000/08/16 22:59:00  bard
// Moved the varlabels to MPMLabel.
//
// Revision 1.30  2000/08/16 20:35:12  bard
// Added logic which makes algorithm more robust in a number of situations.
//
// Revision 1.29  2000/08/08 01:32:43  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.28  2000/07/28 22:13:14  bard
// Added logic to handle degenerate case (previously could cause divide by zero)
// and fixed a typo.
//
// Revision 1.27  2000/07/05 23:43:36  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.26  2000/06/19 23:46:34  guilkey
// changed a requires from pStressLabel to pStressLabel_preReloc.
// Changed corresponding "gets" as well.
//
// Revision 1.25  2000/06/17 07:06:37  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.24  2000/06/15 21:57:08  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.23  2000/05/30 21:07:37  dav
// delt to delT
//
// Revision 1.22  2000/05/30 20:19:08  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.21  2000/05/30 17:08:54  dav
// Changed delt to delT
//
// Revision 1.20  2000/05/26 22:05:40  jas
// Using Singleton class MPMLabel for label management.
//
// Revision 1.19  2000/05/25 23:05:07  guilkey
// Created addComputesAndRequiresInterpolated and addComputesAndRequiresIntegrated
// for each of the three derived Contact classes.  Also, got the NullContact
// class working.  It doesn't do anything besides carry forward the data
// into the "MomExed" variable labels.
//
// Revision 1.18  2000/05/18 23:00:27  guilkey
// Commented out some stupid code that was printing out a big tecplot
// file for every time step.
//
// Revision 1.17  2000/05/11 20:10:16  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.16  2000/05/10 20:02:48  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.15  2000/05/08 22:45:34  guilkey
// Fixed a few stupid errors in the FrictionContact.
//
// Revision 1.14  2000/05/08 21:55:54  guilkey
// Added calculation of surface normals on the boundary.
//
// Revision 1.13  2000/05/08 18:42:46  guilkey
// Added an initializeContact function to all contact classes.  This is
// a null function for all but the FrictionContact.
//
// Revision 1.12  2000/05/06 11:03:25  guilkey
// Removed some hardwired crap.  Now using getLowIndex and getHighIndex.
//
// Revision 1.11  2000/05/06 00:01:49  bard
// Finished frictional contact logic.  Compiles but doesn't yet work.
//
// Revision 1.10  2000/05/05 22:37:27  bard
// Added frictional contact logic.  Compiles but doesn't yet work.
//
// Revision 1.9  2000/05/05 19:32:00  guilkey
// Implemented more of FrictionContact.  Fixed some problems, put in
// some code to write tecplot files :(  .
//
// Revision 1.8  2000/05/05 04:09:08  guilkey
// Uncommented the code which previously wouldn't compile.
//
// Revision 1.7  2000/05/05 02:24:35  guilkey
// Added more stuff to FrictionContact, most of which is currently
// commented out until a compilation issue is resolved.
//
// Revision 1.6  2000/05/02 18:41:18  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
// Revision 1.5  2000/05/02 17:54:27  sparker
// Implemented more of SerialMPM
//
// Revision 1.4  2000/05/02 06:07:14  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.3  2000/04/28 07:35:29  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.2  2000/04/27 21:28:57  jas
// Contact is now created using a factory.
//
// Revision 1.1  2000/04/27 20:00:26  guilkey
// Finished implementing the SingleVelContact class.  Also created
// FrictionContact class which Scott will be filling in to perform
// frictional type contact.
//
