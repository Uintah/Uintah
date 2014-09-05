
#include "FrictionContact.h"
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

using namespace Uintah;
using namespace SCIRun;
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
                                        int matlindex,
                                        DataWarehouseP& new_dw)
{
  NCVariable<double> normtraction;
  NCVariable<Vector> surfnorm;

  new_dw->allocate(normtraction,lb->gNormTractionLabel,matlindex, patch);
  new_dw->allocate(surfnorm,lb->gSurfNormLabel,matlindex, patch);

  normtraction.initialize(0.0);
  surfnorm.initialize(Vector(0.0,0.0,0.0));

  new_dw->put(normtraction,lb->gNormTractionLabel,matlindex, patch);
  new_dw->put(surfnorm,lb->gSurfNormLabel,matlindex, patch);

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

  int numMatls = d_sharedState->getNumMPMMatls();

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity(numMatls);
  vector<NCVariable<double> > normtraction(numMatls);
  vector<NCVariable<Vector> > surfnorm(numMatls);
  
  // Retrieve necessary data from DataWarehouse
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      new_dw->get(gmass[m], lb->gMassLabel,matlindex, patch,
		  Ghost::None, 0);
      new_dw->get(gvelocity[m], lb->gVelocityLabel, matlindex, patch,
		  Ghost::None, 0);
      old_dw->get(normtraction[m],lb->gNormTractionLabel,matlindex,
	   patch, Ghost::None, 0);
      old_dw->get(surfnorm[m],lb->gSurfNormLabel,matlindex, patch,
		  Ghost::None, 0);
  }
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int n = 0; n < numMatls; n++){
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
      for(int n = 0; n < numMatls; n++){
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
  for(int n=0; n< numMatls; n++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
    int matlindex = mpm_matl->getDWIndex();
    new_dw->put(gvelocity[n], lb->gMomExedVelocityLabel, matlindex, patch);
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

  int numMatls = d_sharedState->getNumMPMMatls();

  // This model requires getting the normal component of the
  // surface traction.  The first step is to calculate the
  // surface normals of each object.  Next, interpolate the
  // stress to the grid.  The quantity we want is n^T*stress*n
  // at each node.

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity_star(numMatls);
  vector<NCVariable<Vector> > gacceleration(numMatls);
  vector<NCVariable<double> > normtraction(numMatls);
  vector<NCVariable<Vector> > gsurfnorm(numMatls);

  Vector surnor;

  // First, calculate the gradient of the mass everywhere
  // normalize it, and stick it in surfNorm
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[m], lb->gMassLabel, dwi, patch, Ghost::AroundNodes, 1);
      new_dw->allocate(gsurfnorm[m],lb->gSurfNormLabel, dwi, patch);

      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));

      IntVector low(gsurfnorm[m].getLowIndex());
      IntVector high(gsurfnorm[m].getHighIndex());

      int ILOW,IHIGH,JLOW,JHIGH,KLOW,KHIGH;
      // First, figure out some ranges for for loops
      for(Patch::FaceType face = Patch::startFace;
      		  face <= Patch::endFace; face=Patch::nextFace(face)){
	Patch::BCType bc_type = patch->getBCType(face);

	if(face==Patch::xminus){
	  if(bc_type == Patch::Neighbor) { ILOW = low.x(); }
	  else if(bc_type == Patch::None){ ILOW = low.x()+1; }
	}
	if(face==Patch::xplus){
	  if(bc_type == Patch::Neighbor) { IHIGH = high.x(); }
	  else if(bc_type == Patch::None){ IHIGH = high.x()-1; }
	}
	if(face==Patch::yminus){
	  if(bc_type == Patch::Neighbor) { JLOW = low.y(); }
	  else if(bc_type == Patch::None){ JLOW = low.y()+1; }
	}
	if(face==Patch::yplus){
	  if(bc_type == Patch::Neighbor) { JHIGH = high.y(); }
	  else if(bc_type == Patch::None){ JHIGH = high.y()-1; }
	}
	if(face==Patch::zminus){
	  if(bc_type == Patch::Neighbor) { KLOW = low.z(); }
	  else if(bc_type == Patch::None){ KLOW = low.z()+1; }
	}
	if(face==Patch::zplus){
	  if(bc_type == Patch::Neighbor) { KHIGH = high.z(); }
	  else if(bc_type == Patch::None){ KHIGH = high.z()-1; }
	}
      }

      // Compute the normals for all of the interior nodes
      for(int i = ILOW; i < IHIGH; i++){
        for(int j = JLOW; j < JHIGH; j++){
          for(int k = KLOW; k < KHIGH; k++){
	     surnor = Vector(
	        -(gmass[m][IV(i+1,j,k)] - gmass[m][IV(i-1,j,k)])/dx.x(),
         	-(gmass[m][IV(i,j+1,k)] - gmass[m][IV(i,j-1,k)])/dx.y(), 
	        -(gmass[m][IV(i,j,k+1)] - gmass[m][IV(i,j,k-1)])/dx.z()); 
	     double length = surnor.length();
	     if(length>0.0){
	    	 gsurfnorm[m][IntVector(i,j,k)] = surnor/length;;
	     }
          }
        }
      }


     // Fix the normals on the surface nodes
      for(Patch::FaceType face = Patch::startFace;
      		  face <= Patch::endFace; face=Patch::nextFace(face)){
	Patch::BCType bc_type = patch->getBCType(face);
	// First the nodes which border another patch
	if (bc_type == Patch::Neighbor){
		// Do nothing, we already got those
	}
	// Next the nodes which make up the problem domain
	else if (bc_type == Patch::None) {
	  if(face==Patch::xplus || face==Patch::xminus){
            int I;
            if(face==Patch::xminus){ I=low.x(); }
            if(face==Patch::xplus) { I=high.x()-1; }
            for (int j = JLOW; j<JHIGH; j++) {
              for (int k = KLOW; k<KHIGH; k++) {
           	surnor = Vector( 0.0,
                       -(gmass[m][IV(I,j+1,k)] - gmass[m][IV(I,j-1,k)])/dx.y(),
                       -(gmass[m][IV(I,j,k+1)] - gmass[m][IV(I,j,k-1)])/dx.z());
                double length = surnor.length();
                if(length>0.0){
                   gsurfnorm[m][IntVector(I,j,k)] = surnor/length;;
                }
              }
            }
          }

          if(face==Patch::yplus || face==Patch::yminus){
            int J;
            if(face==Patch::yminus){ J=low.y(); }
            if(face==Patch::yplus) { J=high.y()-1; }
            for (int i = ILOW; i<IHIGH; i++) {
              for (int k = KLOW; k<KHIGH; k++) {
		surnor = Vector(
	               -(gmass[m][IV(i+1,J,k)] - gmass[m][IV(i-1,J,k)])/dx.x(),
			 0.0,
		       -(gmass[m][IV(i,J,k+1)] - gmass[m][IV(i,J,k-1)])/dx.z());
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,J,k)] = surnor/length;;
		}
              }
            }
          }

          if(face==Patch::zplus || face==Patch::zminus){
            int K;
            if(face==Patch::zminus){ K=low.z(); }
            if(face==Patch::zplus) { K=high.z()-1; }
            for (int i = ILOW; i<IHIGH; i++) {
              for (int j = JLOW; j<JHIGH; j++) {
		surnor = Vector(
	               -(gmass[m][IV(i+1,j,K)] - gmass[m][IV(i-1,j,K)])/dx.x(),
	               -(gmass[m][IV(i,j+1,K)] - gmass[m][IV(i,j-1,K)])/dx.y(), 
	               0.0);
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,j,K)] = surnor/length;;
		}
              }
            }
          }

	}
      }

      new_dw->put(gsurfnorm[m],lb->gSurfNormLabel, dwi, patch);
  }

  // Next, interpolate the stress to the grid
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle stress and grid stress
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
                                               Ghost::AroundNodes, 1,
                                               lb->pXLabel);
      ParticleVariable<Matrix3> pstress;
      NCVariable<Matrix3>       gstress;
      new_dw->get(pstress, lb->pStressAfterStrainRateLabel, pset);
      new_dw->allocate(gstress, lb->gStressLabel, matlindex, patch);
      gstress.initialize(Matrix3(0.0));
      
      ParticleVariable<Point> px;
      old_dw->get(px, lb->pXLabel, pset);

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;

         // Get the node indices that surround the cell
         IntVector ni[8];
	 double S[8];
         patch->findCellAndWeights(px[idx], ni, S);
         // Add each particles contribution to the local mass & velocity
         // Must use the node indices
         for(int k = 0; k < 8; k++) {
	   if (patch->containsNode(ni[k]))
	     gstress[ni[k]] += pstress[idx] * S[k];
         }
      }
      new_dw->put(gstress, lb->gStressLabel, matlindex, patch);
  }

  // Compute the normal component of the traction
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();
    NCVariable<Matrix3>      gstress;
    NCVariable<double>       gnormtraction;
    NCVariable<Vector>       surfnorm;
    new_dw->get(gstress, lb->gStressLabel,   matlindex, patch, Ghost::None, 0);
    new_dw->get(surfnorm,lb->gSurfNormLabel, matlindex, patch, Ghost::None, 0);
    new_dw->allocate(gnormtraction,lb->gNormTractionLabel, matlindex, patch);

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	gnormtraction[*iter]=
			Dot(surfnorm[*iter]*gstress[*iter],surfnorm[*iter]);
    }

    new_dw->put(gnormtraction,lb->gNormTractionLabel, matlindex, patch);
  }


  // FINALLY, we have all the pieces in place, compute the proper interaction

  // Retrieve necessary data from DataWarehouse
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      new_dw->get(gmass[m], lb->gMassLabel,matlindex,
					patch,Ghost::None, 0);
      new_dw->get(gvelocity_star[m], lb->gVelocityStarLabel,matlindex,
					patch, Ghost::None, 0);
      new_dw->get(gacceleration[m],lb->gAccelerationLabel,matlindex,
					patch, Ghost::None, 0);
      new_dw->get(normtraction[m],lb->gNormTractionLabel,matlindex,
					patch, Ghost::None, 0);
      new_dw->get(gsurfnorm[m],lb->gSurfNormLabel,matlindex,
					patch, Ghost::None, 0);
  }

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);
  double epsilon_max_max=0.0;

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int  n = 0; n < numMatls; n++){
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
      for(int n = 0; n < numMatls; n++){
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
  //  static int ts=0;
  //  static ofstream tmpout("max_strain.dat");

  //  tmpout << ts << " " << epsilon_max_max << endl;
  //  ts++;

  // Store new velocities and accelerations in DataWarehouse
  for(int n = 0; n < numMatls; n++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
    int matlindex = mpm_matl->getDWIndex();
    new_dw->put(gvelocity_star[n], lb->gMomExedVelocityStarLabel,
						matlindex, patch);
    new_dw->put(gacceleration[n], lb->gMomExedAccelerationLabel,
						matlindex, patch);
  }
}

void FrictionContact::addComputesAndRequiresInterpolated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  t->requires( old_dw, lb->gNormTractionLabel,idx, patch, Ghost::None, 0);
  t->requires( old_dw, lb->gSurfNormLabel,    idx, patch, Ghost::None, 0);
  t->requires( new_dw, lb->gMassLabel,        idx, patch, Ghost::None);
  t->requires( new_dw, lb->gVelocityLabel,    idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityLabel, idx, patch );
}

void FrictionContact::addComputesAndRequiresIntegrated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  t->requires( new_dw, lb->pStressAfterStrainRateLabel, idx, patch,
                        			   Ghost::AroundNodes, 1);
  t->requires(new_dw,  lb->gMassLabel,         idx, patch,
						   Ghost::AroundNodes, 1);
  t->requires(new_dw,  lb->gVelocityStarLabel, idx, patch, Ghost::None);
  t->requires(new_dw,  lb->gAccelerationLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gNormTractionLabel,        idx, patch);
  t->computes( new_dw, lb->gSurfNormLabel,            idx, patch);
  t->computes( new_dw, lb->gMomExedVelocityStarLabel, idx, patch);
  t->computes( new_dw, lb->gMomExedAccelerationLabel, idx, patch);
  t->computes( new_dw, lb->gStressLabel,              idx, patch);
}

