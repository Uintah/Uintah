/* REFERENCED */
static char *id="@(#) $Id$";

// Friction.cc
//

#include "FrictionContact.h"
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>
#include <iostream>
#include <fstream>

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

  gNormTractionLabel = new VarLabel( "g.normtraction",
                   NCVariable<double>::getTypeDescription() );

  gSurfNormLabel = new VarLabel( "g.surfnorm",
                   NCVariable<Vector>::getTypeDescription() );

  gStressLabel   = new VarLabel( "g.stress",
                   NCVariable<Matrix3>::getTypeDescription() );

  pStressLabel   = new VarLabel( "p.stress",
                   ParticleVariable<Matrix3>::getTypeDescription() );

  pXLabel        = new VarLabel( "p.x",
	           ParticleVariable<Point>::getTypeDescription(),
                   VarLabel::PositionVariable);

}

FrictionContact::~FrictionContact()
{
  // Destructor

}

void FrictionContact::exMomInterpolated(const ProcessorContext*,
					const Region* region,
					const DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMatls();
  int NVFs = d_sharedState->getNumVelFields();

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  vector<NCVariable<double> > gmass(NVFs);
  vector<NCVariable<Vector> > gvelocity(NVFs);
  vector<NCVariable<double> > normtraction(NVFs);
  vector<NCVariable<Vector> > surfnorm(NVFs);

  // Retrieve necessary data from DataWarehouse
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], gMassLabel,vfindex , region, 0);
      new_dw->get(gvelocity[vfindex], gVelocityLabel, vfindex, region, 0);
      old_dw->get(normtraction[vfindex],gNormTractionLabel,vfindex , region, 0);
      old_dw->get(surfnorm[vfindex], gSurfNormLabel,vfindex , region, 0);
    }
  }

  for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
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

      // Loop over velocity fields.  Only proceed if velocity field
      // is nonzero (not numerical noise) and the difference from
      // the centerOfMassVelocity is nonzero (More than one velocity
      // field is contributing to grid vertex).
      for(int n = 0; n < NVFs; n++){
        Vector deltaVelocity=gvelocity[n][*iter]-centerOfMassVelocity;
        if(!compare(gvelocity[n][*iter].length(),0.0)
           && !compare(deltaVelocity.length(),0.0)){

          // Apply frictional contact if the surface is in compression
          // or the surface is stress free and surface is approaching.
          // Otherwise apply free surface conditions (do nothing).
          double normalDeltaVelocity=Dot(deltaVelocity,surfnorm[n][*iter]);
          if((normtraction[n][*iter] < 0.0) ||
             (!compare(fabs(normtraction[n][*iter]),0.0) &&
              normalDeltaVelocity>0.0)){

              // Specialize algorithm in case where approach velocity
              // is in direction of surface normal.
              if(compare( (deltaVelocity
                        -surfnorm[n][*iter]*normalDeltaVelocity).length(),0.0)){
                gvelocity[n][*iter]-= surfnorm[n][*iter]*normalDeltaVelocity;
              }
	      else{
                Vector surfaceTangent=
		(deltaVelocity-surfnorm[n][*iter]*normalDeltaVelocity)/
                (deltaVelocity-surfnorm[n][*iter]*normalDeltaVelocity).length();
                double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                double frictionCoefficient=
                  Min(d_mu,tangentDeltaVelocity/normalDeltaVelocity);
                gvelocity[n][*iter]-=
                  (surfnorm[n][*iter]+surfaceTangent*frictionCoefficient)*
                                                      normalDeltaVelocity;
	     }

          }
	}
      }
    }
  }

  // Store new velocities in DataWarehouse
  for(int n=0; n< NVFs; n++){
    new_dw->put(gvelocity[n], gMomExedVelocityLabel, n, region);
  }
}

void FrictionContact::exMomIntegrated(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt;
  double centerOfMassMass;
  IntVector onex(1,0,0), oney(0,1,0), onez(0,0,1);
  typedef IntVector IV;
  Vector dx = region->dCell();

  int numMatls = d_sharedState->getNumMatls();
  int NVFs = d_sharedState->getNumVelFields();

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
  vector<NCVariable<Vector> > surfnorm(NVFs);

  Vector surnor;

  // First, calculate the gradient of the mass everywhere
  // normalize it, and stick it in surfNorm
  NCVariable<Vector> gsurfnorm;
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfi = matl->getVFIndex();
      new_dw->get(gmass[vfi], gMassLabel,vfi , region, 0);
      new_dw->allocate(gsurfnorm, gSurfNormLabel, vfi, region);

//      IntVector lowi(gsurfnorm.getLowIndex());
//      IntVector highi(gsurfnorm.getHighIndex());
      IntVector lowi(0,0,0);
      IntVector highi(20,20,3);

      for(int i = lowi.x()+1; i < highi.x(); i++){
        for(int j = lowi.y()+1; j < highi.y(); j++){
          for(int k = lowi.z()+1; k < highi.z(); k++){
             // WARNING  This assumes dx=dy=dz.  This will be fixed eventually.
//	     gsurfnorm[IntVector(i,j,k)] = Vector(
	     surnor = Vector(
	        -(gmass[vfi][IV(i+1,j,k)] - gmass[vfi][IV(i-1,j,k)])/dx.x(),
         	-(gmass[vfi][IV(i,j+1,k)] - gmass[vfi][IV(i,j-1,k)])/dx.y(), 
	        -(gmass[vfi][IV(i,j,k+1)] - gmass[vfi][IV(i,j,k-1)])/dx.z()); 
		double length = surnor.length();
		if(length>0.0){
	    	 gsurfnorm[IntVector(i,j,k)] = surnor/length;;
		}
          }
        }
      }

      new_dw->put(gsurfnorm, gSurfNormLabel, vfi, region);

    }
  }

  IntVector lowi(0,0,0);
  IntVector highi(20,20,3);
  ofstream tfile("tecplotfile");
  int I=21,J=21,K=4;
  static int ts=0;
  tfile << "TITLE = \"Time Step # " << ts <<"\"," << endl;
  tfile << "VARIABLES = X,Y,Z,SNX,SNY,SNZ" << endl;
  tfile << "ZONE T=\"GRID\", I=" << I <<", J="<< J <<" K="<< K;
  tfile <<", F=BLOCK" << endl;
  new_dw->get(gsurfnorm, gSurfNormLabel, 0, region,0);
  int n=0;
  for(int k = lowi.z(); k <= highi.z(); k++){
    for(int j = lowi.y(); j <= highi.y(); j++){
      for(int i = lowi.x(); i <= highi.x(); i++){
	  tfile << i*dx.x() << " ";
	  n++;
	  if((n % 20) == 0){ tfile << endl; }
     }
    }
  }
  cout << n << endl;
  tfile << endl;
  n = 0;
  for(int k = lowi.z(); k <= highi.z(); k++){
    for(int j = lowi.y(); j <= highi.y(); j++){
      for(int i = lowi.x(); i <= highi.x(); i++){
	  tfile << j*dx.y() << " ";
	  n++;
	  if((n % 20) == 0){ tfile << endl; }
      }
    }
  }
  tfile << endl;
  n = 0;
  for(int k = lowi.z(); k <= highi.z(); k++){
    for(int j = lowi.y(); j <= highi.y(); j++){
      for(int i = lowi.x(); i <= highi.x(); i++){
	  tfile << k*dx.z() << " ";
	  n++;
	  if((n % 20) == 0){ tfile << endl; }
      }
    }
  }
  tfile << endl;
  n = 0;
  for(int k = lowi.z(); k <= highi.z(); k++){
    for(int j = lowi.y(); j <= highi.y(); j++){
      for(int i = lowi.x(); i <= highi.x(); i++){
        tfile << gsurfnorm[IntVector(i,j,k)].x() << " " ;
        n++;
        if((n % 20) == 0){ tfile << endl; }
      }
    }
  }
  tfile << endl;
  n = 0;
  for(int k = lowi.z(); k <= highi.z(); k++){
    for(int j = lowi.y(); j <= highi.y(); j++){
      for(int i = lowi.x(); i <= highi.x(); i++){
        tfile << gsurfnorm[IntVector(i,j,k)].y() << " " ;
        n++;
        if((n % 20) == 0){ tfile << endl; }
      }
    }
  }
  tfile << endl;
  n = 0;
  for(int k = lowi.z(); k <= highi.z(); k++){
    for(int j = lowi.y(); j <= highi.y(); j++){
      for(int i = lowi.x(); i <= highi.x(); i++){
        tfile << gsurfnorm[IntVector(i,j,k)].z() << " " ;
        n++;
        if((n % 20) == 0){ tfile << endl; }
      }
    }
  }
  tfile << endl;
 
  // Next, interpolate the stress to the grid
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Create arrays for the particle stress and grid stress
      ParticleVariable<Matrix3> pstress;
      NCVariable<Matrix3>       gstress;
      new_dw->get(pstress, pStressLabel, matlindex, region, 0);
      new_dw->allocate(gstress, gStressLabel, vfindex, region);
      gstress.initialize(Matrix3(0.0));

      ParticleVariable<Point> px;
      old_dw->get(px, pXLabel, matlindex, region, 0);


      ParticleSubset* pset = pstress.getParticleSubset();
      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;

         // Get the node indices that surround the cell
         IntVector ni[8];
	 double S[8];
         if(!region->findCellAndWeights(px[idx], ni, S))
            continue;
         // Add each particles contribution to the local mass & velocity
         // Must use the node indices
         for(int k = 0; k < 8; k++) {
             gstress[ni[k]] += pstress[idx] * S[k];
         }
      }
      new_dw->put(gstress, gStressLabel, vfindex, region);

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
      NCVariable<Vector>       gsurfnorm;
      new_dw->get(gstress, gStressLabel, vfindex, region,0);
      new_dw->get(gsurfnorm, gSurfNormLabel, vfindex, region,0);
      new_dw->allocate(gnormtraction, gNormTractionLabel, vfindex, region);

      for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
//	gnormtraction[*iter]=SCICore::Geometry::Dot(gsurfnorm[*iter]*gstress[*iter],gsurfnorm[*iter]);
	gnormtraction[*iter]=
			Dot(gsurfnorm[*iter]*gstress[*iter],gsurfnorm[*iter]);
      }

      new_dw->put(gnormtraction, gNormTractionLabel, vfindex, region);
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
      new_dw->get(gmass[vfindex], gMassLabel,vfindex , region, 0);
      new_dw->get(gvelocity_star[vfindex], gVelocityStarLabel,
						 vfindex, region, 0);
      new_dw->get(gacceleration[vfindex],gAccelerationLabel,vfindex,region,0);
      new_dw->get(normtraction[vfindex],gNormTractionLabel,vfindex , region, 0);
      new_dw->get(surfnorm[vfindex], gSurfNormLabel,vfindex , region, 0);
    }
  }
  delt_vartype delt;
  old_dw->get(delt, deltLabel);

  for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
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

      // Loop over velocity fields.  Only proceed if velocity field
      // is nonzero (not numerical noise) and the difference from
      // the centerOfMassVelocity is nonzero (More than one velocity
      // field is contributing to grid vertex).
      for(int n = 0; n < NVFs; n++){
        Vector deltaVelocity=gvelocity_star[n][*iter]-centerOfMassVelocity;
        if(!compare(gvelocity_star[n][*iter].length(),0.0)
           && !compare(deltaVelocity.length(),0.0)){

          // Apply frictional contact if the surface is in compression
          // or the surface is stress free and surface is approaching.
          // Otherwise apply free surface conditions (do nothing).
          double normalDeltaVelocity=Dot(deltaVelocity,surfnorm[n][*iter]);
          if((normtraction[n][*iter] < 0.0) ||
	     (!compare(fabs(normtraction[n][*iter]),0.0) &&
              normalDeltaVelocity>0.0)){

	    // Specialize algorithm in case where approach velocity
	    // is in direction of surface normal.
	    Dvdt = -gvelocity_star[n][*iter];
	    if(compare( (deltaVelocity
			 -surfnorm[n][*iter]*normalDeltaVelocity).length(),0.0)){
	      gvelocity_star[n][*iter]-= surfnorm[n][*iter]*normalDeltaVelocity;
	    }
	    else{
	      Vector surfaceTangent=
		(deltaVelocity-surfnorm[n][*iter]*normalDeltaVelocity)/
                (deltaVelocity-surfnorm[n][*iter]*normalDeltaVelocity).length();
	      double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
	      double frictionCoefficient=
		Min(d_mu,tangentDeltaVelocity/normalDeltaVelocity);
	      gvelocity_star[n][*iter]-=
		(surfnorm[n][*iter]+surfaceTangent*frictionCoefficient)*
		normalDeltaVelocity;
	    }
	    Dvdt+=gvelocity_star[n][*iter];
	    Dvdt=Dvdt/delt;
	    gacceleration[n][*iter]+=Dvdt;
          }
	}
      }
    }
  }

  // Store new velocities and accelerations in DataWarehouse
  for(int n = 0; n < NVFs; n++){
    new_dw->put(gvelocity_star[n], gMomExedVelocityStarLabel, n, region);
    new_dw->put(gacceleration[n], gMomExedAccelerationLabel, n, region);
  }
}

// $Log$
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
