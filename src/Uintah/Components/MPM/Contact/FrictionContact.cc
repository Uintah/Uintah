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
using namespace Uintah::MPM;
using SCICore::Geometry::Vector;
using SCICore::Geometry::IntVector;
using std::vector;
using std::string;


FrictionContact::FrictionContact(ProblemSpecP& ps,
				 SimulationStateP& d_sS)
{
  // Constructor
  IntVector v_f;
  double mu;

  ps->require("vel_fields",v_f);
  ps->require("mu",mu);

  d_sharedState = d_sS;

  gTractionLabel = new VarLabel( "g.traction",
                   NCVariable<double>::getTypeDescription() );

  gSurfNormLabel = new VarLabel( "g.surfnorm",
                   NCVariable<Vector>::getTypeDescription() );

//  gStressLabel   = new VarLabel( "g.stress",
//                   NCVariable<Matrix3>::getTypeDescription() );

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
					const DataWarehouseP&,
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

  // Retrieve necessary data from DataWarehouse
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], gMassLabel,vfindex , region, 0);
      new_dw->get(gvelocity[vfindex], gVelocityLabel, vfindex, region, 0);
    }
  }

  for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int n = 0; n < NVFs; n++){
      centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
      centerOfMassMass+=gmass[n][*iter]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int n = 0; n < NVFs; n++){
	gvelocity[n][*iter] = centerOfMassVelocity;
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

#if 0
  // First, calculate the gradient of the mass everywhere
  // normalize it, and stick it in surfNorm
  NCVariable<Vector> gsurfnorm;
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], gMassLabel,vfindex , region, 0);
      new_dw->allocate(gsurfnorm, gSurfNormLabel, vfindex, region);

      for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
	// Stick calculation of gsurfnorm here
        // WARNING  This assumes dx=dy=dz.  This will be fixed eventually.
	gsurfnorm[*iter] = Vector(
		gmass[vfindex][*iter+onex] - gmass[vfindex][*iter-onex], 
		gmass[vfindex][*iter+oney] - gmass[vfindex][*iter-oney], 
		gmass[vfindex][*iter+onez] - gmass[vfindex][*iter-onez]);
      }
      new_dw->put(gsurfnorm, gSurfNormLabel, vfindex, region);

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
//      new_dw->put(gstress, gStressLabel, vfindex, region);

    }
  }

  // Finally, compute the normal component of the traction
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      NCVariable<Matrix3>      gstress;
      NCVariable<Vector>       gtraction;
      NCVariable<Vector>       gsurfnorm;
      new_dw->get(gstress, gStressLabel, vfindex, region,0);
      new_dw->get(gsurfnorm, gSurfNormLabel, vfindex, region,0);
      new_dw->allocate(gtraction, gTractionLabel, vfindex, region);

      //Compute traction here.  In the morning, when I'm not
      //about to pass out from hunger.

      new_dw->put(gtraction, gTractionLabel, vfindex, region);

    }
  }
#endif


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

    // Set each field's velocity equal to the center of mass velocity
    // and adjust the acceleration of each field to account for this
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int  n = 0; n < NVFs; n++){
        Dvdt = (centerOfMassVelocity - gvelocity_star[n][*iter])/delt;
	gvelocity_star[n][*iter] = centerOfMassVelocity;
	gacceleration[n][*iter]+=Dvdt;
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
