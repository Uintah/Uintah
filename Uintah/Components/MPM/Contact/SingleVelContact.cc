//
// $Id$
//

// SingleVel.cc
//
// One of the derived Contact classes.  This particular
// class contains methods for recapturing single velocity
// field behavior from objects belonging to multiple velocity
// fields.  The main purpose of this type of contact is to
// ensure that one can get the same answer using prescribed
// contact as can be gotten using "automatic" contact.

#include "SingleVelContact.h"
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
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/VarLabel.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace std;
using namespace Uintah::MPM;
using SCICore::Geometry::Vector;
using SCICore::Geometry::IntVector;
using std::vector;

SingleVelContact::SingleVelContact(ProblemSpecP& ps, 
				    SimulationStateP& d_sS)
{
  // Constructor

  IntVector v_f;
  ps->require("vel_fields",v_f);
  std::cout << "vel_fields = " << v_f << endl;
  
  d_sharedState = d_sS;
}

SingleVelContact::~SingleVelContact()
{
  // Destructor

}

void SingleVelContact::initializeContact(const Patch* /*patch*/,
					 int /*vfindex*/,
					 DataWarehouseP& /*new_dw*/)
{

}

void SingleVelContact::exMomInterpolated(const ProcessorGroup*,
					 const Patch* patch,
					 DataWarehouseP&,
					 DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMPMMatls();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity(numMatls);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int vfindex = mpm_matl->getVFIndex();
    new_dw->get(gmass[vfindex], lb->gMassLabel,vfindex , patch,
		  Ghost::None, 0);
    new_dw->get(gvelocity[vfindex], lb->gVelocityLabel, vfindex, patch,
		  Ghost::None, 0);
  }

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int n = 0; n < numMatls; n++){
      centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
      centerOfMassMass+=gmass[n][*iter]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int n = 0; n < numMatls; n++){
	gvelocity[n][*iter] = centerOfMassVelocity;
      }
    }
  }

  // Store new velocities in DataWarehouse
  for(int n=0; n< numMatls; n++){
    new_dw->put(gvelocity[n], lb->gMomExedVelocityLabel, n, patch);
  }
}

void SingleVelContact::exMomIntegrated(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt;
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMPMMatls();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity_star(numMatls);
  vector<NCVariable<Vector> > gacceleration(numMatls);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int vfindex = mpm_matl->getVFIndex();
    new_dw->get(gmass[vfindex],lb->gMassLabel,vfindex ,patch, Ghost::None, 0);
    new_dw->get(gvelocity_star[vfindex], lb->gVelocityStarLabel, vfindex,
		  patch, Ghost::None, 0);
    new_dw->get(gacceleration[vfindex], lb->gAccelerationLabel, vfindex,
		  patch, Ghost::None, 0);
  }

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
    centerOfMassMom=zero;
    centerOfMassMass=0.0; 
    for(int  n = 0; n < numMatls; n++){
       centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
       centerOfMassMass+=gmass[n][*iter]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    // and adjust the acceleration of each field to account for this
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int  n = 0; n < numMatls; n++){
        Dvdt = (centerOfMassVelocity - gvelocity_star[n][*iter])/delT;
	gvelocity_star[n][*iter] = centerOfMassVelocity;
	gacceleration[n][*iter]+=Dvdt;
      }
    }
  }

  // Store new velocities and accelerations in DataWarehouse
  for(int n = 0; n < numMatls; n++){
    new_dw->put(gvelocity_star[n], lb->gMomExedVelocityStarLabel, n, patch);
    new_dw->put(gacceleration[n], lb->gMomExedAccelerationLabel, n, patch);
  }
}

void SingleVelContact::addComputesAndRequiresInterpolated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  t->requires( new_dw, lb->gMassLabel, idx, patch, Ghost::None);
  t->requires( new_dw, lb->gVelocityLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityLabel, idx, patch );

}

void SingleVelContact::addComputesAndRequiresIntegrated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& old_dw,
                                             DataWarehouseP& new_dw) const
{

  int idx = matl->getDWIndex();
  t->requires(new_dw, lb->gMassLabel,         idx, patch, Ghost::None);
  t->requires(new_dw, lb->gVelocityStarLabel, idx, patch, Ghost::None);
  t->requires(new_dw, lb->gAccelerationLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityStarLabel, idx, patch);
  t->computes( new_dw, lb->gMomExedAccelerationLabel, idx, patch);

}

// $Log$
// Revision 1.25  2000/11/07 22:52:22  guilkey
// Changed the way that materials are looped over.  Instead of each
// function iterating over all materials, and then figuring out which ones
// are MPMMaterials on the fly, SimulationState now stores specific information
// about MPMMaterials, so that for doing MPM, only those materials are returned
// and then looped over.  This will make coupling with a cfd code easier I hope.
//
// Revision 1.24  2000/09/25 20:23:20  sparker
// Quiet g++ warnings
//
// Revision 1.23  2000/07/05 23:43:36  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.22  2000/06/17 07:06:38  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.21  2000/05/30 21:07:37  dav
// delt to delT
//
// Revision 1.20  2000/05/30 20:19:10  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.19  2000/05/30 17:08:54  dav
// Changed delt to delT
//
// Revision 1.18  2000/05/26 21:37:35  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.17  2000/05/25 23:05:10  guilkey
// Created addComputesAndRequiresInterpolated and addComputesAndRequiresIntegrated
// for each of the three derived Contact classes.  Also, got the NullContact
// class working.  It doesn't do anything besides carry forward the data
// into the "MomExed" variable labels.
//
// Revision 1.16  2000/05/11 20:10:17  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.15  2000/05/10 20:02:49  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.14  2000/05/08 22:45:34  guilkey
// Fixed a few stupid errors in the FrictionContact.
//
// Revision 1.13  2000/05/08 21:55:54  guilkey
// Added calculation of surface normals on the boundary.
//
// Revision 1.12  2000/05/08 18:42:46  guilkey
// Added an initializeContact function to all contact classes.  This is
// a null function for all but the FrictionContact.
//
// Revision 1.11  2000/05/02 18:41:18  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
// Revision 1.10  2000/05/02 17:54:27  sparker
// Implemented more of SerialMPM
//
// Revision 1.9  2000/05/02 06:07:14  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.8  2000/04/28 07:35:29  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.7  2000/04/27 21:28:58  jas
// Contact is now created using a factory.
//
// Revision 1.6  2000/04/27 20:00:26  guilkey
// Finished implementing the SingleVelContact class.  Also created
// FrictionContact class which Scott will be filling in to perform
// frictional type contact.
//
// Revision 1.5  2000/04/26 06:48:21  sparker
// Streamlined namespaces
//
// Revision 1.4  2000/04/25 22:57:30  guilkey
// Fixed Contact stuff to include VarLabels, SimulationState, etc, and
// made more of it compile.
//
// Revision 1.3  2000/04/20 23:21:02  dav
// updated to match Contact.h
//
// Revision 1.2  2000/03/21 01:29:41  dav
// working to make MPM stuff compile successfully
//
// Revision 1.1  2000/03/20 23:50:44  dav
// renames SingleVel to SingleVelContact
//
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:13  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
