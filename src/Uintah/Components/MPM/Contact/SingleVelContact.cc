/* REFERENCED */
static char *id="@(#) $Id$";

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
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/VarTypes.h>
#include <vector>
#include <iostream>
#include <fstream>

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

void SingleVelContact::initializeContact(const Region* region,
					 int vfindex,
					 DataWarehouseP& new_dw)
{

}

void SingleVelContact::exMomInterpolated(const ProcessorContext*,
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

  cout << "NVFs " << NVFs << endl;

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(NVFs);
  vector<NCVariable<Vector> > gvelocity(NVFs);
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

void SingleVelContact::exMomIntegrated(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
  Vector zero(0.0,0.0,0.0);
  Vector centerOfMassVelocity(0.0,0.0,0.0);
  Vector centerOfMassMom(0.0,0.0,0.0);
  Vector Dvdt;
  double centerOfMassMass;

  int numMatls = d_sharedState->getNumMatls();
  int NVFs = d_sharedState->getNumVelFields();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(NVFs);
  vector<NCVariable<Vector> > gvelocity_star(NVFs);
  vector<NCVariable<Vector> > gacceleration(NVFs);
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(gmass[vfindex], gMassLabel,vfindex , region, 0);
      new_dw->get(gvelocity_star[vfindex], gVelocityStarLabel, vfindex, region, 0);
      new_dw->get(gacceleration[vfindex], gAccelerationLabel, vfindex, region, 0);
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
