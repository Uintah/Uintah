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
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>
using namespace Uintah::MPM;
using SCICore::Geometry::Vector;
using std::vector;


SingleVelContact::SingleVelContact(const SimulationStateP& d_sS)
{
  // Constructor
   gMassLabel         = new VarLabel( "g.mass",
                        NCVariable<double>::getTypeDescription() );
   gVelocityLabel     = new VarLabel( "g.velocity",
                        NCVariable<Vector>::getTypeDescription() );
   gVelocityStarLabel = new VarLabel( "g.velocity_star",
                        NCVariable<Vector>::getTypeDescription() );
   gAccelerationLabel = new VarLabel( "g.acceleration",
                        NCVariable<Vector>::getTypeDescription() );

  d_sharedState = d_sS;
}

SingleVelContact::~SingleVelContact()
{
  // Destructor

}

void SingleVelContact::exMomInterpolated(const ProcessorContext*,
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

  for(NodeIterator iter = region->begin();
             iter != region->end(); iter++){
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
    new_dw->put(gvelocity[n], gVelocityLabel, n, region);
  }
}

void
SingleVelContact::exMomIntegrated(const ProcessorContext*,
				  const Region* region,
				  const DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{
#if 0
  Vector zero(0.0,0.0,0.0);
  Vector CenterOfMassVelocity(0.0,0.0,0.0);
  Vector CenterOfMassMom(0.0,0.0,0.0);
  double centerOfMassMass;
  int n;

  // Retrieve necessary data from DataWarehouse
  for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
    NCVariable<double> gmass[n];
    new_dw->get(gmass[n], "g.mass", n, region, 0);
    NCVariable<Vector> gvelocity_star[n];
    new_dw->get(gvelocity_star[n], "g.velocity_star", n, region, 0);
    NCVariable<Vector> acceleration[n];
    new_dw->get(acceleration[n], "g.acceleration", n, region, 0);
  }
  SoleVariable<double> delt;
  old_dw->get(delt, "delt");


  for(NodeIterator iter = region->begin();
             iter != region->end(); iter++){
    CenterOfMassMom=zero;
    CenterOfMassMass=0.0; 
    for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
       centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
       centerOfMassMass+=mass[n][i]; 
    }

    // Set each field's velocity equal to the center of mass velocity
    // and adjust the acceleration of each field to account for this
    if(!compare(centerOfMassMass,0.0)){
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for( n=firstMPMVelField; n<(firstMPMVelField+numMPMVelFields); n++){
        Dvdt = (centerOfMassVelocity - velocity_star[n][*iter])/delt;
	gvelocity_star[n][*iter] = centerOfMassVelocity;
	acceleration[n][*iter]+=Dvdt;
      }
    }
  }

  // Store new velocities and accelerations in DataWarehouse
  for( n=firstMPMVelField; n<=numMPMVelFields; n++){
    new_dw->put(gvelocity_star[n], "g.velocity_star", n, region, 0);
    new_dw->put(acceleration[n], "g.acceleration", n, region, 0);
  }
#endif
}

// $Log$
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
