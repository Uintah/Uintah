/* REFERENCED */
static char *id="@(#) $Id$";

// NullContact.cc
//
// One of the derived Contact classes.  This particular
// class is used when no contact is desired.  This would
// be used for example when a single velocity field is
// present in the problem, so doing contact wouldn't make
// sense.

#include "NullContact.h"

using namespace Uintah::MPM;

NullContact::NullContact(ProblemSpecP& /*ps*/,SimulationStateP& /*ss*/)
{
  // Constructor
 

}

NullContact::~NullContact()
{
  // Destructor

}

void NullContact::initializeContact(const Region* /*region*/,
                                    int /*vfindex*/,
                                    DataWarehouseP& /*new_dw*/)
{

}

void NullContact::exMomInterpolated(const ProcessorContext*,
				    const Region*,
				    DataWarehouseP& /*old_dw*/,
				    DataWarehouseP& /*new_dw*/)
{
  

}

void NullContact::exMomIntegrated(const ProcessorContext*,
				  const Region*,
                                  DataWarehouseP& /*old_dw*/,
                                  DataWarehouseP& /*new_dw*/)
{

  
}

// $Log$
// Revision 1.9  2000/05/11 20:10:17  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.8  2000/05/10 20:02:48  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.7  2000/05/08 18:42:46  guilkey
// Added an initializeContact function to all contact classes.  This is
// a null function for all but the FrictionContact.
//
// Revision 1.6  2000/05/02 06:07:14  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.5  2000/04/27 21:28:58  jas
// Contact is now created using a factory.
//
// Revision 1.4  2000/04/26 06:48:20  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/03/20 23:50:44  dav
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
