// NullContact.cc
//
// One of the derived Contact classes.  This particular
// class is used when no contact is desired.  This would
// be used for example when a single velocity field is
// present in the problem, so doing contact wouldn't make
// sense.

#include "NullContact.h"

#ifdef WONT_COMPILE_YET

NullContact::NullContact()
{
  // Constructor

}

NullContact::~NullContact()
{
  // Destructor

}

void NullContact::exMomInterpolated(const Region* region,
                                  const DataWarehouseP& old_dw,
                                  DataWarehouseP& new_dw)
{

}

void NullContact::exMomIntegrated(const Region* region,
                                  const DataWarehouseP& old_dw,
                                  DataWarehouseP& new_dw)
{

}

#endif

// $Log$
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:13  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
