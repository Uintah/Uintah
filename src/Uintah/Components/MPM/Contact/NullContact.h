// NullContact.h

#ifndef __NULL_CONTACT_H__
#define __NULL_CONTACT_H__

#include "Contact.h"

#ifdef WONT_COMPILE_YET

class NullContact : public Contact {
 private:

  // Prevent copying of this class
  // copy constructor
  NullContact(const NullContact &con);
  NullContact& operator=(const NullContact &con);

 public:
   // Constructor
   NullContact();

   // Destructor
   virtual ~NullContact();

   // Basic contact methods
   virtual void exMomInterpolated(const Region* region,
                                  const DataWarehouseP& old_dw,
                                  DataWarehouseP& new_dw);

   virtual void exMomIntegrated(const Region* region,
                                const DataWarehouseP& old_dw,
                                DataWarehouseP& new_dw);

};

#endif

#endif /* __NULL_CONTACT_H__ */

// $Log$
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:13  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
