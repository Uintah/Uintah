// SingleVel.h

#ifndef __SINGLE_VEL_H__
#define __SINGLE_VEL_H__

#include "Contact.h"

namespace Uintah {
namespace Components {

/**************************************

CLASS
   SingleVelContact
   
   Short description...

GENERAL INFORMATION

   SingleVelContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Single_Velocity

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class SingleVelContact : public Contact {
 private:

  // Prevent copying of this class
  // copy constructor
  SingleVelContact(const SingleVelContact &con);
  SingleVelContact& operator=(const SingleVelContact &con);

 public:
   // Constructor
   SingleVelContact();

   // Destructor
   virtual ~SingleVelContact();

   // Basic contact methods
   virtual void exMomInterpolated(const Region* region,
                                  const DataWarehouseP& old_dw,
                                  DataWarehouseP& new_dw);

   virtual void exMomIntegrated(const Region* region,
                                const DataWarehouseP& old_dw,
                                DataWarehouseP& new_dw);

};

} // end namespace Components
} // end namespace Uintah

#endif /* __SINGLE_VEL_H__ */

// $Log$
// Revision 1.1  2000/03/20 23:50:44  dav
// renames SingleVel to SingleVelContact
//
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:14  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
