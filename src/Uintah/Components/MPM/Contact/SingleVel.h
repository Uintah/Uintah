// SingleVel.h

#ifndef __SINGLE_VEL_H__
#define __SINGLE_VEL_H__

#include "Contact.h"

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

#endif __SINGLE_VEL_H__

// $Log$
// Revision 1.1  2000/03/16 01:05:14  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
