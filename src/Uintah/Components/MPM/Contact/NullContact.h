// NullContact.h

#ifndef __NULL_CONTACT_H__
#define __NULL_CONTACT_H__

#include "Contact.h"
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>

namespace Uintah {
  namespace MPM {

/**************************************

CLASS
   NullContact
   
   Short description...

GENERAL INFORMATION

   NullContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Null

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class NullContact : public Contact {
    private:
      
      // Prevent copying of this class
      // copy constructor
      NullContact(const NullContact &con);
      NullContact& operator=(const NullContact &con);
      
    public:
      // Constructor
      NullContact(ProblemSpecP& ps,SimulationStateP& ss);
      
      // Destructor
      virtual ~NullContact();
      
      // Basic contact methods
      virtual void exMomInterpolated(const ProcessorContext*,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw);
      
      virtual void exMomIntegrated(const ProcessorContext*,
				   const Region* region,
				   const DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);
      
    };
    
  } // end namespace MPM
} // end namespace Uintah

// $Log$
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

#endif /* __NULL_CONTACT_H__ */

