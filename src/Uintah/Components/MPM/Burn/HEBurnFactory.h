#ifndef _HEBURNFACTORY_H_
#define _HEBURNFACTORY_H_

#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
   namespace MPM {
      class HEBurn;

      class HEBurnFactory
      {
      public:
	 static HEBurn* create(ProblemSpecP& ps);
	 
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah

#endif /* _HEBURNFACTORY_H_ */

// $Log$
// Revision 1.1  2000/06/02 22:48:25  jas
// Added infrastructure for Burn models.
//
