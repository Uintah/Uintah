#ifndef _CONTACTFACTORY_H_
#define _CONTACTFACTORY_H_

#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/SimulationStateP.h>

namespace Uintah {
  namespace MPM {
    class Contact;
    
    class ContactFactory
      {
      public:
	
	
	// this function has a switch for all known mat_types
	// and calls the proper class' readParameters()
	// addMaterial() calls this
	static Contact* create(const ProblemSpecP& ps,SimulationStateP& ss);
	
	
      };
    
  } // end namespace MPM
} // end namespace Uintah


#endif /* _CONTACTFACTORY_H_ */

// $Log$
// Revision 1.1  2000/04/27 21:28:57  jas
// Contact is now created using a factory.
//
