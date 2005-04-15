#ifndef _EQUATION_OF_STATE_FACTORY_H_
#define _EQUATION_OF_STATE_FACTORY_H_

#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
   namespace ICESpace {
      class EquationOfState;

      class EquationOfStateFactory
      {
      public:
	 // this function has a switch for all known mat_types
	 // and calls the proper class' readParameters()
	 // addMaterial() calls this
	 static EquationOfState* create(ProblemSpecP& ps);
      };
      
   } // end namespace ICE
} // end namespace Uintah

#endif /*_EQUATION_OF_STATE_FACTORY_H_ */

//$Log$
//Revision 1.1.2.1  2000/10/19 05:17:41  sparker
//Merge changes from main branch into csafe_risky1
//
//Revision 1.1  2000/10/06 04:02:16  jas
//Move into a separate EOS directory.
//
//Revision 1.2  2000/10/04 20:17:52  jas
//Change namespace ICE to ICESpace.
//
//Revision 1.1  2000/10/04 19:26:14  guilkey
//Initial commit of some classes to help mainline ICE.
//
