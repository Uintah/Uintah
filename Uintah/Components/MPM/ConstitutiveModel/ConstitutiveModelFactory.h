#ifndef _CONSTITUTIVEMODELFACTORY_H_
#define _CONSTITUTIVEMODELFACTORY_H_

#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
   namespace MPM {
      class ConstitutiveModel;

      class ConstitutiveModelFactory
      {
      public:
	 
	 
	 // this function has a switch for all known mat_types
	 // and calls the proper class' readParameters()
	 // addMaterial() calls this
	 static ConstitutiveModel* create(ProblemSpecP ps);
	 
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah


#endif /* _CONSTITUTIVEMODELFACTORY_H_ */
