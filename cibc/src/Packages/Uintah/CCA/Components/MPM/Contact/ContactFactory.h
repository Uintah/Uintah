#ifndef _CONTACTFACTORY_H_
#define _CONTACTFACTORY_H_

#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {

  class Contact;
  class MPMLabel;
  class MPMFlags;

  class ContactFactory
  {
  public:
	
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static Contact* create(const ProcessorGroup* myworld,
                           const ProblemSpecP& ps,SimulationStateP& ss,
			   MPMLabel* lb, MPMFlags* MFlag);
  };
} // End namespace Uintah
  
#endif /* _CONTACTFACTORY_H_ */

