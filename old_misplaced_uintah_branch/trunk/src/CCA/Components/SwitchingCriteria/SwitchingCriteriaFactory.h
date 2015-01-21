#ifndef Packages_Uintah_CCA_Components_Switching_Criteria_Factory_h
#define Packages_Uintah_CCA_Components_Switching_Criteria_Factory_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SwitchingCriteria.h>

#include <CCA/Components/SwitchingCriteria/uintahshare.h>
namespace Uintah {

  class ProcessorGroup;

  class UINTAHSHARE SwitchingCriteriaFactory
    {
    public:
      // this function has a switch for all known SwitchingCriteria
    
      static SwitchingCriteria* create(ProblemSpecP& ps,
                                       const ProcessorGroup* world);

    };
} // End namespace Uintah

#endif 
