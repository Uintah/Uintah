#ifndef Packages_Uintah_CCA_Components_Switching_Criteria_Factory_h
#define Packages_Uintah_CCA_Components_Switching_Criteria_Factory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SwitchingCriteria.h>

#include <Packages/Uintah/CCA/Components/SwitchingCriteria/share.h>
namespace Uintah {

  class ProcessorGroup;

  class SCISHARE SwitchingCriteriaFactory
    {
    public:
      // this function has a switch for all known SwitchingCriteria
    
      static SwitchingCriteria* create(ProblemSpecP& ps,
                                       const ProcessorGroup* world);

    };
} // End namespace Uintah

#endif 
