#ifndef Packages_Uintah_CCA_Components_SimulationController_SimulationControllerFactory_h
#define Packages_Uintah_CCA_Components_SimulationController_SimulationControllerFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>

namespace Uintah {

  class ProcessorGroup;

  class SimulationControllerFactory
  {
  public:
    // this function has a switch for all known load balancers
    
    static SimulationController* create(ProblemSpecP& ps,
                                        const ProcessorGroup* world);


  };
} // End namespace Uintah


#endif
