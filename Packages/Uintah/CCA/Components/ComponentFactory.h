#ifndef Packages_Uintah_CCA_Components_Component_ComponentFactory_h
#define Packages_Uintah_CCA_Components_Component_ComponentFactory_h

#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class ProcessorGroup;

  class SimulationComponent {
  public:
    SimulationComponent() {
      d_sim = 0;
      d_comp = 0;
    };
    ~SimulationComponent() {
      delete d_sim;
    };
    SimulationInterface* d_sim;
    UintahParallelComponent* d_comp;
  };

  class ComponentFactory  {
  
  public:
    // this function has a switch for all known components
    
    static SimulationComponent* create(ProblemSpecP& ps,
                                       const ProcessorGroup* world);


  };
} // End namespace Uintah


#endif
