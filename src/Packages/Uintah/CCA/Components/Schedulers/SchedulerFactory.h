#ifndef Packages_Uintah_CCA_Components_Scheduler_SchedulerFactory_h
#define Packages_Uintah_CCA_Components_Scheduler_SchedulerFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/Output.h>

namespace Uintah {

  class ProcessorGroup;
 
  class SchedulerFactory
  {
  public:
    // this function has a switch for all known load balancers
    
    static SchedulerCommon* create(ProblemSpecP& ps,
                                   const ProcessorGroup* world,
                                   Output* ouput);


  };
} // End namespace Uintah


#endif
