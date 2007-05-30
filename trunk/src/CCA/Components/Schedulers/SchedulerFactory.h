#ifndef Packages_Uintah_CCA_Components_Scheduler_SchedulerFactory_h
#define Packages_Uintah_CCA_Components_Scheduler_SchedulerFactory_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Ports/Output.h>

#include <CCA/Components/Schedulers/share.h>
namespace Uintah {

  class ProcessorGroup;
 
  class SCISHARE SchedulerFactory
  {
  public:
    // this function has a switch for all known load balancers
    
    static SchedulerCommon* create(ProblemSpecP& ps,
                                   const ProcessorGroup* world,
                                   Output* ouput);


  };
} // End namespace Uintah


#endif
