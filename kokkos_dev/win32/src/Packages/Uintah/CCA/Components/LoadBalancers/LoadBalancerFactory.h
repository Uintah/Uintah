#ifndef Packages_Uintah_CCA_Components_LoadBalancer_LoadBalancerFactory_h
#define Packages_Uintah_CCA_Components_LoadBalancer_LoadBalancerFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>

namespace Uintah {

  class ProcessorGroup;

  class LoadBalancerFactory
  {
  public:
    // this function has a switch for all known load balancers
    
    static LoadBalancerCommon* create(ProblemSpecP& ps,
                                      const ProcessorGroup* world);


  };
} // End namespace Uintah


#endif
