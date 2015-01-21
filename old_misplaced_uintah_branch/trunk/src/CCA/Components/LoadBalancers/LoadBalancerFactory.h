#ifndef Packages_Uintah_CCA_Components_LoadBalancer_LoadBalancerFactory_h
#define Packages_Uintah_CCA_Components_LoadBalancer_LoadBalancerFactory_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/LoadBalancers/LoadBalancerCommon.h>

#include <CCA/Components/LoadBalancers/uintahshare.h>
namespace Uintah {

  class ProcessorGroup;

  class UINTAHSHARE LoadBalancerFactory
  {
  public:
    // this function has a switch for all known load balancers
    
    static LoadBalancerCommon* create(ProblemSpecP& ps,
                                      const ProcessorGroup* world);


  };
} // End namespace Uintah


#endif
