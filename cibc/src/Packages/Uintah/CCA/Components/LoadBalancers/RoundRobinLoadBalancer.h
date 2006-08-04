#ifndef UINTAH_HOMEBREW_RoundRobinLoadBalancer_H
#define UINTAH_HOMEBREW_RoundRobinLoadBalancer_H

#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <set>

namespace Uintah {
   /**************************************
     
     CLASS
       RoundRobinLoadBalancer
      
       Short Description...
      
     GENERAL INFORMATION
      
       RoundRobinLoadBalancer.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       RoundRobinLoadBalancer
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
  class RoundRobinLoadBalancer : public LoadBalancerCommon {
  public:
    RoundRobinLoadBalancer(const ProcessorGroup* myworld);
    ~RoundRobinLoadBalancer();
    
    virtual int getPatchwiseProcessorAssignment(const Patch* patch);
    
  private:
    RoundRobinLoadBalancer(const RoundRobinLoadBalancer&);
    RoundRobinLoadBalancer& operator=(const RoundRobinLoadBalancer&);
    
   };
} // End namespace Uintah


#endif

