#ifndef UINTAH_HOMEBREW_SimpleLoadBalancer_H
#define UINTAH_HOMEBREW_SimpleLoadBalancer_H

#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <set>

namespace Uintah {
   /**************************************
     
     CLASS
       SimpleLoadBalancer
      
       Short Description...
      
     GENERAL INFORMATION
      
       SimpleLoadBalancer.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       SimpleLoadBalancer
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
  class SimpleLoadBalancer : public LoadBalancerCommon {
  public:
    SimpleLoadBalancer(const ProcessorGroup* myworld);
    ~SimpleLoadBalancer();
    
    virtual int getPatchwiseProcessorAssignment(const Patch* patch);
    
  private:
    SimpleLoadBalancer(const SimpleLoadBalancer&);
    SimpleLoadBalancer& operator=(const SimpleLoadBalancer&);
    
   };
} // End namespace Uintah


#endif

