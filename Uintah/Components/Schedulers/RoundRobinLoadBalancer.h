#ifndef UINTAH_HOMEBREW_RoundRobinLoadBalancer_H
#define UINTAH_HOMEBREW_RoundRobinLoadBalancer_H

#include <Uintah/Interface/LoadBalancer.h>
#include <Uintah/Parallel/UintahParallelComponent.h>

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
    
   class RoundRobinLoadBalancer : public LoadBalancer, public UintahParallelComponent {
   public:
      RoundRobinLoadBalancer(const ProcessorGroup* myworld);
      ~RoundRobinLoadBalancer();
      virtual void assignResources(TaskGraph& tg, const ProcessorGroup*);
   private:
      RoundRobinLoadBalancer(const RoundRobinLoadBalancer&);
      RoundRobinLoadBalancer& operator=(const RoundRobinLoadBalancer&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/17 07:04:54  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//

#endif

