#ifndef UINTAH_HOMEBREW_SingleProcessorLoadBalancer_H
#define UINTAH_HOMEBREW_SingleProcessorLoadBalancer_H

#include <Uintah/Interface/LoadBalancer.h>
#include <Uintah/Parallel/UintahParallelComponent.h>

namespace Uintah {
   
   /**************************************
     
     CLASS
       SingleProcessorLoadBalancer
      
       Short Description...
      
     GENERAL INFORMATION
      
       SingleProcessorLoadBalancer.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       SingleProcessorLoadBalancer
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class SingleProcessorLoadBalancer : public LoadBalancer, public UintahParallelComponent {
   public:
      SingleProcessorLoadBalancer(const ProcessorGroup* myworld);
      ~SingleProcessorLoadBalancer();
      virtual void assignResources(TaskGraph& tg, const ProcessorGroup*);
   private:
      SingleProcessorLoadBalancer(const SingleProcessorLoadBalancer&);
      SingleProcessorLoadBalancer& operator=(const SingleProcessorLoadBalancer&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/17 07:04:55  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//

#endif

