#ifndef UINTAH_HOMEBREW_SingleProcessorLoadBalancer_H
#define UINTAH_HOMEBREW_SingleProcessorLoadBalancer_H

#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

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
      virtual int getPatchwiseProcessorAssignment(const Patch* patch,
						  const ProcessorGroup* resources);
   private:
      SingleProcessorLoadBalancer(const SingleProcessorLoadBalancer&);
      SingleProcessorLoadBalancer& operator=(const SingleProcessorLoadBalancer&);
      
   };
} // End namespace Uintah


#endif

