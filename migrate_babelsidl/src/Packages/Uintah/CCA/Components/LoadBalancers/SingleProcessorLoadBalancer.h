#ifndef UINTAH_HOMEBREW_SingleProcessorLoadBalancer_H
#define UINTAH_HOMEBREW_SingleProcessorLoadBalancer_H

#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>
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
    
  class SingleProcessorLoadBalancer : public LoadBalancerCommon {
  public:
    SingleProcessorLoadBalancer(const ProcessorGroup* myworld);
    ~SingleProcessorLoadBalancer();
    virtual void assignResources(DetailedTasks& tg);
    virtual int getPatchwiseProcessorAssignment(const Patch* patch);
    virtual void createNeighborhood(const GridP& grid);
    virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*);
    virtual bool inNeighborhood(const Patch*);
    
    virtual const PatchSet* createPerProcessorPatchSet(const LevelP& level);
  private:
    SingleProcessorLoadBalancer(const SingleProcessorLoadBalancer&);
    SingleProcessorLoadBalancer& operator=(const SingleProcessorLoadBalancer&);
      
   };
} // End namespace Uintah


#endif

