#ifndef UINTAH_HOMEBREW_RoundRobinLoadBalancer_H
#define UINTAH_HOMEBREW_RoundRobinLoadBalancer_H

#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
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
    
   class RoundRobinLoadBalancer : public LoadBalancer, public UintahParallelComponent {
   public:
     RoundRobinLoadBalancer(const ProcessorGroup* myworld);
     ~RoundRobinLoadBalancer();
     virtual void assignResources(DetailedTasks& tg, const ProcessorGroup*);
     virtual int getPatchwiseProcessorAssignment(const Patch* patch,
						  const ProcessorGroup* resources);
     virtual void createNeighborhood(const GridP& grid, const ProcessorGroup*,
				    const Scheduler*);
     virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*);
     virtual bool inNeighborhood(const Patch*);

     virtual const PatchSet* createPerProcessorPatchSet(const LevelP& level,
							const ProcessorGroup* resources);
   private:
     RoundRobinLoadBalancer(const RoundRobinLoadBalancer&);
     RoundRobinLoadBalancer& operator=(const RoundRobinLoadBalancer&);
      
     std::set<const Patch*> d_neighbors;
   };
} // End namespace Uintah


#endif

