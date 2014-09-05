#ifndef UINTAH_HOMEBREW_SimpleLoadBalancer_H
#define UINTAH_HOMEBREW_SimpleLoadBalancer_H

#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
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
    
   class SimpleLoadBalancer : public LoadBalancer, public UintahParallelComponent {
   public:
     SimpleLoadBalancer(const ProcessorGroup* myworld);
     ~SimpleLoadBalancer();
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
     std::set<const Patch*> d_neighbors;
     SimpleLoadBalancer(const SimpleLoadBalancer&);
     SimpleLoadBalancer& operator=(const SimpleLoadBalancer&);
      
   };
} // End namespace Uintah

#endif
