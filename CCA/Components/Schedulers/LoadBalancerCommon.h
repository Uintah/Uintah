#ifndef UINTAH_HOMEBREW_LoadBalancerCommon_H
#define UINTAH_HOMEBREW_LoadBalancerCommon_H

#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <set>
#include <string>

namespace Uintah {
   /**************************************
     
     CLASS
       LoadBalancerCommon
      
       Short Description...
      
     GENERAL INFORMATION
      
       LoadBalancerCommon.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       LoadBalancerCommon
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class LoadBalancerCommon : public LoadBalancer, public UintahParallelComponent {
   public:
     LoadBalancerCommon(const ProcessorGroup* myworld);
     ~LoadBalancerCommon();
     virtual void assignResources(DetailedTasks& tg);
     virtual int getPatchwiseProcessorAssignment(const Patch* patch) = 0;
     virtual int getOldProcessorAssignment(const VarLabel*,
                                           const Patch* patch, const int)
       { return getPatchwiseProcessorAssignment(patch); }

     virtual void createNeighborhood(const GridP& grid);
     virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*);
     virtual bool inNeighborhood(const Patch*);
     virtual void problemSetup(ProblemSpecP& pspec, SimulationStateP& state);
     virtual void setDynamicAlgorithm(std::string, double, int, float) {}
     virtual const PatchSet* createPerProcessorPatchSet(const LevelP& level);

     virtual void dynamicReallocation(const GridP&, const SchedulerP&) {}
     virtual int getNthProc() { return d_outputNthProc; }
   private:
     LoadBalancerCommon(const LoadBalancerCommon&);
     LoadBalancerCommon& operator=(const LoadBalancerCommon&);
   protected:
     SimulationStateP d_sharedState; // to keep track of timesteps
     
     std::set<const Patch*> d_neighbors;

     // output on every nth processor.  This variable needs to be shared 
     // with the DataArchiver as well, but we keep it here because the lb
     // needs it to assign the processor resource.
     int d_outputNthProc;

   };
} // End namespace Uintah

#endif
