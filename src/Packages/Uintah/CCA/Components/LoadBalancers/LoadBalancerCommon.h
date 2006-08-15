#ifndef UINTAH_HOMEBREW_LoadBalancerCommon_H
#define UINTAH_HOMEBREW_LoadBalancerCommon_H

#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Level.h>

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
    
   /// Load Balancer Common.  Implements many functions in common among 
   /// the load balancer subclasses.  The main function that sets load balancers
   /// apart is getPatchwiseProcessorAssignment - how it determines which patch
   /// to assign on which procesor.
   class LoadBalancerCommon : public LoadBalancer, public UintahParallelComponent {
   public:
     LoadBalancerCommon(const ProcessorGroup* myworld);
     ~LoadBalancerCommon();

     /// Goes through the Detailed tasks and assigns each to its own processor.
     virtual void assignResources(DetailedTasks& tg);

     /// Creates the Load Balancer's Neighborhood.  This is a vector of patches 
     /// that represent any patch that this load balancer will potentially have to 
     /// receive data from.
     virtual void createNeighborhood(const GridP& grid, const GridP& oldGrid);

     /// Asks the load balancer if a patch in the patch subset is in the neighborhood.
     virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*);

     /// Asks the load balancer if patch is in the neighborhood.
     virtual bool inNeighborhood(const Patch*);

     /// Reads the problem spec file for the LoadBalancer section, and looks 
     /// for entries such as outputNthProc, dynamicAlgorithm, and interval.
     virtual void problemSetup(ProblemSpecP& pspec, SimulationStateP& state);

     // for DynamicLoadBalancer mostly, but if we're called then it also means the 
     // grid might have changed and need to create a new perProcessorPatchSet
     virtual bool possiblyDynamicallyReallocate(const GridP&, bool /*force*/);

     //! Returns n - data gets output every n procs.
     virtual int getNthProc() { return d_outputNthProc; }

    //! Returns the patchset of all patches that have work done on this processor.
    virtual const PatchSet* getPerProcessorPatchSet(const LevelP& level) { return levelPerProcPatchSets[level->getIndex()].get_rep(); }
    virtual const PatchSet* getPerProcessorPatchSet(const GridP& grid) { return gridPerProcPatchSet.get_rep(); }
    virtual const PatchSet* getOutputPerProcessorPatchSet(const LevelP& level) { return outputPatchSets[level->getIndex()].get_rep(); };

   private:
     LoadBalancerCommon(const LoadBalancerCommon&);
     LoadBalancerCommon& operator=(const LoadBalancerCommon&);
   protected:
     /// Creates a patchset of all patches that have work done on each processor.
     //    - There are two versions of this function.  The first works on a per level
     //      basis.  The second works on the entire grid and will provide a PatchSet
     //      that contains all patches.
     //    - For example, if processor 1 works on patches 1,2 on level 0 and patch 3 on level 1,
     //      and processor 2 works on 4,5 on level 0, and 6 on level 1, then
     //      - Version 1 (for Level 1) will create {{3},{6}}
     //      - Version 2 (for all levels) will create {{1,2,3},{4,5,6}}
     virtual const PatchSet* createPerProcessorPatchSet(const LevelP& level);
     virtual const PatchSet* createOutputPatchSet(const LevelP& level);
     virtual const PatchSet* createPerProcessorPatchSet(const GridP& grid);

     SimulationStateP d_sharedState; ///< to keep track of timesteps
     Scheduler* d_scheduler; ///< store the scheduler to not have to keep passing it in
     std::set<const Patch*> d_neighbors; ///< the neighborhood.  \See createNeighborhood
     //! output on every nth processor.  This variable needs to be shared 
     //! with the DataArchiver as well, but we keep it here because the lb
     //! needs it to assign the processor resource.
     int d_outputNthProc;

     vector<Handle<const PatchSet> > levelPerProcPatchSets;
     Handle<const PatchSet> gridPerProcPatchSet;
     vector<Handle<const PatchSet> > outputPatchSets;

   };

} // End namespace Uintah

#endif
