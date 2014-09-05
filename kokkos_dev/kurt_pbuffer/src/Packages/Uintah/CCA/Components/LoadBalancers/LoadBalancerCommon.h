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
     virtual void assignResources(DetailedTasks3& tg);

     /// Creates the Load Balancer's Neighborhood.  This is a vector of patches 
     /// that represent any patch that this load balancer will potentially have to 
     /// receive data from.
     virtual void createNeighborhood(const GridP& grid);

     /// Asks the load balancer if a patch in the patch subset is in the neighborhood.
     virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*);

     /// Asks the load balancer if patch is in the neighborhood.
     virtual bool inNeighborhood(const Patch*);

     /// Reads the problem spec file for the LoadBalancer section, and looks 
     /// for entries such as outputNthProc, dynamicAlgorithm, and interval.
     virtual void problemSetup(ProblemSpecP& pspec, SimulationStateP& state);

     /// determines the dynamic algorithm based on values received in problemSetup.
     virtual void setDynamicAlgorithm(std::string, double, int, float, bool, double /*threshold*/) {}

     /// creates a patchset of all patches that have work done on this processor.
     virtual const PatchSet* createPerProcessorPatchSet(const LevelP& level);

     //! Returns n - data gets output every n procs.
     virtual int getNthProc() { return d_outputNthProc; }
   private:
     LoadBalancerCommon(const LoadBalancerCommon&);
     LoadBalancerCommon& operator=(const LoadBalancerCommon&);
   protected:
     SimulationStateP d_sharedState; ///< to keep track of timesteps
     Scheduler* d_scheduler; ///< store the scheduler to not have to keep passing it in
     std::set<const Patch*> d_neighbors; ///< the neighborhood.  \See createNeighborhood
     //! output on every nth processor.  This variable needs to be shared 
     //! with the DataArchiver as well, but we keep it here because the lb
     //! needs it to assign the processor resource.
     int d_outputNthProc;
   };
} // End namespace Uintah

#endif
