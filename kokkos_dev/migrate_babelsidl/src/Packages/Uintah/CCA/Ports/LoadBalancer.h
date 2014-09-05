
#ifndef UINTAH_HOMEBREW_LOADBALANCER_H
#define UINTAH_HOMEBREW_LOADBALANCER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/Region.h>
#include <string>

#include <Packages/Uintah/CCA/Ports/share.h>


namespace Uintah {

  class Patch;
  class ProcessorGroup;
  class DetailedTasks;
  class Scheduler;
  class VarLabel;

  typedef vector<SCIRun::IntVector> SizeList;
/****************************************

CLASS
   LoadBalancer
   
   Short description...

GENERAL INFORMATION

   LoadBalancer.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  //! The Load Balancer is responsible for assigning tasks to do their work
  //! on specified processors.  Different subclasses differ in the way this is done.
  class SCISHARE LoadBalancer : public UintahParallelPort {
  public:
    LoadBalancer();
    virtual ~LoadBalancer();
    
    //! Assigns each task in tg to its corresponding processor.
    //! Uses the patchwise processor assignment. 
    //! @see getPatchwiseProcessorAssignment.
    virtual void assignResources(DetailedTasks& tg) = 0;

    //! Gets the processor that this patch will be assigned to.  
    //! This is different with the different load balancers.
    virtual int getPatchwiseProcessorAssignment(const Patch* patch) = 0;

    //! Gets the processor that this patch was assigned to on the last timestep.
    //! This is the same as getPatchwiseProcessorAssignment for non-dynamic load balancers.
    //! See getPatchwiseProcessorAssignment.
    virtual int getOldProcessorAssignment(const VarLabel*,
					  const Patch* patch, const int)
      { return getPatchwiseProcessorAssignment(patch); }

    //! Determines if the Load Balancer requests a taskgraph recompile.
    //! Only possible for Dynamic Load Balancers.
    virtual bool needRecompile(double, double, const GridP&)
      { return false; }

    //! In AMR, we need to tell the Load balancer that we're regridding
    //    virtual void doRegridTimestep() {}

    //! Reads the problem spec file for the LoadBalancer section, and looks 
    //! for entries such as outputNthProc, dynamicAlgorithm, and interval.
    virtual void problemSetup(ProblemSpecP&, SimulationStateP& state) = 0;

    //! Creates the Load Balancer's Neighborhood.  
    //! This is a vector of patches that represent any patch that this load 
    //! balancer will potentially have to receive data from.
    virtual void createNeighborhood(const GridP& grid, const GridP& oldGrid) = 0;

    //! Asks the Load Balancer if it is dynamic.
    virtual bool isDynamic() { return false; }

    //! Asks if a patch in the patch subset is in the neighborhood.
    virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*) = 0;

    //! Asks the load balancer if patch is in the neighborhood.
    virtual bool inNeighborhood(const Patch*) = 0;

    //! Returns the patchset of all patches that have work done on this processor.
    virtual const PatchSet* getPerProcessorPatchSet(const LevelP& level) = 0;
    virtual const PatchSet* getPerProcessorPatchSet(const GridP& grid) = 0;
    virtual const PatchSet* getOutputPerProcessorPatchSet(const LevelP& level) = 0;

    //! For dynamic load balancers, Check if we need to rebalance the load, and do so if necessary.
    virtual bool possiblyDynamicallyReallocate(const GridP&, int state) = 0;

    //! For dymanic load balancers, Load balance patch set and possibly split patches to improve load balance.
    virtual void dynamicallyLoadBalanceAndSplit(const GridP&, SizeList, vector<vector<Region> >&, bool) = 0;

    //! Returns the value of n (every n procs it performs output tasks).
    virtual int getNthProc() { return 1; }

    //! Tells the load balancer on which procs data was output.
    virtual void restartInitialize(ProblemSpecP&, std::string file /*tsurl*/, const GridP&) {}
    
    // state variables
    enum {
      check = 0, init, regrid, restart
    };
  private:
    LoadBalancer(const LoadBalancer&);
    LoadBalancer& operator=(const LoadBalancer&);
  };
} // End namespace Uintah
    
#endif
