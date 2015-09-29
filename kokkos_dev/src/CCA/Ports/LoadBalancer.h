/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_HOMEBREW_LOADBALANCER_H
#define UINTAH_HOMEBREW_LOADBALANCER_H

#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Region.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/UintahParallelPort.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <set>
#include <string>

namespace Uintah {

  class Patch;
  class ProcessorGroup;
  class DetailedTasks;
  class Scheduler;
  class VarLabel;
  class DataArchive;
  class DetailedTask;

  typedef std::vector<IntVector> SizeList;

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
  
   
KEYWORDS
   Scheduler

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  //! The Load Balancer is responsible for assigning tasks to do their work
  //! on specified processors.  Different subclasses differ in the way this is done.
  class LoadBalancer : public UintahParallelPort {
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
    virtual int getOldProcessorAssignment( const Patch * patch ) = 0;

    //! Determines if the Load Balancer requests a taskgraph recompile.
    //! Only possible for Dynamic Load Balancers.
    virtual bool needRecompile( double, double, const GridP& ) { return false; }

    //! In AMR, we need to tell the Load balancer that we're regridding
    //    virtual void doRegridTimestep() {}

    //! Reads the problem spec file for the LoadBalancer section, and looks 
    //! for entries such as outputNthProc, dynamicAlgorithm, and interval.
    virtual void problemSetup(ProblemSpecP&, GridP& grid, SimulationStateP& state) = 0;

    //! Creates the Load Balancer's Neighborhood.  
    //! This is a vector of patches that represent any patch that this load 
    //! balancer will potentially have to receive data from.
    virtual void createNeighborhood(const GridP& grid, const GridP& oldGrid) = 0;

    //! Asks the Load Balancer if it is dynamic.
    virtual bool isDynamic() { return false; }

    //! returns all processors in this processors neighborhood
    virtual const std::set<int>& getNeighborhoodProcessors() = 0;

    //! Asks if a patch in the patch subset is in the neighborhood.
    virtual bool inNeighborhood(const PatchSubset*) = 0;

    //! Asks the load balancer if patch is in the neighborhood.
    virtual bool inNeighborhood(const Patch*) = 0;

    //! Returns the patchset of all patches that have work done on this processor.
    virtual const PatchSet* getPerProcessorPatchSet(const LevelP& level) = 0;
    virtual const PatchSet* getPerProcessorPatchSet(const GridP& grid) = 0;
    virtual const PatchSet* getOutputPerProcessorPatchSet(const LevelP& level) = 0;

    //! For dynamic load balancers, Check if we need to rebalance the load, and do so if necessary.
    virtual bool possiblyDynamicallyReallocate(const GridP&, int state) = 0;

    //! Returns the value of n (every n procs it performs output tasks).
    virtual int getNthProc() { return 1; }

    //! Returns the processor the patch will be output on (not patchwiseProcessor
    //! if outputNthProc is set)
    virtual int getOutputProc( const Patch * patch ) = 0;

    //! Tells the load balancer on which procs data was output.
    virtual void restartInitialize( DataArchive * archive, const int time_index, const std::string & ts_url, const GridP & grid ) = 0;
    
    // state variables
    enum {
      check = 0, init, regrid, restart
    };

    //////////////////////////////////////////////////
    // Cost profiling functions:

    // Update the contribution for this patch.
    virtual void addContribution( DetailedTask *task, double cost ) = 0;

    // Finalize the contributions (updates the weight, should be called once per timestep).
    virtual void finalizeContributions( const GridP & currentgrid ) = 0;

    // Initializes the weights in regions in the new grid that are not in the old level.
    virtual void initializeWeights( const Grid* oldgrid, const Grid* newgrid ) = 0;

    // Resets forecaster to the defaults.
    virtual void resetCostForecaster() = 0;
    
  private:
    LoadBalancer(const LoadBalancer&);
    LoadBalancer& operator=(const LoadBalancer&);
  };
} // End namespace Uintah
    
#endif
