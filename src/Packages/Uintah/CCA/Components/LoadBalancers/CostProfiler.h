#ifndef UINTAH_HOMEBREW_CostProfiler_H
#define UINTAH_HOMEBREW_CostProfiler_H

#include <map>
#include <vector>
using namespace std;

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Region.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/ProfileDriver.h>
namespace Uintah {
   /**************************************
     
     CLASS
       CostProfiler 
      
       Profiles the execution costs of regions of the domain for use by the 
       DynamicLoadBalancer cost model
      
     GENERAL INFORMATION
      
       CostProfiler.h
      
       Justin Luitjens
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       CostProfiler
       DynamicLoadBalancer
      
     DESCRIPTION
       Profiles the execution costs of regions of the domain for use by the 
       DynamicLoadBalancer.  The domain is broken into square patches which
       are use as sample points.  At every timestep the scheduler needs to 
       update the execution costs using addContribution and then finalize the 
       contributions at the end of the timestep using finalizeContribution.

       The DLB uses the getWeights function to assign weights to regions which 
       can then be used to load balance the calculation.
      
     WARNING
      
     ****************************************/

  class CostProfiler {
  public:
    CostProfiler(const ProcessorGroup* myworld) : d_myworld(myworld), d_profiler(myworld), d_fineProfiler(myworld) {};
    void setMinPatchSize(const vector<IntVector> &min_patch_size);
    void setRefinementRatios(const vector<IntVector> &refinementRatios) {d_refinementRatios=refinementRatios;}
    //add the contribution for region r on level l
    void addContribution(DetailedTask *task, double cost);
    //finalize the contributions for this timestep
    void finalizeContributions(const GridP currentGrid);
    //outputs the error associated with the profiler
    void outputError(const GridP currentGrid);
    //get the contribution for region r on level l
    void getWeights(int l, const vector<Region> &regions, vector<double> &weights);
    //sets the decay rate for the exponential average
    void setTimestepWindow(int window) {d_profiler.setTimestepWindow(window);d_fineProfiler.setTimestepWindow(window);}
    //initializes the regions in the new level that are not in the old level
    void initializeWeights(const Grid* oldgrid, const Grid* newgrid);
    //resets all counters to zero
    void reset();
    //returns true if profiling data exists
    bool hasData() {return d_profiler.hasData();}
    //sets the maximum number of levels
    void setNumLevels(int num) {d_numLevels=num;}
  private:
    const ProcessorGroup* d_myworld;
    //the profiler for normal tasks
    ProfileDriver d_profiler;
    //the profiler for fine tasks
    ProfileDriver d_fineProfiler;
    
    vector<IntVector> d_refinementRatios;
    vector<IntVector> d_minPatchSize;
    int d_numLevels;

  };
} // End namespace Uintah


#endif

