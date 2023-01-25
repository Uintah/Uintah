#ifndef UINTAH_HOMEBREW_DynamicLoadBalancer_H
#define UINTAH_HOMEBREW_DynamicLoadBalancer_H

/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/LoadBalancers/CostForecasterBase.h>
#include <CCA/Components/LoadBalancers/CostProfiler.h>
#include <CCA/Components/LoadBalancers/LoadBalancerCommon.h>

#include <sci_defs/uintah_defs.h>

#include <set>
#include <string>

namespace Uintah {
   /**************************************
     
     CLASS
       DynamicLoadBalancer
      
       Short Description...
      
     GENERAL INFORMATION
      
       DynamicLoadBalancer.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
             
     KEYWORDS
       DynamicLoadBalancer
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/

  class DynamicLoadBalancer : public LoadBalancerCommon {
  public:
    DynamicLoadBalancer(const ProcessorGroup* myworld);
    ~DynamicLoadBalancer();

    virtual void problemSetup(ProblemSpecP& pspec, GridP& grid, const MaterialManagerP& materialManager);
    virtual bool needRecompile(const GridP& grid); 

    /// call one of the assignPatches functions.
    /// Will initially need to load balance (on first timestep), and thus  
    /// be called by the SimulationController before the first compile.  
    /// Afterwards, if needRecompile function tells us we need to check for 
    /// load balancing, and then call
    /// this function (not called from simulation controller.)  We will then 
    /// go through the motions of load balancing, and if it determines we need
    /// to load balance (that the gain is greater than some threshold), it will
    /// set the patches to their new location,
    /// return true, signifying that we need to recompile.
    /// However, if force is true, it will re-loadbalance regardless of the
    /// threshold.
    virtual bool possiblyDynamicallyReallocate(const GridP& grid, int state);

    //! Asks the load balancer if it is dynamic.
    virtual bool isDynamic() { return true; }

    // Cost profiling functions
    // Update the contribution for this patch.
    virtual void addContribution( DetailedTask * task ,double cost ) { d_costForecaster->addContribution(task,cost); }

    // Finalize the contributions (updates the weight, should be called once per timestep):
    virtual void finalizeContributions( const GridP & currentGrid );

    // Initializes the regions in the new level that are not in the old level.
    virtual void initializeWeights(const Grid* oldgrid, const Grid* newgrid) { d_costForecaster->initializeWeights(oldgrid,newgrid); }

    // Resets the profiler counters to zero
    virtual void resetCostForecaster() { d_costForecaster->reset(); }
    
    // Helper for assignPatchesFactor.  Collects each patch's particles
    void collectParticles(const Grid* grid, std::vector<std::vector<int> >& num_particles);
    // Same, but can be called after a regrid when patches have not been load balanced yet.
    void collectParticlesForRegrid( const Grid                               * oldGrid,
                                    const std::vector< std::vector<Region> > & newGridRegions,
                                          std::vector< std::vector<int> >    & particles );


  private:
    
    struct double_int {
      double val;
      int loc;
      double_int(double val, int loc): val(val), loc(loc) {}
      double_int(): val(0), loc(-1) {}
    };

    std::vector<IntVector> d_minPatchSize;
    CostForecasterBase * d_costForecaster{nullptr};
    enum { static_lb, cyclic_lb, random_lb, patch_factor_lb };

    DynamicLoadBalancer(const DynamicLoadBalancer&);
    DynamicLoadBalancer& operator=(const DynamicLoadBalancer&);

    /// Helpers for possiblyDynamicallyRelocate.  These functions take care of setting 
    /// d_tempAssignment on all procs and dynamicReallocation takes care of maintaining 
    /// the state
    bool assignPatchesFactor(const GridP& grid, bool force);
    bool assignPatchesRandom(const GridP& grid, bool force);
    bool assignPatchesCyclic(const GridP& grid, bool force);

    bool thresholdExceeded(const std::vector<std::vector<double> >& patch_costs);

    //Assign costs to a list of patches
    void getCosts(const Grid* grid, std::vector<std::vector<double> >&costs);

    bool   d_levelIndependent;
    
    bool   d_do_AMR{false};
    ProblemSpecP d_pspec{nullptr};
    
    double d_lbThreshold; //< gain threshold to exceed to require lb'ing
    
    double d_cellCost;      //cost weight per cell 
    double d_extraCellCost; //cost weight per extra cell
    double d_particleCost;  //cost weight per particle
    double d_patchCost;     //cost weight per patch
    
    int  d_dynamicAlgorithm{patch_factor_lb};
    bool d_collectParticles{false};
  };
} // End namespace Uintah

#endif
