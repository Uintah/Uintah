/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_DynamicLoadBalancer_H
#define UINTAH_HOMEBREW_DynamicLoadBalancer_H

#include <CCA/Components/LoadBalancers/LoadBalancerCommon.h>

#include <CCA/Components/LoadBalancers/CostForecasterBase.h>
#include <CCA/Components/LoadBalancers/CostProfiler.h>
#include <CCA/Ports/SFC.h>
#include <Core/Grid/Grid.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <sci_defs/uintah_defs.h>
#if defined( HAVE_ZOLTAN )
#  include <zoltan_cpp.h>
#endif

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

#if defined( HAVE_ZOLTAN )
  class ZoltanFuncs {
  public:
    //Zoltan input functions
    static void zoltan_get_object_list( void *data, int sizeGID, int sizeLID,
                                        ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                                        int wgt_dim, float *obj_wgts, int *ierr );
    static int zoltan_get_number_of_objects( void *data, int *ierr );
    static int zoltan_get_number_of_geometry( void *data, int *ierr );
    static void zoltan_get_geometry_list( void *data, int sizeGID, int sizeLID,
                                          int num_obj,
                                          ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                                          int num_dim, double *geom_vec, int *ierr );
  };
#endif

  class DynamicLoadBalancer : public LoadBalancerCommon {
  public:
    DynamicLoadBalancer(const ProcessorGroup* myworld);
    ~DynamicLoadBalancer();
    virtual int getPatchwiseProcessorAssignment(const Patch* patch);
    virtual int getOldProcessorAssignment(const VarLabel* var,
					  const Patch* patch, const int matl);
    virtual void problemSetup(ProblemSpecP& pspec, GridP& grid, SimulationStateP& state);
    virtual bool needRecompile(double time, double delt, const GridP& grid); 

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

    //! Assigns the patches to the processors they ended up on in the previous
    //! Simulation.  Returns true if we need to re-load balance (if we have a 
    //! different number of procs than were saved to disk
    virtual void restartInitialize( DataArchive* archive, int time_index,
                                    ProblemSpecP& pspec,
                                    string tsurl, const GridP& grid );
   
  //cost profiling functions
    //update the contribution for this patch
    void addContribution(DetailedTask *task ,double cost) {d_costForecaster->addContribution(task,cost);}
    //finalize the contributions (updates the weight, should be called once per timestep)
    void finalizeContributions(const GridP currentGrid);
    //initializes the regions in the new level that are not in the old level
    void initializeWeights(const Grid* oldgrid, const Grid* newgrid) {
            d_costForecaster->initializeWeights(oldgrid,newgrid); }
    //resets the profiler counters to zero
    void resetCostForecaster() {d_costForecaster->reset();}
    
    // Helper for assignPatchesFactor.  Collects each patch's particles
    void collectParticles(const Grid* grid, vector<vector<int> >& num_particles);
    // same, but can be called after a regrid when patches have not been load balanced yet
    void collectParticlesForRegrid(const Grid* oldGrid, const vector<vector<Region> >& newGridRegions,  vector<vector<int> >& particles);


  private:
    
    struct double_int
    {
      double val;
      int loc;
      double_int(double val, int loc): val(val), loc(loc) {}
      double_int(): val(0), loc(-1) {}
    };

    vector<IntVector> d_minPatchSize;
    CostForecasterBase *d_costForecaster;
    enum { static_lb, cyclic_lb, random_lb, patch_factor_lb, zoltan_sfc_lb };

    DynamicLoadBalancer(const DynamicLoadBalancer&);
    DynamicLoadBalancer& operator=(const DynamicLoadBalancer&);

    /// Helpers for possiblyDynamicallyRelocate.  These functions take care of setting 
    /// d_tempAssignment on all procs and dynamicReallocation takes care of maintaining 
    /// the state
    bool assignPatchesFactor(const GridP& grid, bool force);
    bool assignPatchesRandom(const GridP& grid, bool force);
    bool assignPatchesCyclic(const GridP& grid, bool force);
    bool assignPatchesZoltanSFC(const GridP& grid, bool force);

    // calls space-filling curve on level, and stores results in pre-allocated output
    void useSFC(const LevelP& level, int* output);
    bool thresholdExceeded(const std::vector<vector<double> >& patch_costs);

    //Assign costs to a list of patches
    void getCosts(const Grid* grid, vector<vector<double> >&costs);

    std::vector<int> d_processorAssignment; ///< stores which proc each patch is on
    std::vector<int> d_oldAssignment; ///< stores which proc each patch used to be on
    std::vector<int> d_tempAssignment; ///< temp storage for checking to reallocate

    // the assignment vectors are stored 0-n.  This stores the start patch number so we can
    // detect if something has gone wrong when we go to look up what proc a patch is on.
    int d_assignmentBasePatch;   
    int d_oldAssignmentBasePatch;

    double d_lbInterval;
    double d_lastLbTime;

    bool d_levelIndependent;
    
    int d_lbTimestepInterval;
    int d_lastLbTimestep;
    
    bool d_do_AMR;
    ProblemSpecP d_pspec;
    
    double d_lbThreshold; //< gain threshold to exceed to require lb'ing
    
    double d_cellCost;      //cost weight per cell 
    double d_extraCellCost; //cost weight per extra cell
    double d_particleCost;  //cost weight per particle
    double d_patchCost;     //cost weight per patch
    
    int d_dynamicAlgorithm;
    bool d_doSpaceCurve;
    bool d_collectParticles;
    bool d_checkAfterRestart;

    SFC <double> sfc;

#if defined( HAVE_ZOLTAN )    
    // Zoltan global vars
    Zoltan * zz;
    string d_zoltanAlgorithm;  //This will be the algorithm that zoltan will use (HSFC, RCB, etc)
    string d_zoltanIMBTol;     //This will be the amount of imalance should be acceptable
#endif
  };
} // End namespace Uintah


#endif

