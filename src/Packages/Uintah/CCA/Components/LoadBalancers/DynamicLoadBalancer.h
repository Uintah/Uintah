#ifndef UINTAH_HOMEBREW_DynamicLoadBalancer_H
#define UINTAH_HOMEBREW_DynamicLoadBalancer_H

#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SFC.h>
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
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       DynamicLoadBalancer
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
  struct ProcInfo {
    int rank;
    double cost;
  };
  class ProcCostCompare {
    public:
      inline bool operator()(const ProcInfo& p1, const ProcInfo& p2) const {
        return p1.cost>p2.cost;
    }
  };
  struct PatchInfo {
    PatchInfo(int i, int n, int a) {id = i; numParticles = n; assigned = a;}
    PatchInfo() {assigned = 0;}
    
    int id;
    int numParticles;
    int assigned;
    double cost;
  };

  class ParticleCompare {
  public:
    inline bool operator()(const PatchInfo& p1, const PatchInfo& p2) const {
      return p1.numParticles < p2.numParticles || 
        ( p1.numParticles == p2.numParticles && p1.id < p2.id);
    }
  };

  class PatchCompare {
  public:
    inline bool operator()(const PatchInfo& p1, const PatchInfo& p2) const {
      return p1.id < p2.id;
    }
  };
  class CostCompare {
    public:
    inline bool operator()(const PatchInfo& p1, const PatchInfo& p2) const {
      return p1.cost > p2.cost;
    }
  };

  class DynamicLoadBalancer : public LoadBalancerCommon {
  public:
    DynamicLoadBalancer(const ProcessorGroup* myworld);
    ~DynamicLoadBalancer();
    virtual int getPatchwiseProcessorAssignment(const Patch* patch);
    virtual int getOldProcessorAssignment(const VarLabel* var,
					  const Patch* patch, const int matl);
    virtual void problemSetup(ProblemSpecP& pspec, SimulationStateP& state);
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
    virtual void restartInitialize(ProblemSpecP& pspec, string, const GridP& grid);
    
  private:
    enum { static_lb, cyclic_lb, random_lb, patch_factor_lb };

    DynamicLoadBalancer(const DynamicLoadBalancer&);
    DynamicLoadBalancer& operator=(const DynamicLoadBalancer&);

    /// Helpers for possiblyDynamicallyRelocate.  These functions take care of setting 
    /// d_tempAssignment on all procs and dynamicReallocation takes care of maintaining 
    /// the state
    bool assignPatchesFactor(const GridP& grid, bool force);
    bool assignPatchesRandom(const GridP& grid, bool force);
    bool assignPatchesCyclic(const GridP& grid, bool force);

    /// Helper for assignPatchesFactor.  Collects each patch's particles
    void collectParticles(const GridP& grid, std::vector<PatchInfo>& allParticles);

    // calls space-filling curve on level, and stores results in pre-allocated output
    void useSFC(const LevelP& level, DistributedIndex* output);
    bool thresholdExceeded(const std::vector<PatchInfo>& patches);

    std::vector<int> d_processorAssignment; ///< stores which proc each patch is on
    std::vector<int> d_oldAssignment; ///< stores which proc each patch used to be on
    std::vector<int> d_tempAssignment; ///< temp storage for checking to reallocate

    // the assignment vectors are stored 0-n.  This stores the start patch number so we can
    // detect if something has gone wrong when we go to look up what proc a patch is on.
    int d_assignmentBasePatch;   
    int d_oldAssignmentBasePatch;

    double d_lbInterval;
    double d_lastLbTime;
    
    int d_lbTimestepInterval;
    int d_lastLbTimestep;
    
    bool d_do_AMR;
    ProblemSpecP d_pspec;
    
    double d_lbThreshold; //< gain threshold to exceed to require lb'ing
    double d_targetImb;
    float d_cellFactor;
    int d_dynamicAlgorithm;
    bool d_levelIndependent;
    bool d_timeRefineWeight;
    bool d_doSpaceCurve;
    bool d_collectParticles;
    bool d_checkAfterRestart;
  };
} // End namespace Uintah


#endif

