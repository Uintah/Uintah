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

#ifndef CCA_COMPONENTS_LOADBALANCERS_LOADBALANCERCOMMON_H
#define CCA_COMPONENTS_LOADBALANCERS_LOADBALANCERCOMMON_H

#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/SFC.h>

#include <CCA/Components/Schedulers/RuntimeStatsEnum.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/InfoMapper.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace Uintah {

class ApplicationInterface;

/**************************************

CLASS
    LoadBalancerCommon

GENERAL INFORMATION

    LoadBalancerCommon.h

    Steven G. Parker
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    
KEYWORDS
    LoadBalancerCommon

DESCRIPTION

****************************************/

struct PatchInfo {

  PatchInfo(int id, int num_particles)
    : m_id{id}
    , m_num_particles{num_particles}
  {}

  PatchInfo() = default;

  int m_id{0};
  int m_num_particles{0};
};


class ParticleCompare {

public:

  inline bool operator()(const PatchInfo& p1, const PatchInfo& p2) const {
    return p1.m_num_particles < p2.m_num_particles ||
      ( p1.m_num_particles == p2.m_num_particles && p1.m_id < p2.m_id);
  }
};


class PatchCompare {

public:

  inline bool operator()(const PatchInfo& p1, const PatchInfo& p2) const {
    return p1.m_id < p2.m_id;
  }
};

/// Load Balancer Common.  Implements many functions in common among 
/// the load balancer subclasses.  The main function that sets load balancers
/// apart is getPatchwiseProcessorAssignment - how it determines which patch
/// to assign on which procesor.
class LoadBalancerCommon : public LoadBalancer, public UintahParallelComponent {

public:

  LoadBalancerCommon( const ProcessorGroup * myworld );

  virtual ~LoadBalancerCommon();

  // Methods for managing the components attached via the ports.
  virtual void setComponents( UintahParallelComponent *comp ) {};
  virtual void getComponents();
  virtual void releaseComponents();

  //! Returns the MPI rank of the process on which the patch is to be executed.
  virtual int getPatchwiseProcessorAssignment( const Patch * patch );

  //! The implementation in LoadBalancerCommon.cc is for dynamic load balancers.
  //! The Simple and SingleProcessor override this function with default implementations.
  virtual int getOldProcessorAssignment( const Patch * patch );

  virtual bool needRecompile( const GridP& ) = 0;

  /// Goes through the Detailed tasks and assigns each to its own processor.
  virtual void assignResources( DetailedTasks & tg );

  /// Creates the Load Balancer's Neighborhood.  This is a vector of patches 
  /// that represent any patch that this load balancer will potentially have to 
  /// receive data from.
  virtual void createNeighborhoods( const GridP& grid, const GridP& oldGrid,  const bool hasDistalReqs = false);

  virtual const std::unordered_set<int>& getNeighborhoodProcessors() { return m_local_neighbor_processes; }

  virtual const std::unordered_set<int>& getDistalNeighborhoodProcessors() { return m_distal_neighbor_processes; }

  /// Asks the load balancer if a patch in the patch subset is in the neighborhood.
  virtual bool inNeighborhood( const PatchSubset * pss, const bool hasDistalReqs = false );

  /// Asks the load balancer if patch is in the neighborhood.
  virtual bool inNeighborhood( const Patch * patch, const bool hasDistalReqs = false );

  /// Reads the problem spec file for the LoadBalancer section, and looks 
  /// for entries such as outputNthProc, dynamicAlgorithm, and interval.
  virtual void problemSetup( ProblemSpecP & pspec,
			     GridP & grid,
			     const MaterialManagerP & materialManager );

  // for DynamicLoadBalancer mostly, but if we're called then it also means the 
  // grid might have changed and need to create a new perProcessorPatchSet
  virtual bool possiblyDynamicallyReallocate( const GridP &, int state );

  // Cost profiling functions
  // Update the contribution for this patch.
  virtual void addContribution( DetailedTask * task, double cost );
  
  // Finalize the contributions (updates the weight, should be called once per timestep):
  virtual void finalizeContributions( const GridP & currentGrid );

  // Initializes the regions in the new level that are not in the old level.
  virtual void initializeWeights( const Grid * oldgrid, const Grid * newgrid );

  // Resets the profiler counters to zero
  virtual void resetCostForecaster();

  //! Returns n - data gets output every n procs.
  virtual int  getNthRank() { return m_output_Nth_proc; }
  virtual void setNthRank( int nth ) { m_output_Nth_proc = nth; }

  //! Returns the MPI rank of the process that the patch will be output on (it will different from
  //! the patchwise processor assignment if outputNthProc is greater than 1).
  virtual int getOutputRank( const Patch * patch ) {
    return ( getPatchwiseProcessorAssignment( patch ) / m_output_Nth_proc ) * m_output_Nth_proc;
  }

  //! Returns the patchset of all patches that have work done on this processor.
  virtual const PatchSet* getPerProcessorPatchSet( const LevelP & level ) {
    return m_level_perproc_patchsets[level->getIndex()].get_rep();
  }

  virtual const PatchSet* getPerProcessorPatchSet( const GridP & grid ) {
    return m_grid_perproc_patchsets.get_rep();
  }
  
  virtual const PatchSet* getOutputPerProcessorPatchSet( const LevelP & level ) {
    return m_output_patchsets[level->getIndex()].get_rep();
  };

  //! Assigns the patches to the processors they ended up on in the previous
  //! Simulation.  Returns true if we need to re-load balance (if we have a 
  //! different number of procs than were saved to disk
  virtual void restartInitialize(       DataArchive  * archive
                                , const int            time_index
                                , const std::string  & tsurl
                                , const GridP        & grid
                                );
   
  int  getNumDims() const { return m_numDims; };
  int* getActiveDims() { return m_activeDims; };
  void setDimensionality(bool x, bool y, bool z);

  void setRuntimeStats( ReductionInfoMapper< RuntimeStatsEnum, double > *runtimeStats) { d_runtimeStats = runtimeStats; };     

protected:

  ApplicationInterface* m_application{nullptr};
  
  // Calls space-filling curve on level, and stores results in pre-allocated output
  void useSFC( const LevelP & level, int * output) ;
    
  /// Creates a patchset of all patches that have work done on each processor.
  //    - There are two versions of this function.  The first works on a per level
  //      basis.  The second works on the entire grid and will provide a PatchSet
  //      that contains all patches.
  //    - For example, if processor 1 works on patches 1,2 on level 0 and patch 3 on level 1,
  //      and processor 2 works on 4,5 on level 0, and 6 on level 1, then
  //      - Version 1 (for Level 1) will create {{3},{6}}
  //      - Version 2 (for all levels) will create {{1,2,3},{4,5,6}}
  virtual const PatchSet* createPerProcessorPatchSet( const LevelP & level );
  virtual const PatchSet* createPerProcessorPatchSet( const GridP  & grid  );
  virtual const PatchSet* createOutputPatchSet(       const LevelP & level );

  int    m_lb_timeStep_interval{0};
  int    m_last_lb_timeStep{0};

  double m_lb_interval{0.0};
  double m_last_lb_simTime{0.0};
  bool   m_check_after_restart{false};

  // The assignment vectors are stored 0-n.  This stores the start patch number so we can
  // detect if something has gone wrong when we go to look up what proc a patch is on.
  int m_assignment_base_patch{-1};
  int m_old_assignment_base_patch{-1};

  std::vector<int> m_processor_assignment; ///< stores which proc each patch is on
  std::vector<int> m_old_assignment;       ///< stores which proc each patch used to be on
  std::vector<int> m_temp_assignment;      ///< temp storage for checking to reallocate

  SFC <double> m_sfc;
  bool         m_do_space_curve{false};

  MaterialManagerP                    m_materialManager;      ///< to keep track of timesteps
  Scheduler                         * m_scheduler {nullptr};  ///< store the scheduler to not have to keep passing it in
  
  std::unordered_set<const Patch*>    m_local_neighbor_patches;     ///< the "local" neighborhood of patches.  See createNeighborhood
  std::unordered_set<int>             m_local_neighbor_processes;   ///< a list of "local" processes that are in this processors neighborhood
  std::unordered_set<const Patch*>    m_distal_neighbor_patches;    ///< the "distal" neighborhood of patches.  See createNeighborhood
  std::unordered_set<int>             m_distal_neighbor_processes;  ///< a list of "distal" processes that are in this processors neighborhood

  //! output on every nth processor.  This variable needs to be shared 
  //! with the DataArchiver as well, but we keep it here because the LB
  //! needs it to assign the processor resource.
  int m_output_Nth_proc{1};

  std::vector< Handle<const PatchSet> > m_level_perproc_patchsets;
  Handle< const PatchSet >              m_grid_perproc_patchsets;
  std::vector< Handle<const PatchSet> > m_output_patchsets;

  // Which dimensions are active.  Get the number of dimensions, and
  // then that many indices of activeDims are set to which dimensions
  // are being used.
  int m_numDims{0};
  int m_activeDims[3];

  ReductionInfoMapper< RuntimeStatsEnum, double > * d_runtimeStats{nullptr};

  DebugStream stats;
  DebugStream times;
  DebugStream lbout;
  
private:

  // eliminate copy, assignment and move
  LoadBalancerCommon( const LoadBalancerCommon & )            = delete;
  LoadBalancerCommon& operator=( const LoadBalancerCommon & ) = delete;
  LoadBalancerCommon( LoadBalancerCommon && )                 = delete;
  LoadBalancerCommon& operator=( LoadBalancerCommon && )      = delete;

  ///< Convenience method for patch/proc neighborhood creation
  void addPatchesAndProcsToNeighborhood( const Level                            * const level
                                       , const IntVector                        & low
                                       , const IntVector                        & high
                                       ,       std::unordered_set<const Patch*> & neighbors
                                       ,       std::unordered_set<int>          & processors
                                       );
};

} // namespace Uintah

#endif // CCA_COMPONENTS_LOADBALANCERS_LOADBALANCERCOMMON_H
