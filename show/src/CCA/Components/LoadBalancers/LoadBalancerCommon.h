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

#ifndef UINTAH_HOMEBREW_LoadBalancerCommon_H
#define UINTAH_HOMEBREW_LoadBalancerCommon_H

#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/SFC.h>

#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/UintahParallelComponent.h>

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

    
    KEYWORDS
    LoadBalancerCommon

    DESCRIPTION
    Long description...

    WARNING

****************************************/

struct PatchInfo {
  PatchInfo(int i, int n) {id = i; numParticles = n;}
  PatchInfo() {}

  int id;
  int numParticles;
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

/// Load Balancer Common.  Implements many functions in common among 
/// the load balancer subclasses.  The main function that sets load balancers
/// apart is getPatchwiseProcessorAssignment - how it determines which patch
/// to assign on which procesor.
class LoadBalancerCommon : public LoadBalancer, public UintahParallelComponent {
public:
  LoadBalancerCommon(const ProcessorGroup* myworld);
  ~LoadBalancerCommon();

  virtual int getPatchwiseProcessorAssignment( const Patch * patch );

  //! The implementation in LoadBalancerCommon.cc is for dynamice load balancers.
  //! The Simple and SingleProcessor override this function with default implementations.
  virtual int getOldProcessorAssignment(       const Patch * patch );

  /// Goes through the Detailed tasks and assigns each to its own processor.
  virtual void assignResources( DetailedTasks & tg );

  /// Creates the Load Balancer's Neighborhood.  This is a vector of patches 
  /// that represent any patch that this load balancer will potentially have to 
  /// receive data from.
  virtual void createNeighborhood( const GridP& grid, const GridP& oldGrid );

  const std::set<int>& getNeighborhoodProcessors() { return d_neighborProcessors; }

  /// Asks the load balancer if a patch in the patch subset is in the neighborhood.
  virtual bool inNeighborhood(const PatchSubset*);

  /// Asks the load balancer if patch is in the neighborhood.
  virtual bool inNeighborhood(const Patch*);

  /// Reads the problem spec file for the LoadBalancer section, and looks 
  /// for entries such as outputNthProc, dynamicAlgorithm, and interval.
  virtual void problemSetup(ProblemSpecP& pspec, GridP& grid, SimulationStateP& state);

  // for DynamicLoadBalancer mostly, but if we're called then it also means the 
  // grid might have changed and need to create a new perProcessorPatchSet
  virtual bool possiblyDynamicallyReallocate(const GridP&, int state);

  // Cost profiling functions
  // Update the contribution for this patch.
  virtual void addContribution( DetailedTask * task, double cost );
  
  // Finalize the contributions (updates the weight, should be called once per timestep):
  virtual void finalizeContributions( const GridP & currentGrid );

  // Initializes the regions in the new level that are not in the old level.
  virtual void initializeWeights( const Grid* oldgrid, const Grid* newgrid );

  // Resets the profiler counters to zero
  virtual void resetCostForecaster();

  //! Returns n - data gets output every n procs.
  virtual int getNthProc() { return d_outputNthProc; }

  //! Returns the processor the patch will be output on (not patchwiseProcessor
  //! if outputNthProc is set)
  virtual int getOutputProc(const Patch* patch) { return (getPatchwiseProcessorAssignment(patch)/d_outputNthProc)*d_outputNthProc; }

  //! Returns the patchset of all patches that have work done on this processor.
  virtual const PatchSet* getPerProcessorPatchSet(const LevelP& level) { return d_levelPerProcPatchSets[level->getIndex()].get_rep(); }
  virtual const PatchSet* getPerProcessorPatchSet(const GridP& grid) { return d_gridPerProcPatchSet.get_rep(); }
  virtual const PatchSet* getOutputPerProcessorPatchSet(const LevelP& level) { return d_outputPatchSets[level->getIndex()].get_rep(); };

  //! Assigns the patches to the processors they ended up on in the previous
  //! Simulation.  Returns true if we need to re-load balance (if we have a 
  //! different number of procs than were saved to disk
  virtual void restartInitialize(       DataArchive  * archive,
                                  const int            time_index,
                                  const std::string  & tsurl,
                                  const GridP        & grid );
   
private:
  LoadBalancerCommon(const LoadBalancerCommon&);
  LoadBalancerCommon& operator=(const LoadBalancerCommon&);

protected:

  // Calls space-filling curve on level, and stores results in pre-allocated output
  void useSFC(const LevelP& level, int* output);
    
  double d_lastLbTime;

  double d_lbInterval;
  
  bool   d_checkAfterRestart;

  // The assignment vectors are stored 0-n.  This stores the start patch number so we can
  // detect if something has gone wrong when we go to look up what proc a patch is on.
  int    d_assignmentBasePatch;   
  int    d_oldAssignmentBasePatch;

  std::vector<int> d_processorAssignment; ///< stores which proc each patch is on
  std::vector<int> d_oldAssignment; ///< stores which proc each patch used to be on
  std::vector<int> d_tempAssignment; ///< temp storage for checking to reallocate

  SFC <double> d_sfc;
  bool         d_doSpaceCurve;

  /// Creates a patchset of all patches that have work done on each processor.
  //    - There are two versions of this function.  The first works on a per level
  //      basis.  The second works on the entire grid and will provide a PatchSet
  //      that contains all patches.
  //    - For example, if processor 1 works on patches 1,2 on level 0 and patch 3 on level 1,
  //      and processor 2 works on 4,5 on level 0, and 6 on level 1, then
  //      - Version 1 (for Level 1) will create {{3},{6}}
  //      - Version 2 (for all levels) will create {{1,2,3},{4,5,6}}
  virtual const PatchSet* createPerProcessorPatchSet( const LevelP & level );
  virtual const PatchSet* createPerProcessorPatchSet( const GridP  & grid );
  virtual const PatchSet* createOutputPatchSet(       const LevelP & level );

  SimulationStateP d_sharedState; ///< to keep track of timesteps
  Scheduler* d_scheduler; ///< store the scheduler to not have to keep passing it in
  std::set<const Patch*> d_neighbors; ///< the neighborhood.  See createNeighborhood
  std::set<int> d_neighborProcessors; //a list of processors that are in this processors neighborhood
  //! output on every nth processor.  This variable needs to be shared 
  //! with the DataArchiver as well, but we keep it here because the lb
  //! needs it to assign the processor resource.
  int d_outputNthProc;

  std::vector< Handle<const PatchSet> > d_levelPerProcPatchSets;
  Handle< const PatchSet >              d_gridPerProcPatchSet;
  std::vector< Handle<const PatchSet> > d_outputPatchSets;

};

} // End namespace Uintah

#endif
