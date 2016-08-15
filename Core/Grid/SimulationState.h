/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_HOMEBREW_SimulationState_H
#define UINTAH_HOMEBREW_SimulationState_H

#include <Core/Util/InfoMapper.h>
#include <Core/Util/RefCounted.h>
#include <Core/Util/DebugStream.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Ghost.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <sci_defs/visit_defs.h>

#include <map>
#include <vector>
#include <iostream>


#define OVERHEAD_WINDOW 40
namespace Uintah {


class VarLabel;
class Material; 
class ICEMaterial;
class MPMMaterial;
class CZMaterial;
class ArchesMaterial; 
class WasatchMaterial;
class SimpleMaterial;
class Level;

/**************************************
      
    CLASS
      SimulationState
      
      Short Description...
      
    GENERAL INFORMATION
      
      SimulationState.h
      
      Steven G. Parker
      Department of Computer Science
      University of Utah
      
      Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
    KEYWORDS
      SimulationState
      
    DESCRIPTION
      Global data structure to store data accessible
      by all components
      
    WARNING
      
****************************************/

class SimulationState : public RefCounted {
public:
  SimulationState(ProblemSpecP &ps);
  ~SimulationState();

  void clearMaterials();

  const VarLabel* getDeltLabel() const {
    return deltLabel;
  }
  const VarLabel* getRefineFlagLabel() const {
    return refineFlagLabel;
  }
  const VarLabel* getOldRefineFlagLabel() const {
    return oldRefineFlagLabel;
  }
  const VarLabel* getRefinePatchFlagLabel() const {
    return refinePatchFlagLabel;
  }
  const VarLabel* getSwitchLabel() const {
    return switchLabel;
  }
  const VarLabel* getOutputIntervalLabel() const {
    return outputIntervalLabel;
  }
  const VarLabel* getOutputTimestepIntervalLabel() const {
    return outputTimestepIntervalLabel;
  }
  const VarLabel* getCheckpointIntervalLabel() const {
    return checkpointIntervalLabel;
  }
  const VarLabel* getCheckpointTimestepIntervalLabel() const {
    return checkpointTimestepIntervalLabel;
  }

  void registerSimpleMaterial(SimpleMaterial*);
  void registerMPMMaterial(MPMMaterial*);
  void registerMPMMaterial(MPMMaterial*,unsigned int index);
  void registerCZMaterial(CZMaterial*);
  void registerCZMaterial(CZMaterial*,unsigned int index);
  void registerArchesMaterial(ArchesMaterial*);
  void registerICEMaterial(ICEMaterial*);
  void registerICEMaterial(ICEMaterial*,unsigned int index);
  void registerWasatchMaterial(WasatchMaterial*);
  void registerWasatchMaterial(WasatchMaterial*,unsigned int index);

  void registerMaterial( Material* );
  void registerMaterial( Material*, unsigned int index );

  int getNumMatls() const {
    return (int)matls.size();
  }
  int getNumMPMMatls() const {
    return (int)mpm_matls.size();
  }
  int getNumCZMatls() const {
    return (int)cz_matls.size();
  }
  int getNumArchesMatls() const {
    return (int)arches_matls.size();
  }
  int getNumICEMatls() const {
    return (int)ice_matls.size();
  }
  int getNumWasatchMatls() const {
    return (int)wasatch_matls.size();
  }  
  MaterialSubset* getAllInOneMatl() {
    return allInOneMatl;
  }
  Material* getMaterial(int idx) const {
    return matls[idx];
  }
  MPMMaterial* getMPMMaterial(int idx) const {
    return mpm_matls[idx];
  }
  CZMaterial* getCZMaterial(int idx) const {
    return cz_matls[idx];
  }
  ArchesMaterial* getArchesMaterial(int idx) const {
    return arches_matls[idx];
  }
  ICEMaterial* getICEMaterial(int idx) const {
    return ice_matls[idx];
  }
  WasatchMaterial* getWasatchMaterial(int idx) const {
    return wasatch_matls[idx];
  }
  SimpleMaterial* getSimpleMaterial(int idx) const {
    return simple_matls[idx];
  }

  void finalizeMaterials();
  const MaterialSet* allMPMMaterials() const;
  const MaterialSet* allCZMaterials() const;
  const MaterialSet* allArchesMaterials() const;
  const MaterialSet* allICEMaterials() const;
  const MaterialSet* allWasatchMaterials() const;
  const MaterialSet* allMaterials() const;
  const MaterialSet* originalAllMaterials() const;
  const MaterialSubset* refineFlagMaterials() const;

  void setOriginalMatlsFromRestart(MaterialSet* matls);
  
  Material* parseAndLookupMaterial(ProblemSpecP& params,
                                   const std::string& name) const;
  Material* getMaterialByName(const std::string& name) const;

  inline int getMaxMatlIndex() const { return max_matl_index; }

  // Ghost Layer
  inline void setParticleGhostLayer(Ghost::GhostType type, int ngc) {
    particleGhostType = type;
    particleGhostLayer = ngc;
  }

  inline void getParticleGhostLayer(Ghost::GhostType& type, int& ngc) {
    type = particleGhostType;
    ngc = particleGhostLayer;
  }

  // Elapsed time
  double getElapsedTime() const { return d_elapsedTime; }
  void   setElapsedTime(double t) { d_elapsedTime = t; }

  // Returns the integer timestep index of the top level of the
  // simulation.  All simulations start with a top level time step
  // number of 0.  This value is incremented by one for each
  // simulation time step processed.  The 'set' function should only
  // be called by the SimulationController at the beginning of a
  // simulation.  The 'increment' function is called by the
  // SimulationController at the end of each timestep.
  int  getCurrentTopLevelTimeStep() const { return d_topLevelTimestep; }
  void setCurrentTopLevelTimeStep( int ts ) { d_topLevelTimestep = ts; }
  void incrementCurrentTopLevelTimeStep() { ++d_topLevelTimestep; }

  bool getSwitchState() const { return d_switchState; }
  void setSwitchState(bool ans) { d_switchState = ans; }    

  double getPrevDelt() const { return d_prevDelt; }   
  void setPrevDelt(bool ans) { d_prevDelt = ans; }    

  double getCurrentDelt() const { return d_currentDelt; }   
  void setCurrentDelt(bool ans) { d_currentDelt = ans; }    

  SimulationTime* getSimTime() const { return d_simTime; }
  void setSimTime(SimulationTime* ans) { d_simTime = ans; }

  bool getCopyDataTimestep() const { return d_CopyDataTimestep; }
  void setCopyDataTimestep(bool ans) { d_CopyDataTimestep = ans; }
  
  bool getRegridTimestep() const { return d_RegridTimestep; }
  void setRegridTimestep(bool ans) { d_RegridTimestep = ans; }

  bool getAdjustDelT() const { return d_adjustDelT; }
  void setAdjustDelT(bool ans) { d_adjustDelT = ans; }
  
  bool getLockstepAMR() const { return d_lockstepAMR; }
  void setLockstepAMR(bool ans) {d_lockstepAMR = ans;}
  
  bool getUpdateOutputInterval() const { return d_updateOutputInterval; }
  void setUpdateOutputInterval(bool ans) { d_updateOutputInterval = ans; }
  
  bool getUpdateCheckpointInterval() const { return d_updateCheckpointInterval; }
  void setUpdateCheckpointInterval(bool ans) { d_updateCheckpointInterval = ans; }

  bool getRecompileTaskGraph() const { return d_recompileTaskGraph; }   
  void setRecompileTaskGraph(bool ans) { d_recompileTaskGraph = ans; }    

  bool getUsingLocalFileSystems() const { return d_usingLocalFileSystems; }   
  void setUsingLocalFileSystems(bool ans) { d_usingLocalFileSystems = ans; }    

  int getNumDims() const { return d_numDims; }
  int* getActiveDims() { return d_activeDims; }
  void setDimensionality(bool x, bool y, bool z);

  // Next sample to write to
  double getOverheadAvg() const { return d_overheadAvg; }
  void calculateOverhead(int d_n, double percent_overhead);  

  // Particle label info for MPM
  std::vector<std::vector<const VarLabel* > > d_particleState;
  std::vector<std::vector<const VarLabel* > > d_particleState_preReloc;

  std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState;
  std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState_preReloc;

  void resetStats();
  
  // timing statistics to test load balance
  enum RunTimeStat
  {
    CompilationTime = 0,       // Note: do not change the order 
    RegriddingTime,            // of these five enumberators.
    RegriddingCompilationTime, // They are use in
    RegriddingCopyDataTime,    // SimulationState::getOverheadTime
    LoadBalancerTime,          // to determine the overhead time.
    
    TaskExecTime,              // Note: do not change the order
    TaskLocalCommTime,         // of these five enumerators.
    TaskGlobalCommTime,        // They are used in
    TaskWaitCommTime,          // SimulationController::printSimulationStats
    TaskWaitThreadTime,        // and SimulationState::getTotalTime.

    OutputFileIOTime ,         // These two enumerators are not used in
    OutputFileIORate,	       // SimulationState::getTotalTime.


    SCIMemoryUsed,
    SCIMemoryMaxUsed,
    SCIMemoryHighwater,

    MemoryUsed,
    MemoryResident,
    
#ifdef USE_PAPI_COUNTERS
    TotalFlops,                // Total FLOPS
    TottalVFlops,              // Total FLOPS optimized to count 
                               // scaled double precision vector operations
    L2Misses,                  // Total L2 cache misses
    L3Misses,                  // Total L3 cache misses
#endif
    MAX_TIMING_STATS
  };

  ReductionInfoMapper< RunTimeStat, double > d_runTimeStats;

private:
  SimulationState( const SimulationState& );
  SimulationState& operator=( const SimulationState& );
      
  // Percent time in overhead samples
  double d_overhead[OVERHEAD_WINDOW];
  double d_overheadWeights[OVERHEAD_WINDOW];
  
  double d_overheadAvg;
  int    d_overheadIndex; // Next sample to write

  const VarLabel* deltLabel;
  const VarLabel* refineFlagLabel;
  const VarLabel* oldRefineFlagLabel;
  const VarLabel* refinePatchFlagLabel;
  const VarLabel* switchLabel;
  const VarLabel* outputIntervalLabel;
  const VarLabel* outputTimestepIntervalLabel;
  const VarLabel* checkpointIntervalLabel;
  const VarLabel* checkpointTimestepIntervalLabel;

  std::vector<Material*>        matls;
  std::vector<MPMMaterial*>     mpm_matls;
  std::vector<CZMaterial*>      cz_matls;
  std::vector<ArchesMaterial*>  arches_matls;
  std::vector<ICEMaterial*>     ice_matls;
  std::vector<WasatchMaterial*> wasatch_matls;
  std::vector<SimpleMaterial*>  simple_matls;

  //! in switcher we need to clear the materials, but don't 
  //! delete them yet or we might have VarLabel problems when 
  //! CMs are destroyed.  Store them here until the end
  std::vector<Material*> old_matls;

  std::map<std::string, Material*> named_matls;

  MaterialSet    * all_mpm_matls;
  MaterialSet    * all_cz_matls;
  MaterialSet    * all_ice_matls;
  MaterialSet    * all_wasatch_matls;  
  MaterialSet    * all_arches_matls;
  MaterialSet    * all_matls;

  // keep track of the original materials if you switch
  MaterialSet    * orig_all_matls;
  MaterialSubset * refine_flag_matls;
  MaterialSubset * allInOneMatl;

  //! for carry over vars in Switcher
  int max_matl_index;
  
  //! so all components can know how many particle ghost cells to ask for
  Ghost::GhostType particleGhostType;
  int particleGhostLayer;

  double d_elapsedTime;
  // The time step that the top level (w.r.t. AMR) is at during a
  // simulation.  Usually corresponds to the Data Warehouse generation
  // number (it does for non-restarted, non-amr simulations).  I'm going to
  // attempt to make sure that it does also for restarts.
  int    d_topLevelTimestep;
  
  bool   d_switchState;
  double d_prevDelt;
  double d_currentDelt;
  
  SimulationTime * d_simTime;

  // some places need to know if this is a copy data timestep or
  // a normal timestep.  (A copy data timestep is AMR's current 
  // method of getting data from an old to a new grid).
  bool d_CopyDataTimestep;
  
  bool d_RegridTimestep;

  // for AMR, how many times to execute a fine level per coarse level execution
  int d_timeRefinementRatio;
  
  bool d_adjustDelT;
  bool d_lockstepAMR;
  bool d_updateCheckpointInterval;
  bool d_updateOutputInterval;
  bool d_recompileTaskGraph;
  bool d_usingLocalFileSystems;  // Denotes whether each MPI node has a separate file system.

  // which dimensions are active.  Get the number of dimensions, and then
  // that many indices of activeDims are set to which dimensions are being used
  int d_numDims;           
  int d_activeDims[3];     

#ifdef HAVE_VISIT
public:
  // Reduction analysis variables for on the fly analysis
  struct analysisVar {
    std::string name;
    int matl;
    int level;
    std::vector< const VarLabel* > labels;
  };
  
  std::vector< analysisVar > d_analysisVars;

  // Interactive variables from the UPS problem spec or other state variables.
  struct interactiveVar {
    std::string name;
    TypeDescription::Type type;
    void *  value;
    bool    modifiable; // If true the variable maybe modified.
    bool    modified;   // If true the variable was modified by the user.
    bool    recompile;  // If true and the variable was modified recompile the task graph.
  };
  
  // Interactive variables from the UPS problem spec.
  std::vector< interactiveVar > d_UPSVars;

  // Interactive state variables from components.
  std::vector< interactiveVar > d_stateVars;

  void setVisIt( bool val ) { d_doVisIt = val; }
  bool getVisIt() { return d_doVisIt; }

  std::vector< DebugStream * > d_debugStreams;
  
private:
  bool d_doVisIt;
#endif      
}; // end class SimulationState

} // End namespace Uintah

#endif
