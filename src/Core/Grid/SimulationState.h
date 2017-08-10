/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <Core/Geometry/Vector.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MinMax.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/InfoMapper.h>
#include <Core/Util/RefCounted.h>
#include <Core/Util/Timers/Timers.hpp>

#include <sci_defs/visit_defs.h>
#include <sci_defs/papi_defs.h> // for PAPI performance counters

#include <map>
#include <vector>
#include <iostream>

namespace Uintah {

class VarLabel;
class Material; 
class ICEMaterial;
class MPMMaterial;
class CZMaterial;
class ArchesMaterial; 
class WasatchMaterial;
class FVMMaterial;
class SimpleMaterial;
class Level;

class DebugStream;
class Dout;
  
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

  const VarLabel* get_delt_label() const {
    return delt_label;
  }
  const VarLabel* get_refineFlag_label() const {
    return refineFlag_label;
  }
  const VarLabel* get_oldRefineFlag_label() const {
    return oldRefineFlag_label;
  }
  const VarLabel* get_refinePatchFlag_label() const {
    return refinePatchFlag_label;
  }
  const VarLabel* get_switch_label() const {
    return switch_label;
  }

  const VarLabel* get_outputInterval_label() const {
    return outputInterval_label;
  }
  const VarLabel* get_outputTimestepInterval_label() const {
    return outputTimestepInterval_label;
  }

  const VarLabel* get_checkpointInterval_label() const {
    return checkpointInterval_label;
  }
  const VarLabel* get_checkpointTimestepInterval_label() const {
    return checkpointTimestepInterval_label;
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
  void registerFVMMaterial(FVMMaterial*);
  void registerFVMMaterial(FVMMaterial*,unsigned int index);

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
  int getNumFVMMatls() const {
    return (int)fvm_matls.size();
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
  FVMMaterial* getFVMMaterial(int idx) const {
    return fvm_matls[idx];
  }

  inline void setParticleGhostLayer(Ghost::GhostType type, int ngc) {
    particle_ghost_type = type;
    particle_ghost_layer = ngc;
  }

  inline void getParticleGhostLayer(Ghost::GhostType& type, int& ngc) {
    type = particle_ghost_type;
    ngc = particle_ghost_layer;
  }

  void finalizeMaterials();
  const MaterialSet* allMPMMaterials() const;
  const MaterialSet* allCZMaterials() const;
  const MaterialSet* allArchesMaterials() const;
  const MaterialSet* allICEMaterials() const;
  const MaterialSet* allFVMMaterials() const;
  const MaterialSet* allWasatchMaterials() const;
  const MaterialSet* allMaterials() const;
  const MaterialSet* originalAllMaterials() const;
  const MaterialSubset* refineFlagMaterials() const;

  void setOriginalMatlsFromRestart(MaterialSet* matls);
  
  double getElapsedSimTime() const { return d_elapsed_sim_time; }
  void   setElapsedSimTime(double t) { d_elapsed_sim_time = t; }

  double getElapsedWallTime() const { return d_elapsed_wall_time; }
  void   setElapsedWallTime(double t) { d_elapsed_wall_time = t; }

  bool maybeLast() const { return d_maybeLast; }
  void maybeLast( bool val) { d_maybeLast = val; }

  // Returns the integer timestep index of the top level of the
  // simulation.  All simulations start with a top level time step
  // number of 0.  This value is incremented by one for each
  // simulation time step processed.  The 'set' function should only
  // be called by the SimulationController at the beginning of a
  // simulation.  The 'increment' function is called by the
  // SimulationController at the end of each timestep.
  int  getCurrentTopLevelTimeStep() const { return d_topLevelTimeStep; }
  void setCurrentTopLevelTimeStep( int ts ) { d_topLevelTimeStep = ts; }
  void incrementCurrentTopLevelTimeStep() { d_topLevelTimeStep++; }


  Material* parseAndLookupMaterial(ProblemSpecP& params,
                                   const std::string& name) const;
  Material* getMaterialByName(const std::string& name) const;

  inline int getMaxMatlIndex() const { return max_matl_index; }

  bool isCopyDataTimestep() const { return d_isCopyDataTimestep; }
  void setCopyDataTimestep(bool is_cdt) { d_isCopyDataTimestep = is_cdt; }
  
  bool isRegridTimestep() const { return d_isRegridTimestep; }
  void setRegridTimestep(bool ans) { d_isRegridTimestep = ans; }

  bool isLockstepAMR() const { return d_lockstepAMR; }
  void setIsLockstepAMR(bool ans) {d_lockstepAMR = ans;}
  
  bool updateOutputInterval() const { return d_updateOutputInterval; }
  void updateOutputInterval(bool ans) { d_updateOutputInterval = ans; }
  
  bool updateCheckpointInterval() const { return d_updateCheckpointInterval; }
  void updateCheckpointInterval(bool ans) { d_updateCheckpointInterval = ans; }

  bool getRecompileTaskGraph() const { return d_recompileTaskGraph; }   
  void setRecompileTaskGraph(bool ans) { d_recompileTaskGraph = ans; }    

  double getOverheadAvg() const { return d_overheadAvg; }   
  void   setOverheadAvg(double val) { d_overheadAvg = val; }

  bool getUseLocalFileSystems() const { return d_usingLocalFileSystems; }   
  void setUseLocalFileSystems(bool ans) { d_usingLocalFileSystems = ans; }    

  bool getSwitchState() const { return d_switchState; }   
  void setSwitchState(bool ans) { d_switchState = ans; }    
    
  void haveModifiedVars( bool val ) { d_haveModifiedVars = val; }
  bool haveModifiedVars() { return d_haveModifiedVars; }

  SimulationTime * getSimulationTime() const { return d_simulationTime; }   
  void             setSimulationTime(SimulationTime * simTime) { d_simulationTime = simTime; }

  int  getNumDims() const { return d_numDims; }
  int* getActiveDims() { return d_activeDims; }
  void setDimensionality(bool x, bool y, bool z);

  std::vector<std::vector<const VarLabel* > > d_particleState;
  std::vector<std::vector<const VarLabel* > > d_particleState_preReloc;

  std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState;
  std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState_preReloc;

  void resetStats();
  
  // timing statistics to test load balance
  enum RunTimeStat
  {
    // These five enumerators are used in SimulationController::printSimulationStats to determine the overhead time.
      CompilationTime = 0
    , RegriddingTime
    , RegriddingCompilationTime
    , RegriddingCopyDataTime
    , LoadBalancerTime
    
    // These five enumerators are used in SimulationController::printSimulationStats to determine task and comm overhead.
    , TaskExecTime
    , TaskLocalCommTime
    , TaskWaitCommTime
    , TaskReduceCommTime
    , TaskWaitThreadTime

    , XMLIOTime
    , OutputIOTime
    , ReductionIOTime
    , CheckpointIOTime
    , CheckpointReductionIOTime
    , TotalIOTime

    , OutputIORate
    , ReductionIORate
    , CheckpointIORate
    , CheckpointReducIORate

    , SCIMemoryUsed
    , SCIMemoryMaxUsed
    , SCIMemoryHighwater

    , MemoryUsed
    , MemoryResident
    
#ifdef USE_PAPI_COUNTERS
    , TotalFlops            // Floating point operations executed
    , TotalVFlops           // Floating point operations executed; optimized to count scaled DP vector ops
    , L2Misses              // L2 cache misses
    , L3Misses              // L3 cache misses
    , TLBMisses             // Total translation lookaside buffer misses
#endif

     , MAX_TIMING_STATS
  };

//#ifdef USE_PAPI_COUNTERS
//  ReductionInfoMapper< RunTimeStat, int > d_runTimeStats;
//#endif

  ReductionInfoMapper< RunTimeStat, double > d_runTimeStats;

  ReductionInfoMapper< unsigned int, double > d_otherStats;

private:

  void registerMaterial( Material* );
  void registerMaterial( Material*, unsigned int index );

  SimulationState( const SimulationState& );
  SimulationState& operator=( const SimulationState& );
      
  const VarLabel* delt_label;
  const VarLabel* refineFlag_label;
  const VarLabel* oldRefineFlag_label;
  const VarLabel* refinePatchFlag_label;
  const VarLabel* switch_label;
  const VarLabel* outputInterval_label;
  const VarLabel* outputTimestepInterval_label;
  const VarLabel* checkpointInterval_label;
  const VarLabel* checkpointTimestepInterval_label;

  std::vector<Material*>        matls;
  std::vector<MPMMaterial*>     mpm_matls;
  std::vector<CZMaterial*>      cz_matls;
  std::vector<ArchesMaterial*>  arches_matls;
  std::vector<ICEMaterial*>     ice_matls;
  std::vector<WasatchMaterial*> wasatch_matls;
  std::vector<SimpleMaterial*>  simple_matls;
  std::vector<FVMMaterial*>     fvm_matls;

  //! for carry over vars in Switcher
  int max_matl_index;

  //! so all components can know how many particle ghost cells to ask for
  Ghost::GhostType particle_ghost_type;
  int particle_ghost_layer;

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
  MaterialSet    * all_fvm_matls;
  MaterialSet    * all_matls;

  // keep track of the original materials if you switch
  MaterialSet    * orig_all_matls;
  MaterialSubset * refine_flag_matls;
  MaterialSubset * allInOneMatl;

  // The time step that the top level (w.r.t. AMR) is at during a
  // simulation.  Usually corresponds to the Data Warehouse generation
  // number (it does for non-restarted, non-amr simulations).  I'm going to
  // attempt to make sure that it does also for restarts.
  int    d_topLevelTimeStep;

  // The elapsed simulation time.
  double d_elapsed_sim_time;

  // The elapsed wall time.
  double d_elapsed_wall_time;

  // Flag for indicating the this timestep maybe the last. It is
  // determined by the Simulation Controller.
  bool d_maybeLast;
  
  // which dimensions are active.  Get the number of dimensions, and then
  // that many indices of activeDims are set to which dimensions are being used
  int d_numDims;           
  int d_activeDims[3];     
  
  // some places need to know if this is a copy data timestep or
  // a normal timestep.  (A copy data timestep is AMR's current 
  // method of getting data from an old to a new grid).
  bool d_isCopyDataTimestep;
  
  bool d_isRegridTimestep;

  // for AMR, how many times to execute a fine level per coarse level execution
  int d_timeRefinementRatio;

  bool d_adjustDelT;
  bool d_lockstepAMR;
  bool d_updateCheckpointInterval;
  bool d_updateOutputInterval;
  bool d_recompileTaskGraph;

  double d_overheadAvg; // Average time spent in overhead.

  // Tells the data archiver that we are running with each MPI node
  // having a separate file system.  (Simulation defaults to running
  // on a shared file system.)
  bool d_usingLocalFileSystems;

  bool d_switchState;
  bool d_haveModifiedVars;

  SimulationTime* d_simulationTime;
  
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
    void * value;
    bool   modifiable; // If true the variable maybe modified.
    bool   modified;   // If true the variable was modified by the user.
    bool   recompile;  // If true and the variable was modified
                       // recompile the task graph.
    double range[2];   // If modifiable min/max range of acceptable values.
  };
  
  // Interactive variables from the UPS problem spec.
  std::vector< interactiveVar > d_UPSVars;

  // Interactive state variables from components.
  std::vector< interactiveVar > d_stateVars;

  // Debug streams that can be turned on or off.
  std::vector< DebugStream * > d_debugStreams;
  std::vector< Dout * > d_douts;
  
  void setVisIt( int val ) { d_doVisIt = val; }
  int  getVisIt() { return d_doVisIt; }

private:
  unsigned int d_doVisIt;
#endif
}; // end class SimulationState

} // End namespace Uintah

#endif
