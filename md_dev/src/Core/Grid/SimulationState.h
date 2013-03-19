/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/Util/RefCounted.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Ghost.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <sci_defs/cuda_defs.h>

#include <map>
#include <vector>
#include <iostream>


#define OVERHEAD_WINDOW 40
namespace Uintah {

using namespace SCIRun;

class VarLabel;
class Material; 
class ICEMaterial;
class MPMMaterial;
class CZMaterial;
class ArchesMaterial; 
class WasatchMaterial;
class SimpleMaterial;
class Level;

#ifdef HAVE_CUDA
  class UnifiedScheduler;
#endif
   
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
    return outputInv_label;
  }

  const VarLabel* get_checkpointInterval_label() const {
    return checkInv_label;
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
  
  void setNeedAddMaterial(int nAM) {
    d_needAddMaterial += nAM;
  }
  int needAddMaterial() const {
    return d_needAddMaterial;
  }

  inline void setParticleGhostLayer(Ghost::GhostType type, int ngc) {
    particle_ghost_type = type;
    particle_ghost_layer = ngc;
  }

  inline void getParticleGhostLayer(Ghost::GhostType& type, int& ngc) {
    type = particle_ghost_type;
    ngc = particle_ghost_layer;
  }

  void resetNeedAddMaterial() {
    d_needAddMaterial = 0;
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
  
  double getElapsedTime() const { return d_elapsed_time; }
  void   setElapsedTime(double t) { d_elapsed_time = t; }

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

  inline int getMaxMatlIndex() { return max_matl_index; }

  bool isCopyDataTimestep() { return d_isCopyDataTimestep; }
  void setCopyDataTimestep(bool is_cdt) { d_isCopyDataTimestep = is_cdt; }
  
  bool isRegridTimestep() { return d_isRegridTimestep; }
  void setRegridTimestep(bool ans) { d_isRegridTimestep = ans; }

  bool isLockstepAMR() { return d_lockstepAMR; }
  
  bool updateOutputInterval() { return d_updateOutputInv; }
  void updateOutputInterval(bool ans) { d_updateOutputInv = ans; }
  bool updateCheckpointInterval() { return d_updateCheckInv; }
  void updateCheckpointInterval(bool ans) { d_updateCheckInv = ans; }

  int getNumDims() { return d_numDims; }
  int* getActiveDims() { return d_activeDims; }
  void setDimensionality(bool x, bool y, bool z);

#ifdef HAVE_CUDA
  void setUnifiedScheduler(UnifiedScheduler* sched);
  inline UnifiedScheduler* getUnifiedScheduler() { return this->gpuDW; }
#endif

  vector<vector<const VarLabel* > > d_particleState;
  vector<vector<const VarLabel* > > d_particleState_preReloc;

  vector<vector<const VarLabel* > > d_cohesiveZoneState;
  vector<vector<const VarLabel* > > d_cohesiveZoneState_preReloc;

  bool d_switchState;
  double d_prev_delt;
  double d_current_delt;

  SimulationTime* d_simTime;

  bool d_lockstepAMR;
  bool d_updateCheckInv;
  bool d_updateOutputInv;

  // timing statistics to test load balance
  void clearStats();
  double compilationTime;
  double regriddingTime;
  double regriddingCompilationTime;
  double regriddingCopyDataTime;
  double loadbalancerTime;
  double taskExecTime;
  double taskLocalCommTime;
  double taskGlobalCommTime;
  double taskWaitCommTime;
  double outputTime;
  double taskWaitThreadTime;

  //percent time in overhead samples
  double overhead[OVERHEAD_WINDOW];
  double overheadWeights[OVERHEAD_WINDOW];
  //next sample to write to
  int overheadIndex;
  double overheadAvg;

private:

  void registerMaterial(Material*);
  void registerMaterial(Material*,unsigned int index);

  SimulationState(const SimulationState&);
  SimulationState& operator=(const SimulationState&);
      
  const VarLabel* delt_label;
  const VarLabel* refineFlag_label;
  const VarLabel* oldRefineFlag_label;
  const VarLabel* refinePatchFlag_label;
  const VarLabel* switch_label;
  const VarLabel* outputInv_label;
  const VarLabel* checkInv_label;

  std::vector<Material*>        matls;
  std::vector<MPMMaterial*>     mpm_matls;
  std::vector<CZMaterial*>      cz_matls;
  std::vector<ArchesMaterial*>  arches_matls;
  std::vector<ICEMaterial*>     ice_matls;
  std::vector<WasatchMaterial*> wasatch_matls;
  std::vector<SimpleMaterial*>  simple_matls;

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
  MaterialSet    * all_matls;

  // keep track of the original materials if you switch
  MaterialSet    * orig_all_matls;
  MaterialSubset * refine_flag_matls;
  MaterialSubset * allInOneMatl;

  int    d_needAddMaterial;

  // The time step that the top level (w.r.t. AMR) is at during a
  // simulation.  Usually corresponds to the Data Warehouse generation
  // number (it does for non-restarted, non-amr simulations).  I'm going to
  // attempt to make sure that it does also for restarts.
  int    d_topLevelTimeStep;
  double d_elapsed_time;

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

#ifdef HAVE_CUDA
  UnifiedScheduler* gpuDW;
#endif

}; // end class SimulationState

} // End namespace Uintah

#endif
