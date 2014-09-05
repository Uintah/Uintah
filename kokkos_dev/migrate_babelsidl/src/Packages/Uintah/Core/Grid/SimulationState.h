#ifndef UINTAH_HOMEBREW_SimulationState_H
#define UINTAH_HOMEBREW_SimulationState_H

#include <Packages/Uintah/Core/Util/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <vector>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/Grid/share.h>
namespace Uintah {

using namespace SCIRun;

class VarLabel;
class Material; 
class ICEMaterial;
class MPMMaterial;
class ArchesMaterial; 
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
      
      Copyright (C) 2000 SCI Group
      
    KEYWORDS
      SimulationState
      
    DESCRIPTION
      Global data structure to store data accessible
      by all components
      
    WARNING
      
****************************************/

class SCISHARE SimulationState : public RefCounted {
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

  void registerSimpleMaterial(SimpleMaterial*);
  void registerMPMMaterial(MPMMaterial*);
  void registerMPMMaterial(MPMMaterial*,unsigned int index);
  void registerArchesMaterial(ArchesMaterial*);
  void registerICEMaterial(ICEMaterial*);
  void registerICEMaterial(ICEMaterial*,unsigned int index);
  int getNumVelFields() const;

  int getNumMatls() const {
    return (int)matls.size();
  }
  int getNumMPMMatls() const {
    return (int)mpm_matls.size();
  }
  int getNumArchesMatls() const {
    return (int)arches_matls.size();
  }
  int getNumICEMatls() const {
    return (int)ice_matls.size();
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
  ArchesMaterial* getArchesMaterial(int idx) const {
    return arches_matls[idx];
  }
  ICEMaterial* getICEMaterial(int idx) const {
    return ice_matls[idx];
  }
  Vector getGravity() const {
    return d_gravity;
  }
  void setNeedAddMaterial(int nAM) {
    d_needAddMaterial += nAM;
  }
  int needAddMaterial() const {
    return d_needAddMaterial;
  }
  void resetNeedAddMaterial() {
    d_needAddMaterial = 0;
  }

  void finalizeMaterials();
  const MaterialSet* allMPMMaterials() const;
  const MaterialSet* allArchesMaterials() const;
  const MaterialSet* allICEMaterials() const;
  const MaterialSet* allMaterials() const;
  const MaterialSubset* refineFlagMaterials() const;

  double getRefPress() const {
    return d_ref_press;
  }

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

  int getNumDims() { return d_numDims; }
  int* getActiveDims() { return d_activeDims; }
  void setDimensionality(bool x, bool y, bool z);

  vector<vector<const VarLabel* > > d_particleState;
  vector<vector<const VarLabel* > > d_particleState_preReloc;

  bool d_switchState;
  double d_prev_delt;

  SimulationTime* d_simTime;

  bool d_lockstepAMR;

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
  double outputTime;

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

  std::vector<Material*>       matls;
  std::vector<MPMMaterial*>    mpm_matls;
  std::vector<ArchesMaterial*> arches_matls;
  std::vector<ICEMaterial*>    ice_matls;
  std::vector<SimpleMaterial*> simple_matls;

  //! for carry over vars in Switcher
  int max_matl_index;

  //! in switcher we need to clear the materials, but don't 
  //! delete them yet or we might have VarLabel problems when 
  //! CMs are destroyed.  Store them here until the end
  std::vector<Material*> old_matls;

  std::map<std::string, Material*> named_matls;

  Vector d_gravity;

  MaterialSet    * all_mpm_matls;
  MaterialSet    * all_ice_matls;
  MaterialSet    * all_arches_matls;
  MaterialSet    * all_matls;
  MaterialSubset * refine_flag_matls;
  MaterialSubset * allInOneMatl;

  double d_ref_press;
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

}; // end class SimulationState

} // End namespace Uintah

#endif
