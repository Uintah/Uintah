#ifndef UINTAH_HOMEBREW_SimulationState_H
#define UINTAH_HOMEBREW_SimulationState_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <map>
#include <vector>
#include <iostream>

namespace Uintah {

using namespace SCIRun;
using std::cerr;
using std::endl;

class VarLabel;
class Material; 
class ICEMaterial;
class MPMMaterial;
class ArchesMaterial; 
class SimpleMaterial;
   
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
    
class SimulationState : public RefCounted {
public:
  SimulationState(ProblemSpecP &ps);
  ~SimulationState();
  const VarLabel* get_delt_label() const {
    return delt_label;
  }
  const VarLabel* get_refineFlag_label() const {
    return refineFlag_label;
  }

  void registerSimpleMaterial(SimpleMaterial*);
  void registerMPMMaterial(MPMMaterial*);
  void registerArchesMaterial(ArchesMaterial*);
  void registerICEMaterial(ICEMaterial*);
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

  void finalizeMaterials();
  const MaterialSet* allMPMMaterials() const;
  const MaterialSet* allArchesMaterials() const;
  const MaterialSet* allICEMaterials() const;
  const MaterialSet* allMaterials() const;

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
private:

  void registerMaterial(Material*);

  SimulationState(const SimulationState&);
  SimulationState& operator=(const SimulationState&);
      
  const VarLabel* delt_label;
  const VarLabel* refineFlag_label;

  std::vector<Material*>       matls;
  std::vector<MPMMaterial*>    mpm_matls;
  std::vector<ArchesMaterial*> arches_matls;
  std::vector<ICEMaterial*>    ice_matls;
  std::vector<SimpleMaterial*> simple_matls;

  std::map<std::string, Material*> named_matls;

  Vector d_gravity;

  MaterialSet    * all_mpm_matls;
  MaterialSet    * all_ice_matls;
  MaterialSet    * all_arches_matls;
  MaterialSet    * all_matls;
  MaterialSubset * allInOneMatl;

  double d_ref_press;
  double d_elapsed_time;

  // The time step that the top level (w.r.t. AMR) is at during a
  // simulation.  Usually corresponds to the Data Warehouse generation
  // number (it does for non-restarted simulations).  I'm going to
  // attempt to make sure that it does also for restarts.
  int    d_topLevelTimeStep;

}; // end class SimulationState

} // End namespace Uintah

#endif
