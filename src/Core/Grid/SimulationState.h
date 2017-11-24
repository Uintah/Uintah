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

#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/RefCounted.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <map>
#include <vector>

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
  SimulationState();
  ~SimulationState();

  void clearMaterials();

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

  void finalizeMaterials();
  const MaterialSet* allMPMMaterials() const;
  const MaterialSet* allCZMaterials() const;
  const MaterialSet* allArchesMaterials() const;
  const MaterialSet* allICEMaterials() const;
  const MaterialSet* allFVMMaterials() const;
  const MaterialSet* allWasatchMaterials() const;
  const MaterialSet* allMaterials() const;
  const MaterialSet* originalAllMaterials() const;

  void setOriginalMatlsFromRestart(MaterialSet* matls);
  
  Material* parseAndLookupMaterial(ProblemSpecP& params,
                                   const std::string& name) const;
  Material* getMaterialByName(const std::string& name) const;

  // Particle state
  inline void setParticleGhostLayer(Ghost::GhostType type, int ngc) {
    particle_ghost_type = type;
    particle_ghost_layer = ngc;
  }

  inline void getParticleGhostLayer(Ghost::GhostType& type, int& ngc) {
    type = particle_ghost_type;
    ngc = particle_ghost_layer;
  }

  std::vector<std::vector<const VarLabel* > > d_particleState;
  std::vector<std::vector<const VarLabel* > > d_particleState_preReloc;

  std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState;
  std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState_preReloc;

  // Misc state that should be moved.
  double getElapsedSimTime() const { return d_elapsed_sim_time; }
  void   setElapsedSimTime(double t) { d_elapsed_sim_time = t; }

  int  getCurrentTopLevelTimeStep() const { return d_topLevelTimeStep; }
  void setCurrentTopLevelTimeStep( int ts ) { d_topLevelTimeStep = ts; }

private:

  SimulationState( const SimulationState& );
  SimulationState& operator=( const SimulationState& );
      
  void registerMaterial( Material* );
  void registerMaterial( Material*, unsigned int index );

  std::vector<Material*>        matls;
  std::vector<MPMMaterial*>     mpm_matls;
  std::vector<CZMaterial*>      cz_matls;
  std::vector<ArchesMaterial*>  arches_matls;
  std::vector<ICEMaterial*>     ice_matls;
  std::vector<WasatchMaterial*> wasatch_matls;
  std::vector<SimpleMaterial*>  simple_matls;
  std::vector<FVMMaterial*>     fvm_matls;

  //! in switcher we need to clear the materials, but don't 
  //! delete them yet or we might have VarLabel problems when 
  //! CMs are destroyed.  Store them here until the end
  std::vector<Material*> old_matls;

  std::map<std::string, Material*> named_matls;

  MaterialSet * all_mpm_matls{nullptr};
  MaterialSet * all_cz_matls{nullptr};
  MaterialSet * all_ice_matls{nullptr};
  MaterialSet * all_wasatch_matls{nullptr};
  MaterialSet * all_arches_matls{nullptr};
  MaterialSet * all_fvm_matls{nullptr};
  MaterialSet * all_matls{nullptr};

  // keep track of the original materials if you switch
  MaterialSet    * orig_all_matls;
  MaterialSubset * allInOneMatl;

  //! so all components can know how many particle ghost cells to ask for
  Ghost::GhostType particle_ghost_type{Ghost::None};
  int particle_ghost_layer{0};

  // Misc state that should be moved.

  // The time step that the top level (w.r.t. AMR) is at during a
  // simulation.  Usually corresponds to the Data Warehouse generation
  // number (it does for non-restarted, non-amr simulations).  I'm going to
  // attempt to make sure that it does also for restarts.

  // NOTE THIS VALUE IS SET THE APPLICATION COMMON IT IS HERE ONLY FOR
  // SHARED STATE COMMUNICATION.
  int    d_topLevelTimeStep{0};

  // The elapsed simulation time.

  // NOTE THIS VALUE IS SET THE APPLICATION COMMON IT IS HERE ONLY FOR
  // SHARED STATE COMMUNICATION.
  double d_elapsed_sim_time{0};
  
}; // end class SimulationState

} // End namespace Uintah

#endif
