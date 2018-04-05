/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <sci_defs/uintah_defs.h>

#include <map>
#include <vector>

namespace Uintah {

class VarLabel;
class Material; 
class SimpleMaterial;

class ArchesMaterial; 
class FVMMaterial;
class ICEMaterial;
class CZMaterial;
class MPMMaterial;
class WasatchMaterial;
  
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
      Global data structure to store data accessible by all components
      
    WARNING
      
****************************************/

class SimulationState : public RefCounted {

public:

  SimulationState();
  ~SimulationState();

  void clearMaterials();
  void finalizeMaterials();

  void registerSimpleMaterial(SimpleMaterial*);
  
  int getNumMatls() const {
    return (int)matls.size();
  }

  MaterialSubset* getAllInOneMatl() {
    return allInOneMatl;
  }

  Material* getMaterial(int idx) const {
    return matls[idx];
  }

  const MaterialSet* allMaterials() const;
  const MaterialSet* originalAllMaterials() const;

  void setOriginalMatlsFromRestart(MaterialSet* matls);
  
  Material* parseAndLookupMaterial(ProblemSpecP& params,
                                   const std::string& name) const;
  Material* getMaterialByName(const std::string& name) const;

  
#ifndef NO_ARCHES
  void registerArchesMaterial(ArchesMaterial*);
  int getNumArchesMatls() const {
    return (int)arches_matls.size();
  }
  ArchesMaterial* getArchesMaterial(int idx) const {
    return arches_matls[idx];
  }
  const MaterialSet* allArchesMaterials() const;
#endif

#ifndef NO_FVM
  void registerFVMMaterial(FVMMaterial*);
  void registerFVMMaterial(FVMMaterial*,unsigned int index);
  int getNumFVMMatls() const {
    return (int)fvm_matls.size();
  }
  FVMMaterial* getFVMMaterial(int idx) const {
    return fvm_matls[idx];
  }
  const MaterialSet* allFVMMaterials() const;
#endif

#ifndef NO_ICE
  void registerICEMaterial(ICEMaterial*);
  void registerICEMaterial(ICEMaterial*,unsigned int index);
  int getNumICEMatls() const {
    return (int)ice_matls.size();
  }
  ICEMaterial* getICEMaterial(int idx) const {
    return ice_matls[idx];
  }
  const MaterialSet* allICEMaterials() const;
#endif

#ifndef NO_MPM
  void registerCZMaterial(CZMaterial*);
  void registerCZMaterial(CZMaterial*,unsigned int index);  
  int getNumCZMatls() const {
    return (int)cz_matls.size();
  }
  CZMaterial* getCZMaterial(int idx) const {
    return cz_matls[idx];
  }
  const MaterialSet* allCZMaterials() const;

  void registerMPMMaterial(MPMMaterial*);
  void registerMPMMaterial(MPMMaterial*,unsigned int index);
  int getNumMPMMatls() const {
    return (int)mpm_matls.size();
  }
  MPMMaterial* getMPMMaterial(int idx) const {
    return mpm_matls[idx];
  }
  const MaterialSet* allMPMMaterials() const;
#endif

#ifndef NO_WASATCH
  void registerWasatchMaterial(WasatchMaterial*);
  void registerWasatchMaterial(WasatchMaterial*,unsigned int index);
  int getNumWasatchMatls() const {
    return (int)wasatch_matls.size();
  }
  WasatchMaterial* getWasatchMaterial(int idx) const {
    return wasatch_matls[idx];
  }
  const MaterialSet* allWasatchMaterials() const;
#endif

#ifndef NO_MPM
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
#endif

private:

  SimulationState( const SimulationState& );
  SimulationState& operator=( const SimulationState& );
      
  static int count;

  void registerMaterial( Material* );
  void registerMaterial( Material*, unsigned int index );

  std::map<std::string, Material*> named_matls;

  std::vector<Material*>        matls;
  std::vector<SimpleMaterial*>  simple_matls;
  MaterialSet * all_matls{nullptr};
  
  //! in switcher we need to clear the materials, but don't 
  //! delete them yet or we might have VarLabel problems when 
  //! CMs are destroyed.  Store them here until the end
  std::vector<Material*> old_matls;

  // keep track of the original materials if switching
  MaterialSet    * orig_all_matls{nullptr};
  MaterialSubset * allInOneMatl{nullptr};

#ifndef NO_ARCHES
  std::vector<ArchesMaterial*>  arches_matls;
  MaterialSet * all_arches_matls{nullptr};
#endif

#ifndef NO_FVM
  std::vector<FVMMaterial*>     fvm_matls;
  MaterialSet * all_fvm_matls{nullptr};
#endif

#ifndef NO_ICE
  std::vector<ICEMaterial*>     ice_matls;
  MaterialSet * all_ice_matls{nullptr};
#endif

#ifndef NO_MPM
  std::vector<CZMaterial*>      cz_matls;
  std::vector<MPMMaterial*>     mpm_matls;
  MaterialSet * all_cz_matls{nullptr};
  MaterialSet * all_mpm_matls{nullptr};
#endif

#ifndef NO_WASATCH
  std::vector<WasatchMaterial*> wasatch_matls;
  MaterialSet * all_wasatch_matls{nullptr};
#endif

#ifndef NO_MPM
  //! so all components can know how many particle ghost cells to ask for
  Ghost::GhostType particle_ghost_type{Ghost::None};
  int particle_ghost_layer{0};
#endif

  // Misc state that should be moved.

  // NOTE THESE VALUES ARE SET THE APPLICATION COMMON AND ARE HERE ONLY FOR
  // SHARED STATE COMMUNICATION.
public:
  // double getElapsedSimTime() const { return d_elapsed_sim_time; }
  // void   setElapsedSimTime(double t) { d_elapsed_sim_time = t; }

  // int  getCurrentTopLevelTimeStep() const { return d_topLevelTimeStep; }
  // void setCurrentTopLevelTimeStep( int ts ) { d_topLevelTimeStep = ts; }
private:
  // int    d_topLevelTimeStep{0};  // The time step.
  // double d_elapsed_sim_time{0};  // The elapsed simulation time.
  
}; // end class SimulationState

} // End namespace Uintah

#endif
