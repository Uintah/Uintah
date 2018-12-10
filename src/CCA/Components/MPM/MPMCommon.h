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

#ifndef UINTAH_HOMEBREW_MPM_COMMON_H
#define UINTAH_HOMEBREW_MPM_COMMON_H

#include <CCA/Components/Application/ApplicationCommon.h>

#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

namespace Uintah {

  class ProcessorGroup;

  class MPMFlags;
  class MPMLabel;
  
  class MPMCommon : public ApplicationCommon
  {
  public:
    MPMCommon(const ProcessorGroup* myworld, MaterialManagerP materialManager);

    virtual ~MPMCommon();

    virtual void materialProblemSetup(const ProblemSpecP& prob_spec,
                                      MPMFlags* flags, bool isRestart);

    virtual void cohesiveZoneProblemSetup(const ProblemSpecP& prob_spec,
                                          MPMFlags* flags);
                                          
    void scheduleUpdateStress_DamageErosionModels(SchedulerP        & sched,
                                                  const PatchSet    * patches,
                                                  const MaterialSet * matls );

    // Used by the switcher
    virtual void setupForSwitching() {
  
      d_cohesiveZoneState.clear();
      d_cohesiveZoneState_preReloc.clear();

      d_particleState.clear();
      d_particleState_preReloc.clear();
    }

  public:
    // Particle state
    std::vector<std::vector<const VarLabel* > > d_particleState;
    std::vector<std::vector<const VarLabel* > > d_particleState_preReloc;
    
    std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState;
    std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState_preReloc;
    
    inline void setParticleGhostLayer(Ghost::GhostType type, int ngc) {
      particle_ghost_type = type;
      particle_ghost_layer = ngc;
    }

    inline void getParticleGhostLayer(Ghost::GhostType& type, int& ngc) {
      type = particle_ghost_type;
      ngc = particle_ghost_layer;
    }

    MPMLabel* lb {nullptr};

  private:
    MPMFlags*             d_flags       = nullptr;
    
  protected:
    //! so all components can know how many particle ghost cells to ask for
    Ghost::GhostType particle_ghost_type{Ghost::None};
    int particle_ghost_layer{0};
    
    /*! update the stress field due to damage & erosion*/
    void updateStress_DamageErosionModels(const ProcessorGroup  *,
                                          const PatchSubset     * patches,
                                          const MaterialSubset  * ,
                                          DataWarehouse         * old_dw,
                                          DataWarehouse         * new_dw );
  };
}

#endif
