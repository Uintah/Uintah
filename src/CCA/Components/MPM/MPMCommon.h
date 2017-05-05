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

#ifndef UINTAH_HOMEBREW_MPM_COMMON_H
#define UINTAH_HOMEBREW_MPM_COMMON_H

#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

namespace Uintah {

  class ProcessorGroup;
  
  class MPMCommon {

  public:

    MPMCommon(const ProcessorGroup* myworld);
    virtual ~MPMCommon();

    virtual void materialProblemSetup(const ProblemSpecP& prob_spec,
                                      SimulationStateP& sharedState,
                                      MPMFlags* flags, bool isRestart);

    virtual void cohesiveZoneProblemSetup(const ProblemSpecP& prob_spec,
                                          SimulationStateP& sharedState,
                                          MPMFlags* flags);
                                          
    void scheduleUpdateStress_DamageErosionModels(SchedulerP        & sched,
                                                  const PatchSet    * patches,
                                                  const MaterialSet * matls );

   private:
    const ProcessorGroup* d_myworld     = nullptr;
    SimulationStateP      d_sharedState;
    MPMFlags*             d_flags       = nullptr;
    
   protected:
    /*! update the stress field due to damage & erosion*/
    void updateStress_DamageErosionModels(const ProcessorGroup  *,
                                          const PatchSubset     * patches,
                                          const MaterialSubset  * ,
                                          DataWarehouse         * old_dw,
                                          DataWarehouse         * new_dw );
  };
}

#endif
