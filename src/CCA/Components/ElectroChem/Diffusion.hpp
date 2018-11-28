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

#ifndef CCA_COMPONENTS_ELECTROCHEM_DIFFUSION_HPP
#define CCA_COMPONENTS_ELECTROCHEM_DIFFUSION_HPP

#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {

  class Diffusion : public ApplicationCommon {
    virtual void initialize(const ProcessorGroup* pg,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DataWarehouse*  old_dw,
                                  DataWarehouse*  new_dw);

    virtual void computeStableTimeStep(const ProcessorGroup* pg,
                                       const PatchSubset*    patches,
                                       const MaterialSubset* matls,
                                             DataWarehouse*  old_dw,
                                             DataWarehouse*  new_dw);

    virtual void timeAdvance(const ProcessorGroup* pg,
                             const PatchSubset*    patches,
                             const MaterialSubset* matls,
                                   DataWarehouse*  old_dw,
                                   DataWarehouse*  new_dw);



    Diffusion(const Diffusion&);
    Diffusion& operator=(const Diffusion&);

    public:
      Diffusion(const ProcessorGroup* myworld,
                const MaterialManagerP materialManager);
    
      virtual ~Diffusion();

      virtual void problemSetup(const ProblemSpecP&     ps,
                                const ProblemSpecP&     restart_ps,
                                      GridP&            grid);

      virtual void scheduleInitialize(const LevelP&     level,
                                            SchedulerP& sched);

      virtual void scheduleRestartInitialize(const LevelP&     level,
                                                   SchedulerP& sched);

      virtual void scheduleComputeStableTimeStep(const LevelP&     level,
                                                       SchedulerP& sched);

      virtual void scheduleTimeAdvance(const LevelP&     level,
                                             SchedulerP& sched);
    protected:
  }; // End class Diffusion

} // End namespace Uintah

#endif // CCA_COMPONENTS_ELECTROCHEM_DIFFUSION_HPP
