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

#ifndef UINTAH_CCA_COMPONENTS_MPMFVM_ESMPM_H
#define UINTAH_CCA_COMPONENTS_MPMFVM_ESMPM_H

#include <CCA/Components/FVM/FVMLabel.h>
#include <CCA/Components/FVM/ElectrostaticSolve.h>
#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <vector>

namespace Uintah {
  class ESMPM : public UintahParallelComponent, public SimulationInterface {
    public:
      ESMPM(const ProcessorGroup* myworld);
      ~ESMPM();

      virtual void problemSetup(const ProblemSpecP& prob_spec,
                                const ProblemSpecP& restart_prob_spec,
                                GridP& grid, SimulationStateP& state);

      virtual void outputProblemSpec(ProblemSpecP& prob_spec);

      virtual void scheduleInitialize(const LevelP& level, SchedulerP& sched);

      virtual void scheduleRestartInitialize(const LevelP& level, SchedulerP& sched);

      virtual void restartInitialize();

      virtual void scheduleComputeStableTimestep(const LevelP& level, SchedulerP& sched);

      virtual void scheduleTimeAdvance( const LevelP& level, SchedulerP& sched);

      virtual void scheduleFinalizeTimestep(const LevelP& level, SchedulerP& sched);


    protected:

      virtual void scheduleInterpolateParticlesToCellFC(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* mpm_matls,
                                                 const MaterialSet* all_matls);

      virtual void interpolateParticlesToCellFC(const ProcessorGroup* pg,
                                                const PatchSubset* patches,
                                                const MaterialSubset* matls,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw);

    protected:
      virtual void fcLinearInterpolator(const Patch* patch, const Point& pos,
                                        std::vector<IntVector>& ni,
                                        std::vector<double>& S);


    private:
      double d_TINY_RHO;

      SimulationStateP d_shared_state;
      Output* d_data_archiver;
      AMRMPM* d_amrmpm;
      ElectrostaticSolve* d_esfvm;
      MPMLabel* d_mpm_lb;
      FVMLabel* d_fvm_lb;
      MPMFlags* d_mpm_flags;

      MaterialSet* d_one_matlset;
      MaterialSubset* d_one_matl;
      SwitchingCriteria* d_switch_criteria;

  };
}
#endif
