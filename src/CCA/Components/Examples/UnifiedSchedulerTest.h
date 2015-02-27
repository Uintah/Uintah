/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
#ifndef Packages_Uintah_CCA_Components_Examples_UnifiedSchedulerTest_h
#define Packages_Uintah_CCA_Components_Examples_UnifiedSchedulerTest_h

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <sci_defs/cuda_defs.h>

namespace Uintah {

  class SimpleMaterial;

  /**************************************

   CLASS
   UnifiedSchedulerTest
   

   GENERAL INFORMATION

   UnifiedSchedulerTest.h

   Alan Humphrey
   Scientific Computing and Imaging Institute
   University of Utah

   
   KEYWORDS
   UnifiedSchedulerTestComponent, Unified Scheduler

   DESCRIPTION
   A simple material Poisson simulation used to test scheduling of GPU tasks
   via the UnifiedScheduler.

   Warning
   None

   ****************************************/

  class UnifiedSchedulerTest : public UintahParallelComponent, public SimulationInterface {

    public:

      UnifiedSchedulerTest(const ProcessorGroup* myworld);

      virtual ~UnifiedSchedulerTest();

      virtual void problemSetup(const ProblemSpecP& params,
                                const ProblemSpecP& restart_prob_spec,
                                GridP& grid,
                                SimulationStateP& simState);

      virtual void scheduleInitialize(const LevelP& level,
                                      SchedulerP& sched);

      virtual void scheduleComputeStableTimestep(const LevelP& level,
                                                 SchedulerP& sched);

      virtual void scheduleTimeAdvance(const LevelP& level,
                                       SchedulerP& sched);

    private:
      SimulationStateP sharedState_;
      double delt_;
      SimpleMaterial* simpleMaterial_;
      const VarLabel* phi_label;
      const VarLabel* residual_label;

      void initialize(const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* new_dw);

      void computeStableTimestep(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);

      void timeAdvanceUnified(Task::CallBackEvent event,
                              const ProcessorGroup* pg,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw,
                              void* stream,
                              int deviceID);

      void timeAdvance1DP(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

      void timeAdvance3DP(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

      // do not allow copy and assignment
      UnifiedSchedulerTest(const UnifiedSchedulerTest& gst);
      UnifiedSchedulerTest& operator=(const UnifiedSchedulerTest& gst);

  };

void launchUnifiedSchedulerTestKernel(dim3 dimGrid,
                                      dim3 dimBlock,
                                      cudaStream_t* stream,
                                      int patchID,
                                      uint3 patchNodeLowIndex,
                                      uint3 patchNodeHighIndex,
                                      uint3 domainLow,
                                      uint3 domainHigh,
                                      GPUDataWarehouse* old_gpudw,
                                      GPUDataWarehouse* new_gpudw);

} //end namespace Uintah

#endif
