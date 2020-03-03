/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef CCA_COMPONENTS_EXAMPLES_UNIFIEDSCHDEDULERTEST_H
#define CCA_COMPONENTS_EXAMPLES_UNIFIEDSCHDEDULERTEST_H

#include <CCA/Components/Application/ApplicationCommon.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>

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
     UnifiedSchedulerTestComponent, Unified Scheduler, GPU tasks

   DESCRIPTION
     A simple material, Poisson simulation used to test scheduling of GPU tasks
     via the UnifiedScheduler.

   ****************************************/

  class UnifiedSchedulerTest : public ApplicationCommon {

    public:

      UnifiedSchedulerTest( const ProcessorGroup   * myworld
                          , const MaterialManagerP   materialManager
                          );

      virtual ~UnifiedSchedulerTest();

      virtual void problemSetup( const ProblemSpecP & params
                               , const ProblemSpecP & restart_prob_spec
                               ,       GridP        & grid
                               );

      virtual void scheduleInitialize( const LevelP     & level
                                     ,       SchedulerP & sched
                                     );

      virtual void scheduleRestartInitialize( const LevelP     & level
                                            ,       SchedulerP & sched
                                            );

      virtual void scheduleComputeStableTimeStep( const LevelP     & level
                                                ,       SchedulerP & sched
                                                );

      virtual void scheduleTimeAdvance( const LevelP     & level
                                      ,       SchedulerP & sched
                                      );


    private:

            double           m_delt{0.0};
            SimpleMaterial * m_simple_material{nullptr};
      const VarLabel       * m_phi_label{nullptr};
      const VarLabel       * m_residual_label{nullptr};

      void initialize( const ProcessorGroup * pg
                     , const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * /*old_dw*/
                     ,       DataWarehouse  * new_dw
                     );

      void computeStableTimeStep( const ProcessorGroup * pg
                                , const PatchSubset    * patches
                                , const MaterialSubset * matls
                                ,       DataWarehouse  * old_dw
                                ,       DataWarehouse  * new_dw
                                );

      void timeAdvance( const PatchSubset           * patches
                      , const MaterialSubset        * matls
                      ,       OnDemandDataWarehouse * old_dw
                      ,       OnDemandDataWarehouse * new_dw
                      ,       UintahParams          & uintahParams
                      ,       ExecutionObject       & execObj
                      );

      void timeAdvance1DP( const ProcessorGroup * pg
                         , const PatchSubset    * patches
                         , const MaterialSubset * matls
                         ,       DataWarehouse  * old_dw
                         ,       DataWarehouse  * new_dw
                         );

      void timeAdvance3DP( const ProcessorGroup * pg
                         , const PatchSubset    * patches
                         , const MaterialSubset * matls
                         ,       DataWarehouse  * old_dw
                         ,       DataWarehouse  * new_dw
                         );

      // disable copy, assignment and move
      UnifiedSchedulerTest( const UnifiedSchedulerTest & )            = delete;
      UnifiedSchedulerTest& operator=( const UnifiedSchedulerTest & ) = delete;
      UnifiedSchedulerTest( UnifiedSchedulerTest && )                 = delete;
      UnifiedSchedulerTest& operator=( UnifiedSchedulerTest && )      = delete;

  };

void launchUnifiedSchedulerTestKernel( dim3               dimGrid
                                     , dim3               dimBlock
                                     , cudaStream_t     * stream
                                     , int                patchID
                                     , uint3              patchNodeLowIndex
                                     , uint3              patchNodeHighIndex
                                     , uint3              domainLow
                                     , uint3              domainHigh
                                     , GPUDataWarehouse * old_gpudw
                                     , GPUDataWarehouse * new_gpudw
                                     );

} // namespace Uintah

#endif // CCA_COMPONENTS_EXAMPLES_UNIFIEDSCHDEDULERTEST_H
