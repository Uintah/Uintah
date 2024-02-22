/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <sci_defs/gpu_defs.h>

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

      template<typename ExecSpace, typename MemSpace>
      void timeAdvance( const PatchSubset           * patches
                      , const MaterialSubset        * matls
                      ,       OnDemandDataWarehouse * old_dw
                      ,       OnDemandDataWarehouse * new_dw
                      ,       UintahParams          & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace>& execObj
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

//______________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
inline void UnifiedSchedulerTest::
timeAdvance( const PatchSubset           * patches
           , const MaterialSubset        * matls
           ,       OnDemandDataWarehouse * old_dw
           ,       OnDemandDataWarehouse * new_dw
           ,       UintahParams          & uintahParams
           ,       ExecutionObject<ExecSpace, MemSpace>& execObj
           )
{

  CallBackEvent event = uintahParams.getCallBackEvent();

  //-----------------------------------------------------------------------------------------------
  // When Task is scheduled to CPU
  if (event == CallBackEvent::CPU) {

    int matl = 0;

    int num_patches = patches->size();
    for (int p = 0; p < num_patches; ++p) {
      const Patch* patch = patches->get(p);
      constNCVariable<double> phi;

      old_dw->get(phi, m_phi_label, matl, patch, Ghost::AroundNodes, 1);
      NCVariable<double> newphi;

      new_dw->allocateAndPut(newphi, m_phi_label, matl, patch);
      newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
      double residual = 0.0;
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex();

      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);

      h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::yplus)  == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::zplus)  == Patch::Neighbor ? 0 : 1);

      //__________________________________
      //  Stencil
      for (NodeIterator iter(l, h); !iter.done(); iter++) {
        IntVector n = *iter;

        newphi[n] = (1. / 6)
                  * (phi[n + IntVector(1, 0, 0)] + phi[n + IntVector(-1, 0, 0)] + phi[n + IntVector(0, 1, 0)]
                  + phi[n + IntVector(0, -1, 0)] + phi[n + IntVector(0, 0, 1)] + phi[n + IntVector(0, 0, -1)]);
        double diff = newphi[n] - phi[n];
        residual += diff * diff;
      }
      new_dw->put(sum_vartype(residual), m_residual_label);
    }
  }  // end CPU task execution
  //-----------------------------------------------------------------------------------------------
}

} // namespace Uintah

#endif // CCA_COMPONENTS_EXAMPLES_UNIFIEDSCHDEDULERTEST_H
