/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifdef HAVE_CUDA
#  include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif

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


  // When Task is scheduled to GPU
#if defined(HAVE_CUDA) && !defined(HAVE_KOKKOS)
  else if (event == CallBackEvent::GPU) {

    // Do time steps
    int num_patches = patches->size();
    for (int p = 0; p < num_patches; ++p) {
      const Patch* patch = patches->get(p);

      // Calculate the memory block size
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex();

      uint3 patchNodeLowIndex = make_uint3(l.x(), l.y(), l.z());
      uint3 patchNodeHighIndex = make_uint3(h.x(), h.y(), h.z());
      IntVector s = h - l;
      int xdim = s.x();
      int ydim = s.y();

      constexpr int BLOCKSIZE = 16;

      // define dimensions of the thread grid to be launched
      int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
      int yblocks = (int)ceil((float)ydim / BLOCKSIZE);
      dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
      dim3 dimGrid(xblocks, yblocks, 1);

      // now calculate the computation domain (ignoring the outside cell regions)
      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);

      h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::yplus)  == Patch::Neighbor ? 0 : 1,
                     patch->getBCType(Patch::zplus)  == Patch::Neighbor ? 0 : 1);

      // Domain extents used by the kernel to prevent out of bounds accesses.
      uint3 domainLow = make_uint3(l.x(), l.y(), l.z());
      uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());

      // setup and launch kernel
      GPUGridVariable<double> device_var;
      new_dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch))->get(device_var, "phi", patch->getID(), 0, 0);

      launchUnifiedSchedulerTestKernel( dimGrid
                                      , dimBlock
                                      , (cudaStream_t*)uintahParams.getStream(0)
                                      , patch->getID()
                                      , patchNodeLowIndex
                                      , patchNodeHighIndex
                                      , domainLow
                                      , domainHigh
                                      , (GPUDataWarehouse*)uintahParams.getOldTaskGpuDW()
                                      , (GPUDataWarehouse*)uintahParams.getNewTaskGpuDW()
                                      );

      // residual is automatically "put" with the D2H copy of the GPUReductionVariable
      // new_dw->put(sum_vartype(residual), m_residual_label);

    } // end patch loop
  } // end GPU task execution
#endif
}

} // namespace Uintah

#endif // CCA_COMPONENTS_EXAMPLES_UNIFIEDSCHDEDULERTEST_H
