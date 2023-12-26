/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_Examples_PortableDependencyTest_h
#define Packages_Uintah_CCA_Components_Examples_PortableDependencyTest_h

#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>

namespace Uintah {
  class SimpleMaterial;


/**************************************

CLASS
   PortableDependencyTest

GENERAL INFORMATION
   PortableDependencyTest.h

   John Holmen
   Scientific Computing and Imaging Institute
   University of Utah

KEYWORDS
   Kokkos

DESCRIPTION
   A modified version of Poisson1.ups used to test
   dependencies across tasks and parallel pattern
   back-ends.

WARNING
  
****************************************/

  class PortableDependencyTest : public ApplicationCommon {
  public:
    PortableDependencyTest( const ProcessorGroup   * myworld
                          , const MaterialManagerP   materialManager
                          );

    virtual ~PortableDependencyTest();

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
                                              ,       SchedulerP &
                                              );

    virtual void scheduleTimeAdvance( const LevelP     & level
                                    ,       SchedulerP &
                                    );

    virtual void scheduleTask1Computes( const LevelP     & level
                                      ,       SchedulerP &
                                      );

    virtual void scheduleTask2Modifies( const LevelP     & level
                                      ,       SchedulerP &
                                      );

    virtual void scheduleTask3Modifies( const LevelP     & level
                                      ,       SchedulerP &
                                      );

    virtual void scheduleTask4Requires( const LevelP     & level
                                      ,       SchedulerP &
                                      );

    template <typename ExecSpace, typename MemSpace>
    void task1Computes( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * old_dw
                      ,       OnDemandDataWarehouse                * new_dw
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      );

    template <typename ExecSpace, typename MemSpace>
    void task2Modifies( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * old_dw
                      ,       OnDemandDataWarehouse                * new_dw
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      );

    template <typename ExecSpace, typename MemSpace>
    void task3Modifies( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * old_dw
                      ,       OnDemandDataWarehouse                * new_dw
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      );

    template <typename ExecSpace, typename MemSpace>
    void task4Requires( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * old_dw
                      ,       OnDemandDataWarehouse                * new_dw
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      );

  private:
    void initialize( const ProcessorGroup *
                   , const PatchSubset    * patches
                   , const MaterialSubset * matls
                   ,       DataWarehouse  * old_dw
                   ,       DataWarehouse  * new_dw
                   );

    void computeStableTimeStep( const ProcessorGroup *
                              , const PatchSubset    * patches
                              , const MaterialSubset * matls
                              ,       DataWarehouse  * old_dw
                              ,       DataWarehouse  * new_dw
                              );

    double delt_;
    SimpleMaterial* mymat_;
    const VarLabel* phi_label;
    const VarLabel* residual_label;

    PortableDependencyTest(const PortableDependencyTest&);
    PortableDependencyTest& operator=(const PortableDependencyTest&);

  };
}

#endif
