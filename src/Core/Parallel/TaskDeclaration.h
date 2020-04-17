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

// The purpose of this file is to provide application developers a clean way to portably declare tasks in all desired
// execution spaces.

// This mechanism allow the user to do two things.
// 1) Specify all execution spaces allowed by this task
// 2) Generate task objects and task object options.
//    At compile time, the compiler will compile the task for all specified execution spaces.
//    At run time, the appropriate if statement logic will determine which task to use.
// The ExecutionSpace and MemorySpace are brought in through the templates (specifically the ExecutionObject)

// taskFunctor is a functor which performs all additional task specific options the user desires
// taskName is the string name of the task.
// pmf1, pmf2, etc., are the function pointer to each potential portable function
// sched, patches, and materials are used for task creation.
// Args... args  are additional variatic Task arguments (note, these are set as by value and not by reference & or move &&
//  as some existing implementations want to use these arguments after they are passed in here, and this function attempts
//  perfect forwarding which will possibly move and make the application developer's arguments empty.

// Logic note, we don't currently allow both a Uintah CPU task and a Kokkos CPU task to exist in the same
// compiled build (thought it wouldn't be hard to implement it).  But we do allow a Kokkos CPU and
// Kokkos GPU task to exist in the same build

#ifndef UINTAH_CORE_PARALLEL_TASK_DECLARATION_H
#define UINTAH_CORE_PARALLEL_TASK_DECLARATION_H

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Core/Grid/Task.h>

namespace Uintah {

//--------------------------------------------------------------------------------------------------
// Three tag version of create_portable_arches_tasks()
template < typename TaskFunctor
         , typename TaskObject
         , typename ExecSpace1, typename MemSpace1
         , typename ExecSpace2, typename MemSpace2
         , typename ExecSpace3, typename MemSpace3
         , typename... Args >
void create_portable_tasks(       TaskFunctor   taskFunctor
                          ,       TaskObject  * ptr
                          , const std::string & taskNamePassed
                          , void (TaskObject::*pmf1)( const PatchSubset                            * patches
                                                    , const MaterialSubset                         * matls
                                                    ,       OnDemandDataWarehouse                  * old_dw
                                                    ,       OnDemandDataWarehouse                  * new_dw
                                                    ,       UintahParams                           & uintahParams
                                                    ,       ExecutionObject<ExecSpace1, MemSpace1> & execObj
                                                    ,       Args...                                  args
                                                    )
                          , void (TaskObject::*pmf2)( const PatchSubset                            * patches
                                                    , const MaterialSubset                         * matls
                                                    ,       OnDemandDataWarehouse                  * old_dw
                                                    ,       OnDemandDataWarehouse                  * new_dw
                                                    ,       UintahParams                           & uintahParams
                                                    ,       ExecutionObject<ExecSpace2, MemSpace2> & execObj
                                                    ,       Args...                                  args
                                                    )
                          , void (TaskObject::*pmf3)( const PatchSubset                            * patches
                                                    , const MaterialSubset                         * matls
                                                    ,       OnDemandDataWarehouse                  * old_dw
                                                    ,       OnDemandDataWarehouse                  * new_dw
                                                    ,       UintahParams                           & uintahParams
                                                    ,       ExecutionObject<ExecSpace3, MemSpace3> & execObj
                                                    ,       Args...                                  args
                                                    )
                          ,       SchedulerP  & sched
                          , const PatchSet    * patches
                          , const MaterialSet * matls
                          , const int           tg_num = -1
                          ,       Args...       args
                          )
{
  Task* task{nullptr};
  std::string taskName = taskNamePassed;
  // Check for GPU tasks
  // GPU tasks take top priority
  if ( Uintah::Parallel::usingDevice() ) {
    if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value || std::is_same<Kokkos::Cuda, ExecSpace2>::value || std::is_same<Kokkos::Cuda, ExecSpace3>::value ) {
      taskName = taskName + " (GPUTask)";
      if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value ) {           /* Task supports Kokkos::Cuda builds */
        task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      }
      else if ( std::is_same<Kokkos::Cuda, ExecSpace2>::value ) {      /* Task supports Kokkos::Cuda builds */
        task = scinew Task( taskName, ptr, pmf2, std::forward<Args>(args)... );
      }
      else if ( std::is_same<Kokkos::Cuda, ExecSpace3>::value ) {      /* Task supports Kokkos::Cuda builds */
        task = scinew Task( taskName, ptr, pmf3, std::forward<Args>(args)... );
      }

      //TODO: Consolodate these
      task->usesDevice(true);
      task->usesKokkosCuda(true);
      task->usesSimVarPreloading(true);

      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_CUDA, TaskAssignedMemorySpace::KOKKOS_CUDASPACE );
    }
  }

  // Check for CPU tasks if a GPU task did not get loaded
  if ( !task ) {
    if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value || std::is_same<Kokkos::OpenMP, ExecSpace2>::value || std::is_same<Kokkos::OpenMP, ExecSpace3>::value ) {
      if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value ) {         /* Task supports Kokkos::OpenMP builds */
        task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      }
      else if ( std::is_same<Kokkos::OpenMP, ExecSpace2>::value ) {    /* Task supports Kokkos::OpenMP builds */
        task = scinew Task( taskName, ptr, pmf2, std::forward<Args>(args)... );
      }
      else if ( std::is_same<Kokkos::OpenMP, ExecSpace3>::value ) {    /* Task supports Kokkos::OpenMP builds */
        task = scinew Task( taskName, ptr, pmf3, std::forward<Args>(args)... );
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_OPENMP, TaskAssignedMemorySpace::KOKKOS_HOSTSPACE );
    }
    else if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value || std::is_same<UintahSpaces::CPU, ExecSpace2>::value || std::is_same<UintahSpaces::CPU, ExecSpace3>::value ) {
      if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value ) {      /* Task supports non-Kokkos builds */
        task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      }
      else if ( std::is_same<UintahSpaces::CPU, ExecSpace2>::value ) { /* Task supports non-Kokkos builds */
        task = scinew Task( taskName, ptr, pmf2, std::forward<Args>(args)... );
      }
      else if ( std::is_same<UintahSpaces::CPU, ExecSpace3>::value ) { /* Task supports non-Kokkos builds */
        task = scinew Task( taskName, ptr, pmf3, std::forward<Args>(args)... );
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::UINTAH_CPU, TaskAssignedMemorySpace::UINTAH_HOSTSPACE );
    }
  }

  if ( task ) {
    taskFunctor(task);
  }

  if ( task ) {
    sched->addTask( task, patches, matls, tg_num );
  }
}

//--------------------------------------------------------------------------------------------------
// Two tag version of create_portable_arches_tasks()
template < typename TaskFunctor
         , typename TaskObject
         , typename ExecSpace1, typename MemSpace1
         , typename ExecSpace2, typename MemSpace2
         , typename... Args >
void create_portable_tasks(       TaskFunctor   taskFunctor
                          ,       TaskObject  * ptr
                          , const std::string & taskNamePassed
                          , void (TaskObject::*pmf1)( const PatchSubset                            * patches
                                                    , const MaterialSubset                         * matls
                                                    ,       OnDemandDataWarehouse                  * old_dw
                                                    ,       OnDemandDataWarehouse                  * new_dw
                                                    ,       UintahParams                           & uintahParams
                                                    ,       ExecutionObject<ExecSpace1, MemSpace1> & execObj
                                                    ,       Args...                                  args
                                                    )
                          , void (TaskObject::*pmf2)( const PatchSubset                            * patches
                                                    , const MaterialSubset                         * matls
                                                    ,       OnDemandDataWarehouse                  * old_dw
                                                    ,       OnDemandDataWarehouse                  * new_dw
                                                    ,       UintahParams                           & uintahParams
                                                    ,       ExecutionObject<ExecSpace2, MemSpace2> & execObj
                                                    ,       Args...                                  args
                                                    )
                          ,       SchedulerP  & sched
                          , const PatchSet    * patches
                          , const MaterialSet * matls
                          , const int           tg_num
                          ,       Args...       args
                          )
{
  Task* task{nullptr};
  std::string taskName = taskNamePassed;
  // Check for GPU tasks
  // GPU tasks take top priority
  if ( Uintah::Parallel::usingDevice() ) {
    if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value || std::is_same<Kokkos::Cuda, ExecSpace2>::value ) {
      taskName = taskName + " (GPUTask)";
      if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value ) {           /* Task supports Kokkos::Cuda builds */
        task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      }
      else if ( std::is_same<Kokkos::Cuda, ExecSpace2>::value ) {      /* Task supports Kokkos::Cuda builds */
        task = scinew Task( taskName, ptr, pmf2, std::forward<Args>(args)... );
      }

      //TODO: Consolodate these
      task->usesDevice(true);
      task->usesKokkosCuda(true);
      task->usesSimVarPreloading(true);

      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_CUDA, TaskAssignedMemorySpace::KOKKOS_CUDASPACE );
    }
  }

  // Check for CPU tasks if a GPU task did not get loaded
  if ( !task ) {
    if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value || std::is_same<Kokkos::OpenMP, ExecSpace2>::value ) {
      if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value ) {         /* Task supports Kokkos::OpenMP builds */
        task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      }
      else if ( std::is_same<Kokkos::OpenMP, ExecSpace2>::value ) {    /* Task supports Kokkos::OpenMP builds */
        task = scinew Task( taskName, ptr, pmf2, std::forward<Args>(args)... );
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_OPENMP, TaskAssignedMemorySpace::KOKKOS_HOSTSPACE );
    }
    else if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value || std::is_same<UintahSpaces::CPU, ExecSpace2>::value ) {
      if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value ) {      /* Task supports non-Kokkos builds */
        task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      }
      else if ( std::is_same<UintahSpaces::CPU, ExecSpace2>::value ) { /* Task supports non-Kokkos builds */
        task = scinew Task( taskName, ptr, pmf2, std::forward<Args>(args)... );
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::UINTAH_CPU, TaskAssignedMemorySpace::UINTAH_HOSTSPACE );
    }
  }

  if ( task ) {
    taskFunctor(task);
  }

  if ( task ) {
    sched->addTask( task, patches, matls, tg_num );
  }
}

//--------------------------------------------------------------------------------------------------
// One tag version of create_portable_arches_tasks()
template < typename TaskFunctor
         , typename TaskObject
         , typename ExecSpace1, typename MemSpace1
         , typename... Args >
void create_portable_tasks(       TaskFunctor   taskFunctor
                          ,       TaskObject  * ptr
                          , const std::string & taskNamePassed
                          , void (TaskObject::*pmf1)( const PatchSubset                            * patches
                                                    , const MaterialSubset                         * matls
                                                    ,       OnDemandDataWarehouse                  * old_dw
                                                    ,       OnDemandDataWarehouse                  * new_dw
                                                    ,       UintahParams                           & uintahParams
                                                    ,       ExecutionObject<ExecSpace1, MemSpace1> & execObj
                                                    ,       Args... args
                                                    )
                          ,       SchedulerP  & sched
                          , const PatchSet    * patches
                          , const MaterialSet * matls
                          , const int           tg_num
                          ,       Args...       args
                          )
{
  Task* task{nullptr};
  std::string taskName = taskNamePassed;
  // Check for GPU tasks
  // GPU tasks take top priority
  if ( Uintah::Parallel::usingDevice() ) {
    if ( std::is_same<Kokkos::Cuda, ExecSpace1>::value ) {           /* Task supports Kokkos::Cuda builds */
      taskName = taskName + " (GPUTask)";
      task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );

      //TODO: Consolodate these
      task->usesDevice(true);
      task->usesKokkosCuda(true);
      task->usesSimVarPreloading(true);

      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_CUDA, TaskAssignedMemorySpace::KOKKOS_CUDASPACE );
    }
  }

  // Check for CPU tasks if a GPU task did not get loaded
  if ( !task ) {
    if ( std::is_same<Kokkos::OpenMP, ExecSpace1>::value ) {         /* Task supports Kokkos::OpenMP builds */
      task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_OPENMP, TaskAssignedMemorySpace::KOKKOS_HOSTSPACE );
    }
    else if ( std::is_same<UintahSpaces::CPU, ExecSpace1>::value ) { /* Task supports non-Kokkos builds */
      task = scinew Task( taskName, ptr, pmf1, std::forward<Args>(args)... );
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::UINTAH_CPU, TaskAssignedMemorySpace::UINTAH_HOSTSPACE );
    }
  }

  if ( task ) {
    taskFunctor(task);
  }

  if ( task ) {
    sched->addTask( task, patches, matls, tg_num );
  }
}

} // end namespace Uintah

#endif // end #ifndef UINTAH_CORE_PARALLEL_TASK_DECLARATION_H
