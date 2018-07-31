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

template <typename TaskFunctor, typename TaskObject,
          typename ES1, typename MS1,
          typename ES2, typename MS2,
          typename ES3, typename MS3,
          typename... Args>
void create_portable_tasks(TaskFunctor taskFunctor,
                           TaskObject* ptr,
                           const std::string& taskName,
                           void (TaskObject::*pmf1)(
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               OnDemandDataWarehouse* old_dw,
                               OnDemandDataWarehouse* new_dw,
                               UintahParams& uintahParams,
                               ExecutionObject<ES1, MS1>& executionObject,
                               Args... args),
                           void (TaskObject::*pmf2)(
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               OnDemandDataWarehouse* old_dw,
                               OnDemandDataWarehouse* new_dw,
                               UintahParams& uintahParams,
                               ExecutionObject<ES2, MS2>& executionObject,
                               Args... args),
                           void (TaskObject::*pmf3)(
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               OnDemandDataWarehouse* old_dw,
                               OnDemandDataWarehouse* new_dw,
                               UintahParams& uintahParams,
                               ExecutionObject<ES3, MS3>& executionObject,
                               Args... args),
                           SchedulerP & sched,
                           const PatchSet    * patches,
                           const MaterialSet * matls,
                           const int           tg_num = -1,
                           Args...          args){

  Task* task{nullptr};

  //See if there are any Cuda Tasks
  if (Uintah::Parallel::usingDevice()) {
    // GPU tasks take top priority
    if ( std::is_same< Kokkos::Cuda , ES1 >::value
        || std::is_same< Kokkos::Cuda , ES2 >::value
        || std::is_same< Kokkos::Cuda , ES3 >::value ) {
      if (std::is_same< Kokkos::Cuda , ES1 >::value) {
        task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      } else if (std::is_same< Kokkos::Cuda , ES2 >::value) {
         task = scinew Task(taskName, ptr, pmf2, std::forward<Args>(args)...);
      } else if (std::is_same< Kokkos::Cuda , ES3 >::value) {
         task = scinew Task(taskName, ptr, pmf3, std::forward<Args>(args)...);
      }

      //TODO: Consolodate these
      task->usesDevice(true);
      task->usesKokkosCuda(true);
      task->usesSimVarPreloading(true);

      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_CUDA, TaskAssignedMemorySpace::KOKKOS_CUDASPACE );
    }
  }
  //If a GPU task didn't get loaded, then check for CPU task options.
  if (!task) {
    if ( std::is_same< Kokkos::OpenMP , ES1 >::value
        || std::is_same< Kokkos::OpenMP , ES2 >::value
        || std::is_same< Kokkos::OpenMP , ES3 >::value ) {
      if (std::is_same< Kokkos::OpenMP , ES1 >::value) {
        task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      } else if (std::is_same< Kokkos::OpenMP , ES2 >::value) {
         task = scinew Task(taskName, ptr, pmf2, std::forward<Args>(args)...);
      } else if (std::is_same< Kokkos::OpenMP , ES3 >::value) {
         task = scinew Task(taskName, ptr, pmf3, std::forward<Args>(args)...);
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_OPENMP, TaskAssignedMemorySpace::KOKKOS_HOSTSPACE );
    } else if ( std::is_same< UintahSpaces::CPU , ES1 >::value
        || std::is_same< UintahSpaces::CPU , ES2 >::value
        || std::is_same< UintahSpaces::CPU , ES3 >::value ) {
      if (std::is_same< UintahSpaces::CPU , ES1 >::value) {
        task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      } else if (std::is_same< UintahSpaces::CPU , ES2 >::value) {
         task = scinew Task(taskName, ptr, pmf2, std::forward<Args>(args)...);
      } else if (std::is_same< UintahSpaces::CPU , ES3 >::value) {
         task = scinew Task(taskName, ptr, pmf3, std::forward<Args>(args)...);
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::UINTAH_CPU, TaskAssignedMemorySpace::UINTAH_HOSTSPACE );
    }
  }


  if (task) {
    taskFunctor(task);
  }

  if (task) {
    sched->addTask(task, patches, matls, tg_num);
  }
}

// The 2 function pointer verison
template <typename TaskFunctor, typename TaskObject,
          typename ES1, typename MS1,
          typename ES2, typename MS2,
          typename... Args>
void create_portable_tasks(TaskFunctor taskFunctor,
                           TaskObject* ptr,
                           const std::string& taskName,
                           void (TaskObject::*pmf1)(
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               OnDemandDataWarehouse* old_dw,
                               OnDemandDataWarehouse* new_dw,
                               UintahParams& uintahParams,
                               ExecutionObject<ES1, MS1>& executionObject,
                               Args... args),
                           void (TaskObject::*pmf2)(
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               OnDemandDataWarehouse* old_dw,
                               OnDemandDataWarehouse* new_dw,
                               UintahParams& uintahParams,
                               ExecutionObject<ES2, MS2>& executionObject,
                               Args... args),
                           SchedulerP & sched,
                           const PatchSet    * patches,
                           const MaterialSet * matls,
                           const int           tg_num = -1,
                           Args...          args){

  Task* task{nullptr};

  //See if there are any Cuda Tasks
  if (Uintah::Parallel::usingDevice()) {
    // GPU tasks take top priority
    if ( std::is_same< Kokkos::Cuda , ES1 >::value
        || std::is_same< Kokkos::Cuda , ES2 >::value ) {
      if (std::is_same< Kokkos::Cuda , ES1 >::value) {
        task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      } else if (std::is_same< Kokkos::Cuda , ES2 >::value) {
         task = scinew Task(taskName, ptr, pmf2, std::forward<Args>(args)...);
      }

      //TODO: Consolodate these
      task->usesDevice(true);
      task->usesKokkosCuda(true);
      task->usesSimVarPreloading(true);

      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_CUDA, TaskAssignedMemorySpace::KOKKOS_CUDASPACE );
    }
  }

  //If a GPU task didn't get loaded, then check for CPU task options.
  if (!task) {
    if ( std::is_same< Kokkos::OpenMP , ES1 >::value
        || std::is_same< Kokkos::OpenMP , ES2 >::value) {
      if (std::is_same< Kokkos::OpenMP , ES1 >::value) {
        task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      } else if (std::is_same< Kokkos::OpenMP , ES2 >::value) {
         task = scinew Task(taskName, ptr, pmf2, std::forward<Args>(args)...);
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_OPENMP, TaskAssignedMemorySpace::KOKKOS_HOSTSPACE );
    } else if ( std::is_same< UintahSpaces::CPU , ES1 >::value
        || std::is_same< UintahSpaces::CPU , ES2 >::value ) {
      if (std::is_same< UintahSpaces::CPU , ES1 >::value) {
        task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      } else if (std::is_same< UintahSpaces::CPU , ES2 >::value) {
         task = scinew Task(taskName, ptr, pmf2, std::forward<Args>(args)...);
      }
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::UINTAH_CPU, TaskAssignedMemorySpace::UINTAH_HOSTSPACE );
    }
  }


  if (task) {
    taskFunctor(task);
  }

  if (task) {
    sched->addTask(task, patches, matls, tg_num);
  }
}


// The 1 function pointer version
template <typename TaskFunctor, typename TaskObject,
          typename ES1, typename MS1,
          typename... Args>
void create_portable_tasks(TaskFunctor taskFunctor,
                           TaskObject* ptr,
                           const std::string& taskName,
                           void (TaskObject::*pmf1)(
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               OnDemandDataWarehouse* old_dw,
                               OnDemandDataWarehouse* new_dw,
                               UintahParams& uintahParams,
                               ExecutionObject<ES1, MS1>& executionObject,
                               Args... args),
                           SchedulerP & sched,
                           const PatchSet    * patches,
                           const MaterialSet * matls,
                           const int           tg_num = -1,
                           Args...          args){

  Task* task{nullptr};

  //See if there are any Cuda Tasks
  if (Uintah::Parallel::usingDevice()) {
    // GPU tasks take top priority
    if ( std::is_same< Kokkos::Cuda , ES1 >::value ) {

      task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);

      //TODO: Consolodate these
      task->usesDevice(true);
      task->usesKokkosCuda(true);
      task->usesSimVarPreloading(true);

      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_CUDA, TaskAssignedMemorySpace::KOKKOS_CUDASPACE );
    }
  }
  //If a GPU task didn't get loaded, then check for CPU task options.
  if (!task) {
    if ( std::is_same< Kokkos::OpenMP , ES1 >::value ) {
      task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::KOKKOS_OPENMP, TaskAssignedMemorySpace::KOKKOS_HOSTSPACE );
    } else if ( std::is_same< UintahSpaces::CPU , ES1 >::value ) {
      task = scinew Task(taskName, ptr, pmf1, std::forward<Args>(args)...);
      task->setExecutionAndMemorySpace( TaskAssignedExecutionSpace::UINTAH_CPU, TaskAssignedMemorySpace::UINTAH_HOSTSPACE );
    }
  }


  if (task) {
    taskFunctor(task);
  }

  if (task) {
    sched->addTask(task, patches, matls, tg_num);
  }
}
} // end namespace Uintah
#endif ///UINTAH_CORE_PARALLEL_TASK_DECLARATION_H
