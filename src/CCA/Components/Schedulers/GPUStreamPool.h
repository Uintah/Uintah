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

#ifndef CCA_COMPONENTS_SCHEDULERS_GPUSTREAMPOOL_H
#define CCA_COMPONENTS_SCHEDULERS_GPUSTREAMPOOL_H

#include <sci_defs/gpu_defs.h>

#include <map>
#include <queue>

namespace Uintah {

#ifdef TASK_MANAGES_EXECSPACE
class Task;
#else
class DetailedTask;
#endif

class GPUStreamPool {

public:

  // Streams
#ifdef TASK_MANAGES_EXECSPACE
  #ifdef USE_KOKKOS_INSTANCE
    // Not needed instances are managed by DetailedTask/Task.
  #elif defined(HAVE_CUDA) // CUDA only when using streams
    static cudaStream_t* getCudaStreamFromPool(const Task * task, int device);
    static void reclaimCudaStreamsIntoPool(intptr_t dTask, Task * task);
    static void freeCudaStreamsFromPool();
  #endif
#else
  static cudaStream_t* getCudaStreamFromPool(const DetailedTask * task, int device);
  static void reclaimCudaStreamsIntoPool(DetailedTask * task);
  static void freeCudaStreamsFromPool();
#endif

private:

  // Streams

  // Thread shared data, needs lock protection when accessed

  // Operations within the same stream are ordered (FIFO) and cannot
  // overlap.  Operations in different streams are unordered and can
  // overlap. For this reason we let each task own a stream, as we
  // want one task to be able to run if it is ready to do work even if
  // another task is not yet ready.  It also enables us to easily
  // determine when a computed variable is "valid" because when that
  // task's stream completes, then we can infer the variable is ready
  // to go.  More about how a task claims a stream can be found in
  // DetailedTasks.cc
#ifdef TASK_MANAGES_EXECSPACE
  #ifdef USE_KOKKOS_INSTANCE
    // Not needed instances are managed by DetailedTask/Task.
  #elif defined(HAVE_CUDA) // CUDA only when using streams
    static std::map <unsigned int, std::queue<cudaStream_t*> > s_idle_streams;
  #endif
  static std::map <unsigned int, std::queue<cudaStream_t*> > s_idle_streams;
#endif
};

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_GPUSTREAMPOOL_H
