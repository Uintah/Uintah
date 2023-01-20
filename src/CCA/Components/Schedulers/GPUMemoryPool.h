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


#ifndef CCA_COMPONENTS_SCHEDULERS_GPUMEMORYPOOL_H
#define CCA_COMPONENTS_SCHEDULERS_GPUMEMORYPOOL_H

#include <sci_defs/cuda_defs.h>
#include <sci_defs/kokkos_defs.h>

#ifdef HAVE_KOKKOS
  #include <Kokkos_Core.hpp>
#endif

#include <map>
#include <queue>

namespace Uintah {

#ifdef TASK_MANAGES_EXECSPACE
class Task;
#else
class DetailedTask;
#endif

class GPUMemoryPool {

public:

  // Memory pools.

  // Item device and address is the key in the memory in use pool.
  struct gpuMemoryPoolDevicePtrItem {

    unsigned int  device_id;
    void*         ptr;

    gpuMemoryPoolDevicePtrItem(unsigned int device_id, void* ptr) {
      this->device_id = device_id;
      this->ptr = ptr;
    }

    // Less than operator so it can be used in an STL map.
    bool operator<(const gpuMemoryPoolDevicePtrItem& right) const {
      if (this->device_id < right.device_id) {
        return true;
      } else if (this->device_id == right.device_id && this->ptr < right.ptr) {
        return true;
      } else {
        return false;
      }
    }
  };

  // The value is the memory size in the memory in use pool.
  struct gpuMemoryPoolDevicePtrValue {
    unsigned int timestep; // Not currently used.
    size_t       size;

    gpuMemoryPoolDevicePtrValue(unsigned int timestep, size_t size) {
      this->timestep = timestep;
      this->size = size;
    }
  };

  // Item device and size is the key in the memory unused pool.
  struct gpuMemoryPoolDeviceSizeItem {

    unsigned int  device_id;
    size_t        size;

    gpuMemoryPoolDeviceSizeItem(unsigned int device_id, size_t size) {
     this->device_id = device_id;
     this->size = size;
    }

    // Less than operator so it can be used in an STL map.
    bool operator<(const gpuMemoryPoolDeviceSizeItem& right) const {
      if (this->device_id < right.device_id) {
        return true;
      } else if (this->device_id == right.device_id &&
                 this->size < right.size) {
        return true;
      } else {
        return false;
      }
    }
  };

  // The value is the memory address in the memory unused pool.
  struct gpuMemoryPoolDeviceSizeValue {
    void * ptr;

    gpuMemoryPoolDeviceSizeValue(void* ptr) {
      this->ptr = ptr;
    }
  };

#if defined(USE_KOKKOS_VIEW) || defined(USE_KOKKOS_INSTANCE)
  struct gpuMemoryPoolDeviceViewItem {

    unsigned int  device_id;
    Kokkos::View<char*, Kokkos::DefaultExecutionSpace::memory_space> view;

    gpuMemoryPoolDeviceViewItem(unsigned int device_id, Kokkos::View<char*, Kokkos::DefaultExecutionSpace::memory_space> view) {
      this->device_id = device_id;
      this->view = view;
    }

    // Less than operator so it can be used in an STL map
    bool operator<(const gpuMemoryPoolDeviceViewItem& right) const {
      if (this->device_id < right.device_id) {
        return true;
      } else if (this->device_id == right.device_id &&
                 this->view.data() < right.view.data()) {
        return true;
      } else {
        return false;
      }
    }
  };
#endif

  static void* allocateCudaSpaceFromPool(unsigned int device_id, size_t memSize);

  static bool freeCudaSpaceFromPool(unsigned int device_id, void* addr);
#if defined(USE_KOKKOS_VIEW) || defined(USE_KOKKOS_INSTANCE)
  static void freeViewsFromPool();
#endif

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

  // Memory pools.
  // For a given device and address, holds the timestep and size.
  static std::multimap<gpuMemoryPoolDevicePtrItem,
                       gpuMemoryPoolDevicePtrValue> *gpuMemoryPoolInUse;

  // For a given device and size, holds the address.
  static std::multimap<gpuMemoryPoolDeviceSizeItem,
                       gpuMemoryPoolDeviceSizeValue> *gpuMemoryPoolUnused;

#if defined(USE_KOKKOS_VIEW) || defined(USE_KOKKOS_INSTANCE)
  // For a given device and view, holds the timestep and size.
  static std::multimap<gpuMemoryPoolDeviceViewItem,
                       gpuMemoryPoolDevicePtrValue> gpuMemoryPoolViewInUse;
#endif

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
#ifdef USE_KOKKOS_INSTANCE
  static std::map <unsigned int, std::queue<Kokkos::DefaultExecutionSpace> > s_idle_instances;
#elif defined(HAVE_CUDA) // CUDA only when using streams
  static std::map <unsigned int, std::queue<cudaStream_t*> > s_idle_streams;
#endif
};

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_GPUMEMORYPOOL_H
