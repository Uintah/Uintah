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


#ifndef CCA_COMPONENTS_SCHEDULERS_GPUMEMORYPOOL_H
#define CCA_COMPONENTS_SCHEDULERS_GPUMEMORYPOOL_H

#include <sci_defs/gpu_defs.h>

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

#if defined(USE_KOKKOS_INSTANCE)
  #if defined(USE_KOKKOS_MALLOC)
  // Not needed for Kokkos malloc
  #else // if defined(USE_KOKKOS_VIEW)
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
#else
// Not needed for CUDA streams
#endif

  static void* allocateMemoryFromPool(unsigned int device_id,
                                      size_t memSize,
                                      const char *name = nullptr);

  static bool reclaimMemoryIntoPool(unsigned int device_id, void* addr);

#if defined(USE_KOKKOS_INSTANCE)
  #if defined(USE_KOKKOS_MALLOC)
    static void freeMemoryFromPool();
  #else // if defined(USE_KOKKOS_VIEW)
    static void freeViewsFromPool();
  #endif
#else
  static void freeMemoryFromPool();
#endif

private:

  // Memory pools.
  // For a given device and address, holds the timestep and size.
  static std::multimap<gpuMemoryPoolDevicePtrItem,
                       gpuMemoryPoolDevicePtrValue> *gpuMemoryPoolInUse;

  // For a given device and size, holds the address.
  static std::multimap<gpuMemoryPoolDeviceSizeItem,
                       gpuMemoryPoolDeviceSizeValue> *gpuMemoryPoolUnused;

#if defined(USE_KOKKOS_INSTANCE)
#if defined(USE_KOKKOS_MALLOC)
// Not needed for Kokkos malloc
#else // if defined(USE_KOKKOS_VIEW)
  // For a given device and view, holds the timestep and size.
  static std::multimap<gpuMemoryPoolDeviceViewItem,
                       gpuMemoryPoolDevicePtrValue> gpuMemoryPoolViewInUse;
#endif
#else
// Not needed for CUDA streams
#endif
};

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_GPUMEMORYPOOL_H
