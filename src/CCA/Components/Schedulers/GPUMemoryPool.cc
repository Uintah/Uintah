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

#include <CCA/Components/Schedulers/GPUMemoryPool.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DebugStream.h>

// Memory pools.
std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem,
              Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>* Uintah::GPUMemoryPool::gpuMemoryPoolInUse =
    new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem,
                      Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>;

std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem,
              Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>* Uintah::GPUMemoryPool::gpuMemoryPoolUnused =
    new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem,
                      Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>;

#if defined(USE_KOKKOS_INSTANCE)
#if defined(USE_KOKKOS_MALLOC)
// Not needed
#else // if defined(USE_KOKKOS_VIEW)
std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceViewItem,
              Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue> Uintah::GPUMemoryPool::gpuMemoryPoolViewInUse;
#endif
#else
// Not needed
#endif

extern Uintah::MasterLock cerrLock;

namespace Uintah {
  extern DebugStream gpu_stats;
}

namespace {
  struct pool_tag{};
  using  pool_monitor = Uintah::CrowdMonitor<pool_tag>;
}

namespace Uintah {

//______________________________________________________________________
//
void*
GPUMemoryPool::allocateCudaMemoryFromPool(unsigned int device_id,
                                          size_t memSize,
                                          const char * name)
{
  // Right now the memory pool assumes that each time step is going to
  // be using variables of the same size as the previous time step So
  // for that reason there should be 100% recycling after the 2nd
  // timestep or so.  If a task is constantly using different memory
  // sizes, this pool doesn't deallocate memory yet, so it will fail.

  void * addr = nullptr;

  {
    pool_monitor pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER };

    gpuMemoryPoolDeviceSizeItem item(device_id, memSize);

    std::multimap<gpuMemoryPoolDeviceSizeItem,
                  gpuMemoryPoolDeviceSizeValue>::iterator ret = gpuMemoryPoolUnused->find(item);

    if (ret != gpuMemoryPoolUnused->end()) {
      // Found an unused chunk of memory on this device.
      addr = ret->second.ptr;

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::UnifiedScheduler::myRankThread()
                    << " GPUMemoryPool::allocateCudaMemoryFromPool() -"
                    << " reusing space starting at " << addr
                    << " on device " << device_id
                    << " with size " << memSize
                    << " from the GPU memory pool"
                    << std::endl;
        }
        cerrLock.unlock();
      }

      // Insert it into the in use pool.
      gpuMemoryPoolDevicePtrItem  insertItem(device_id, addr);
      gpuMemoryPoolDevicePtrValue insertValue(99999, memSize);

      gpuMemoryPoolInUse->insert(std::pair<gpuMemoryPoolDevicePtrItem,
                                           gpuMemoryPoolDevicePtrValue>(insertItem, insertValue));
      // Remove it from the unsed pool.
      gpuMemoryPoolUnused->erase(ret);
    } else {
      // No chunk of memory found on this device so create it.

      // Base call is commented out
      // OnDemandDataWarehouse::uintahSetCudaDevice(device_id);

      // Allocate the memory.
#if defined(USE_KOKKOS_INSTANCE)

    std::string label;
    if(name == nullptr)
      label = std::string("device");
    else
      label = std::string(name);

#if defined(USE_KOKKOS_MALLOC)
      addr =
        Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(label, memSize);

#else // if defined(USE_KOKKOS_VIEW)
      Kokkos::View<char*, Kokkos::DefaultExecutionSpace::memory_space>
        deviceView(label, memSize);

      // ARS - FIX ME - The view should be schelped around.
      addr = deviceView.data();
#endif

#else
      cudaError_t err = cudaMalloc(&addr, memSize);
      if (err == cudaErrorMemoryAllocation) {
        printf("The GPU memory pool is full.  Need to clear!\n");
        exit(-1);
      }
#endif
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::UnifiedScheduler::myRankThread()
              << " GPUMemoryPool::allocateCudaMemoryFromPool() -"
              << " allocated GPU space starting at " << addr
              << " on device " << device_id
              << " with size " << memSize
              << std::endl;
        }
        cerrLock.unlock();
      }

      // Insert the address into the in use pool.
      gpuMemoryPoolDevicePtrItem  insertItem(device_id, addr);
      gpuMemoryPoolDevicePtrValue insertValue(99999, memSize);

      gpuMemoryPoolInUse->insert(std::pair<gpuMemoryPoolDevicePtrItem,
                                           gpuMemoryPoolDevicePtrValue>(insertItem, insertValue));

#if defined(USE_KOKKOS_INSTANCE)
#if defined(USE_KOKKOS_MALLOC)
// Not needed
#else // if defined(USE_KOKKOS_VIEW)
      // Keep a copy of the view so the view reference count is
      // incremented which prevents the memory from being de-allocated.
      gpuMemoryPoolDeviceViewItem insertView(device_id, deviceView);
      gpuMemoryPoolViewInUse.insert(std::pair<gpuMemoryPoolDeviceViewItem,
                                              gpuMemoryPoolDevicePtrValue>(insertView, insertValue));
#endif
#else
// Not needed
#endif
    }
  } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

  return addr;
}


//______________________________________________________________________
//
#if defined(USE_KOKKOS_INSTANCE)
#if defined(USE_KOKKOS_MALLOC)
void GPUMemoryPool::freeCudaMemoryFromPool()
{
  for(const auto &item: *gpuMemoryPoolUnused)
  {
    void * addr = item.second.ptr;
    Kokkos::kokkos_free(addr);
  }

  gpuMemoryPoolUnused->clear();

  // The data warehouses have not been cleared so Kokkos pointers are
  // still valid as they are reference counted.
  for(const auto &item: *gpuMemoryPoolInUse)
  {
    void * addr = item.first.ptr;
    Kokkos::kokkos_free(addr);
  }

  gpuMemoryPoolInUse->clear();
}
#else // if defined(USE_KOKKOS_VIEW)
void GPUMemoryPool::freeViewsFromPool()
{
  // By clearing the pool the view reference count is decremented and if
  // zero is deleted.
  gpuMemoryPoolViewInUse.clear();
}
#endif
#else
void GPUMemoryPool::freeCudaMemoryFromPool()
{
  for(const auto &item: *gpuMemoryPoolUnused)
  {
    void * addr = item.second.ptr;
    cudaFree(addr);
  }

  gpuMemoryPoolUnused->clear();

  // The data warehouses have not been cleared so Kokkos pointers are
  // still valid as they are reference counted.
  // for(const auto &item: *gpuMemoryPoolInUse)
  // {
  //   void * addr = item.first.ptr;
  //   cudaFree(addr);
  // }

  gpuMemoryPoolInUse->clear();
}
#endif


//______________________________________________________________________
//
bool GPUMemoryPool::reclaimCudaMemoryIntoPool(unsigned int device_id, void* addr)
{
  if(addr != nullptr)
  {
    pool_monitor pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER };

    size_t memSize;
    gpuMemoryPoolDevicePtrItem item(device_id, addr);

    std::multimap<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>::iterator ret = gpuMemoryPoolInUse->find(item);

    if (ret != gpuMemoryPoolInUse->end()){
      // Found this chunk of memory on this device.
      memSize = ret->second.size;

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::UnifiedScheduler::myRankThread()
                    << " GPUMemoryPool::reclaimCudaMemoryIntoPool() -"
                    << " space starting at " << addr
                    << " on device " << device_id
                    << " with size " << memSize
                    << " marked for reuse in the GPU memory pool"
                    << std::endl;
        }
        cerrLock.unlock();
      }

      // Insert it into the unused pool.
      gpuMemoryPoolDeviceSizeItem  insertItem(device_id, memSize);
      gpuMemoryPoolDeviceSizeValue insertValue(addr);

      gpuMemoryPoolUnused->insert(std::pair<gpuMemoryPoolDeviceSizeItem,
                                            gpuMemoryPoolDeviceSizeValue>(insertItem, insertValue));
      gpuMemoryPoolInUse->erase(ret);

    } else {
      // Ignore if the pools are empty as Uintah is shutting down.
      if(gpuMemoryPoolUnused->size() || gpuMemoryPoolInUse->size() )
      {
        printf("ERROR: GPUMemoryPool::reclaimCudaMemoryIntoPool - "
               "No memory found at pointer %p on device %u\n", addr, device_id);
        return false;
      }
    }
  } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

  return true;
}

} //end namespace Uintah
