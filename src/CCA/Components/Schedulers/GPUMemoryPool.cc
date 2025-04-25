/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

extern Uintah::MasterLock cerrLock;

namespace Uintah {
  extern DebugStream gpu_stats;  // from KokkosScheduler
}

namespace {
  struct pool_tag{};
  using  pool_monitor = Uintah::CrowdMonitor<pool_tag>;
}

namespace Uintah {

//______________________________________________________________________
//
void*
GPUMemoryPool::allocateMemoryFromPool(unsigned int device_id,
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
                    << " GPUMemoryPool::allocateMemoryFromPool() -"
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

      // Allocate the memory.
      std::string label;
      if(name == nullptr)
	label = std::string("device");
      else
	label = std::string(name);

      addr =
	Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(label, memSize);

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::UnifiedScheduler::myRankThread()
              << " GPUMemoryPool::allocateMemoryFromPool() -"
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
    }
  } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

  return addr;
}


//______________________________________________________________________
//
void GPUMemoryPool::freeMemoryFromPool()
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


//______________________________________________________________________
//
bool GPUMemoryPool::reclaimMemoryIntoPool(unsigned int device_id, void* addr)
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
                    << " GPUMemoryPool::reclaimMemoryIntoPool() -"
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
        printf("ERROR: GPUMemoryPool::reclaimMemoryIntoPool - "
               "No memory found at pointer %p on device %u\n", addr, device_id);
        return false;
      }
    }
  } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

  return true;
}

} //end namespace Uintah
