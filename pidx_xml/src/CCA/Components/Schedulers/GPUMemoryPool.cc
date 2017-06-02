/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Util/DebugStream.h>

#include <mutex>


std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem, Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>* Uintah::GPUMemoryPool::gpuMemoryPoolInUse =
    new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem, Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>;

std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem, Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>* Uintah::GPUMemoryPool::gpuMemoryPoolUnused =
    new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem, Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>;


extern DebugStream gpu_stats;
extern std::mutex  cerrLock;

namespace {

struct pool_tag{};
using  pool_monitor = Uintah::CrowdMonitor<pool_tag>;

}


namespace Uintah {



//______________________________________________________________________
//
void*
GPUMemoryPool::allocateCudaSpaceFromPool(unsigned int device_id, size_t memSize) {

  //Right now the memory pool assumes that each time step is going to be using variables of the same size as the previous time step
  //So for that reason there should be 100% recycling after the 2nd timestep or so.
  //If a task is constantly using different memory sizes, this pool doesn't deallocate memory yet, so it will fail.

  void * addr = nullptr;
  {
    pool_monitor pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER };
    cudaError_t err;

    gpuMemoryPoolDeviceSizeItem item(device_id, memSize);

    std::multimap<gpuMemoryPoolDeviceSizeItem,gpuMemoryPoolDeviceSizeValue>::iterator ret = gpuMemoryPoolUnused->find(item);

    if (ret != gpuMemoryPoolUnused->end()){
      //we found one
      addr = ret->second.ptr;
      if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUMemoryPool::allocateCudaSpaceFromPool() -"
            << " reusing space starting at " << addr
            << " on device " << device_id
            << " with size " << memSize
            << " from the GPU memory pool"
            << std::endl;
      }
      cerrLock.unlock();
      }
      gpuMemoryPoolDevicePtrValue insertValue;
      insertValue.timestep = 99999;
      insertValue.size = memSize;
      gpuMemoryPoolInUse->insert(std::pair<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>(gpuMemoryPoolDevicePtrItem(device_id,addr), insertValue));
      gpuMemoryPoolUnused->erase(ret);
    } else {
      //There wasn't one
      //Set the device
      OnDemandDataWarehouse::uintahSetCudaDevice(device_id);

      //Allocate the memory.
      err = cudaMalloc(&addr, memSize);
      if (err == cudaErrorMemoryAllocation) {
        printf("The pool is full.  Need to clear!\n");
      }

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUMemoryPool::allocateCudaSpaceFromPool() -"
              << " allocated GPU space starting at " << addr
              << " on device " << device_id
              << " with size " << memSize
              << std::endl;
        }
        cerrLock.unlock();
      }
      gpuMemoryPoolDevicePtrValue insertValue;
      insertValue.timestep = 99999;
      insertValue.size = memSize;
      gpuMemoryPoolInUse->insert(std::pair<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>(gpuMemoryPoolDevicePtrItem(device_id,addr), insertValue));
    }
  } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

  return addr;

}


//______________________________________________________________________
//
 bool GPUMemoryPool::freeCudaSpaceFromPool(unsigned int device_id, void* addr) {

  {
    pool_monitor pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER };

    size_t memSize;
    gpuMemoryPoolDevicePtrItem item(device_id, addr);

    std::multimap<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>::iterator ret = gpuMemoryPoolInUse->find(item);

    if (ret != gpuMemoryPoolInUse->end()){
      //We found it
      memSize = ret->second.size;

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUMemoryPool::freeCudaSpaceFromPool() -"
              << " space starting at " << addr
              << " on device " << device_id
              << " with size " << memSize
              << " marked for reuse in the GPU memory pool"
              << std::endl;
        }
        cerrLock.unlock();
      }
      gpuMemoryPoolDeviceSizeItem insertItem(device_id, memSize);
      gpuMemoryPoolDeviceSizeValue insertValue;
      insertValue.ptr = addr;

      gpuMemoryPoolUnused->insert(std::pair<gpuMemoryPoolDeviceSizeItem, gpuMemoryPoolDeviceSizeValue>(insertItem, insertValue));
      gpuMemoryPoolInUse->erase(ret);

    } else {
      printf("ERROR: GPUMemoryPool::freeCudaSpaceFromPool - No memory found at pointer %p on device %u\n", addr, device_id);
      return false;
    }
  } // end pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER }

  return true;


  //TODO: Actually deallocate!!!
}


} //end namespace Uintah
