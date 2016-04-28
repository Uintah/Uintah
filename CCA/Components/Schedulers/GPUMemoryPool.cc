/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <Core/Util/DebugStream.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

//#include <sci_defs/cuda_defs.h>
//#include <map>
//#include <string>

#include <Core/Thread/Mutex.h>



using Uintah::CrowdMonitor;
//using std::endl;

std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolItem, Uintah::GPUMemoryPool::gpuMemoryData>* Uintah::GPUMemoryPool::gpuMemoryPool = new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolItem, Uintah::GPUMemoryPool::gpuMemoryData>;

CrowdMonitor * Uintah::GPUMemoryPool::gpuPoolLock = new CrowdMonitor("gpu pool lock");


extern DebugStream gpu_stats;
extern Mutex cerrLock;
namespace Uintah {



//______________________________________________________________________
//
void*
GPUMemoryPool::allocateCudaSpaceFromPool(int device_id, size_t memSize) {

  gpuPoolLock->writeLock();

  void * addr = NULL;
  bool claimedAnItem = false;

  gpuMemoryPoolItem gpuItem(device_id, memSize);

  std::pair <std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator,
             std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator> ret;

  ret = gpuMemoryPool->equal_range(gpuItem);
  std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator gpuPoolIter = ret.first;
  while (!claimedAnItem && gpuPoolIter != ret.second) {
    if (gpuPoolIter->second.status == 0) {
      //claim this one
      addr = gpuPoolIter->second.ptr;
      gpuPoolIter->second.status = 1;
      claimedAnItem = true;
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUMemoryPool::allocateCudaSpaceFromPool() -"
              << " reusing space starting at " << gpuPoolIter->second.ptr
              << " on device " << device_id
              << " with size " << memSize
              << " from the GPU memory pool"
              << endl;
        }
        cerrLock.unlock();
      }

    } else {
      ++gpuPoolIter;
    }
  }
  //No open spot in the pool, go ahead and allocate it.
  if (!claimedAnItem) {
    OnDemandDataWarehouse::uintahSetCudaDevice(device_id);
    CUDA_RT_SAFE_CALL( cudaMalloc(&addr, memSize) );
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUMemoryPool::allocateCudaSpaceFromPool() -"
            << " allocated GPU space starting at " << addr
            << " on device " << device_id
            << " with size " << memSize
            << endl;
      }
      cerrLock.unlock();
    }
    gpuMemoryData gmd;
    gmd.status = 1;
    gmd.timestep = 99999999; //Fix me
    gmd.ptr = addr;
    gpuPoolIter = gpuMemoryPool->insert(std::pair<gpuMemoryPoolItem, gpuMemoryData>(gpuItem,gmd));
  }
  gpuPoolLock->writeUnlock();
  return addr;

}


//______________________________________________________________________
//
 bool GPUMemoryPool::freeCudaSpaceFromPool(int device_id, size_t memSize, void* addr){
  gpuPoolLock->writeLock();
  if (memSize == 0) {
    printf("ERROR:\nGPUMemoryPool::freeCudaSpaceFromPool(), requesting to free from pool memory of size zero at address %p\n", addr);
    gpuPoolLock->writeUnlock();
    return false;
  }
  bool foundItem = false;


  /*//For debugging, shows everything in the pool
  std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator end;
  for (std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator it = gpuMemoryPool->begin();
       it !=  gpuMemoryPool->end();
       ++it) {
    gpu_stats << "device: " << it->first.device_id
              << " deviceSize: " << it->first.deviceSize << " - "
              << " status: " << it->second.status
              << " timestep: " << it->second.timestep
              << " ptr: " << it->second.ptr
              << endl;
  }
  gpu_stats << endl;
  */
  gpuMemoryPoolItem gpuItem(device_id, memSize);

  std::pair <std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator,
             std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator> ret;

  ret = gpuMemoryPool->equal_range(gpuItem);
  std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator gpuPoolIter = ret.first;

  while (!foundItem && gpuPoolIter != ret.second) {
    if (gpuPoolIter->second.ptr == addr) {
      //Found it.
      //Mark it as reusable
      gpuPoolIter->second.status = 0;
      foundItem = true;

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUMemoryPool::freeCudaSpaceFromPool() -"
              << " space starting at " << addr
              << " on device " << device_id
              << " with size " << memSize
              << " marked for reuse in the GPU memory pool"
              << endl;
        }
        cerrLock.unlock();
      }

    } else {
      ++gpuPoolIter;
    }
  }
  gpuPoolLock->writeUnlock();
  return foundItem;
  //TODO: Actually deallocate!!!
}


} //end namespace Uintah
