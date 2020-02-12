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

#include <CCA/Components/Schedulers/GPUMemoryPool.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DebugStream.h>


std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem, Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>* Uintah::GPUMemoryPool::gpuMemoryPoolInUse =
    new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrItem, Uintah::GPUMemoryPool::gpuMemoryPoolDevicePtrValue>;

std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem, Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>* Uintah::GPUMemoryPool::gpuMemoryPoolUnused =
    new std::multimap<Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeItem, Uintah::GPUMemoryPool::gpuMemoryPoolDeviceSizeValue>;

std::map <unsigned int, std::queue<cudaStream_t*> > * Uintah::GPUMemoryPool::s_idle_streams =
    new std::map <unsigned int, std::queue<cudaStream_t*> >;

extern Uintah::MasterLock cerrLock;

namespace Uintah {
  extern DebugStream        gpu_stats;
}

namespace {
  Uintah::MasterLock idle_streams_mutex{};

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
        gpu_stats << UnifiedScheduler::UnifiedScheduler::myRankThread()
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
        printf("The GPU memory pool is full.  Need to clear!\n");
        exit(-1);
      }

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::UnifiedScheduler::myRankThread()
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
          gpu_stats << UnifiedScheduler::UnifiedScheduler::myRankThread()
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


 //______________________________________________________________________
 //
 void
 GPUMemoryPool::freeCudaStreamsFromPool()
 {
   cudaError_t retVal;


   idle_streams_mutex.lock();
   {
     if (gpu_stats.active()) {
       cerrLock.lock();
       {
         gpu_stats << UnifiedScheduler::myRankThread() << " locking freeCudaStreams" << std::endl;
       }
       cerrLock.unlock();
     }

     unsigned int totalStreams = 0;
     for (std::map<unsigned int, std::queue<cudaStream_t*> >::const_iterator it = s_idle_streams->begin(); it != s_idle_streams->end(); ++it) {
       totalStreams += it->second.size();
       if (gpu_stats.active()) {
         cerrLock.lock();
         {
           gpu_stats << UnifiedScheduler::myRankThread() << " Preparing to deallocate " << it->second.size()
                     << " CUDA stream(s) for device #"
                     << it->first
                     << std::endl;
         }
         cerrLock.unlock();
       }
     }

     for (std::map<unsigned int, std::queue<cudaStream_t*> >::const_iterator it = s_idle_streams->begin(); it != s_idle_streams->end(); ++it) {
       unsigned int device = it->first;
       OnDemandDataWarehouse::uintahSetCudaDevice(device);

       while (!s_idle_streams->operator[](device).empty()) {
         cudaStream_t* stream = s_idle_streams->operator[](device).front();
         s_idle_streams->operator[](device).pop();
         if (gpu_stats.active()) {
           cerrLock.lock();
           {
             gpu_stats << UnifiedScheduler::myRankThread() << " Performing cudaStreamDestroy for stream " << stream
                       << " on device " << device
                       << std::endl;
           }
           cerrLock.unlock();
         }
         CUDA_RT_SAFE_CALL(retVal = cudaStreamDestroy(*stream));
         free(stream);
       }
     }

     if (gpu_stats.active()) {
       cerrLock.lock();
       {
         gpu_stats << UnifiedScheduler::myRankThread() << " unlocking freeCudaStreams " << std::endl;
       }
       cerrLock.unlock();
     }
   }
   idle_streams_mutex.unlock();
 }


 //______________________________________________________________________
 //
 cudaStream_t *
 GPUMemoryPool::getCudaStreamFromPool( int device )
 {
   cudaError_t retVal;
   cudaStream_t* stream;

   idle_streams_mutex.lock();
   {
     if (s_idle_streams->operator[](device).size() > 0) {
       stream = s_idle_streams->operator[](device).front();
       s_idle_streams->operator[](device).pop();
       if (gpu_stats.active()) {
         cerrLock.lock();
         {
           gpu_stats << UnifiedScheduler::myRankThread()
                     << " Issued CUDA stream " << std::hex << stream
                     << " on device " << std::dec << device
                     << std::endl;
         }
         cerrLock.unlock();
       }
     } else {  // shouldn't need any more than the queue capacity, but in case
       OnDemandDataWarehouse::uintahSetCudaDevice(device);

       // this will get put into idle stream queue and ultimately deallocated after final timestep
       stream = ((cudaStream_t*) malloc(sizeof(cudaStream_t)));
       CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));

       if (gpu_stats.active()) {
         cerrLock.lock();
         {
           gpu_stats << UnifiedScheduler::myRankThread()
                     << " Needed to create 1 additional CUDA stream " << std::hex << stream
                     << " for device " << std::dec << device
                     << std::endl;
         }
         cerrLock.unlock();
       }
     }
   }
   idle_streams_mutex.unlock();

   return stream;
 }


 //______________________________________________________________________
 //
 //Operations within the same stream are ordered (FIFO) and cannot overlap.
 //Operations in different streams are unordered and can overlap
 //For this reason we let each task own a stream, as we want one task to be able to run
 //if it is able to do so even if another task is not yet ready.
 void
 GPUMemoryPool::reclaimCudaStreamsIntoPool( DetailedTask * dtask )
 {


   if (gpu_stats.active()) {
     cerrLock.lock();
     {
       gpu_stats << UnifiedScheduler::myRankThread()
                 << " Seeing if we need to reclaim any CUDA streams for task "
                 << dtask->getName()
                 << " at "
                 << dtask
                 << std::endl;
     }
     cerrLock.unlock();
   }

   // reclaim DetailedTask streams
   std::set<unsigned int> deviceNums = dtask->getDeviceNums();
   for (std::set<unsigned int>::iterator iter = deviceNums.begin(); iter != deviceNums.end(); ++iter) {

     cudaStream_t* stream = dtask->getCudaStreamForThisTask(*iter);
     if (stream != nullptr) {

        idle_streams_mutex.lock();
        {
          s_idle_streams->operator[](*iter).push(stream);
        }

        idle_streams_mutex.unlock();

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread() << " Reclaimed CUDA stream " << std::hex << stream
                    << " on device " << std::dec << *iter
                    << " for task " << dtask->getName() << " at " << dtask
                    << std::endl;
        }
        cerrLock.unlock();
      }
      // It seems that task objects persist between timesteps.  So make sure we remove
      // all knowledge of any formerly used streams.
      dtask->clearCudaStreamsForThisTask();
    }
  }
}



} //end namespace Uintah
