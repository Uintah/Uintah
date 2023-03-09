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

#ifdef USE_KOKKOS_INSTANCE
  // Not needed instances are managed by DetailedTask/Task.
#elif defined(HAVE_CUDA) // CUDA only when using streams
std::map <unsigned int, std::queue<cudaStream_t*> > Uintah::GPUMemoryPool::s_idle_streams;//  =
    // new std::map <unsigned int, std::queue<cudaStream_t*> >;
#endif

extern Uintah::MasterLock cerrLock;

namespace Uintah {
  extern DebugStream gpu_stats;
}

namespace {
#ifdef USE_KOKKOS_INSTANCE
  // Not needed instances are managed by DetailedTask/Task.
#else
  Uintah::MasterLock idle_streams_mutex{};
#endif
  struct pool_tag{};
  using  pool_monitor = Uintah::CrowdMonitor<pool_tag>;
}

namespace Uintah {

//______________________________________________________________________
//
void*
GPUMemoryPool::allocateCudaMemoryFromPool(unsigned int device_id, size_t memSize)
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

#if defined(USE_KOKKOS_MALLOC)
      addr = Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(memSize);
      
#else // if defined(USE_KOKKOS_VIEW)
      // Kokkos equivalent - KokkosView
      Kokkos::View<char*, Kokkos::DefaultExecutionSpace::memory_space> deviceView( "device", memSize);

      // ARS - FIX ME - The view should be schelped around.
      // With CUDA the raw data pointer is schelped around.
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
      // ARS - FIX ME Keep a copy of the view so the view reference
      // count is incremented which prevents the memory from being
      // de-allocated.
      gpuMemoryPoolDeviceViewItem insertView(device_id, deviceView);
      gpuMemoryPoolViewInUse.insert(std::pair<gpuMemoryPoolDeviceViewItem,
                                              gpuMemoryPoolDevicePtrValue>(insertView, insertValue));
#endif
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

  for(const auto &item: *gpuMemoryPoolInUse)
  {
    void * addr = item.first.ptr;
    Kokkos::kokkos_free(addr);
  }
  
  gpuMemoryPoolInUse->clear();
}
#endif


//______________________________________________________________________
//
bool GPUMemoryPool::reclaimCudaMemoryIntoPool(unsigned int device_id, void* addr)
{
  {
    pool_monitor pool_write_lock{ Uintah::CrowdMonitor<pool_tag>::WRITER };

    size_t memSize;
    gpuMemoryPoolDevicePtrItem item(device_id, addr);

    std::multimap<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue>::iterator ret = gpuMemoryPoolInUse->find(item);

    if (ret != gpuMemoryPoolInUse->end()){
      // Found this chuck of memory on this device.
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


#ifdef USE_KOKKOS_INSTANCE
// Not needed instances are managed by DetailedTask/Task.
#elif defined(HAVE_CUDA) // CUDA only when using streams
//______________________________________________________________________
//
cudaStream_t*
#ifdef TASK_MANAGES_EXECSPACE
GPUMemoryPool::getCudaStreamFromPool(const Task * task, int device)
#else
GPUMemoryPool::getCudaStreamFromPool(const DetailedTask * task, int device)
#endif
{
  cudaError_t retVal;
  cudaStream_t* stream = nullptr;

  idle_streams_mutex.lock();
  {
    if (s_idle_streams[device].size() > 0) {
      stream = s_idle_streams[device].front();
      s_idle_streams[device].pop();
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
                    << " Issued a CUDA stream " << std::hex << stream
                    << " for task " << task->getName()
                    << " on device " << std::dec << device
                    << std::endl;
        }
        cerrLock.unlock();
      }
    } else {  // Shouldn't need any more than the queue capacity, but in case
      // Base call is commented out
      // OnDemandDataWarehouse::uintahSetCudaDevice(device);

      // This stream will be put into idle stream queue and
      // ultimately deallocated after the final timestep.
      stream = ((cudaStream_t*) malloc(sizeof(cudaStream_t)));
      CUDA_RT_SAFE_CALL(retVal = cudaStreamCreate(&(*stream)));

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
                    << " Created a CUDA stream " << std::hex << stream
                    << " for task " << task->getName()
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
// Operations within the same stream are ordered (FIFO) and cannot
// overlap.  Operations in different streams are unordered and can
// overlap For this reason we let each task own a stream, as we want
// one task to be able to run if it is able to do so even if another
// task is not yet ready.
void
#ifdef TASK_MANAGES_EXECSPACE
GPUMemoryPool::reclaimCudaStreamsIntoPool(intptr_t dTask, Task * task )
#else
GPUMemoryPool::reclaimCudaStreamsIntoPool(DetailedTask * task )
#endif
{
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
                << " Attempting to reclaim CUDA streams for task "
                << task->getName()
                << " at " << task
                << std::endl;
    }
    cerrLock.unlock();
  }

  // Reclaim the streams
#ifdef TASK_MANAGES_EXECSPACE
  std::set<unsigned int> deviceNums = task->getDeviceNums(dTask);
#else
  std::set<unsigned int> deviceNums = task->getDeviceNums();
#endif
  for ( auto &iter : deviceNums) {
#ifdef TASK_MANAGES_EXECSPACE
    cudaStream_t* stream = task->getCudaStreamForThisTask(dTask, iter);
#else
    cudaStream_t* stream = task->getCudaStreamForThisTask(iter);
#endif
    if (stream != nullptr) {

      idle_streams_mutex.lock();
      {
        s_idle_streams.operator[](iter).push(stream);
      }

      idle_streams_mutex.unlock();

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
                    << " Reclaimed CUDA stream " << std::hex << stream
                    << " on device " << std::dec << iter
                    << " for task " << task->getName()
                    << " at " << task
                    << std::endl;
        }
        cerrLock.unlock();
      }
    }
  }

  // Task objects persist between timesteps.  So remove all knowledge
  // of any formerly used streams.
#ifdef TASK_MANAGES_EXECSPACE
  task->clearCudaStreamsForThisTask(dTask);
#else
  task->clearCudaStreamsForThisTask();
#endif
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
        gpu_stats << UnifiedScheduler::myRankThread()
                  << " locking freeCudaStreams" << std::endl;
      }
      cerrLock.unlock();
    }

    for( auto &it : s_idle_streams) {
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
                    << " Preparing to deallocate " << it.second.size()
                    << " CUDA stream(s) for device #" << it.first
                    << std::endl;
        }
        cerrLock.unlock();
      }

      unsigned int device = it.first;

      // Base call is commented out
      // OnDemandDataWarehouse::uintahSetCudaDevice(device);

      while (!s_idle_streams[device].empty()) {
        cudaStream_t* stream = s_idle_streams[device].front();
        s_idle_streams[device].pop();
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                      << " Performing cudaStreamDestroy for stream " << stream
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
        gpu_stats << UnifiedScheduler::myRankThread()
                  << " unlocking freeCudaStreams " << std::endl;
      }
      cerrLock.unlock();
    }
  }

  idle_streams_mutex.unlock();
}
#endif

} //end namespace Uintah
