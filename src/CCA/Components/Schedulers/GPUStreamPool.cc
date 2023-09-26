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

#include <CCA/Components/Schedulers/GPUStreamPool.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DebugStream.h>


#if defined(USE_KOKKOS_INSTANCE)
  // Not needed instances are managed by DetailedTask/Task.
#elif defined(HAVE_CUDA) // CUDA only when using streams
std::map <unsigned int, std::queue<cudaStream_t*> > Uintah::GPUStreamPool::s_idle_streams;
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
}

namespace Uintah {

#ifdef USE_KOKKOS_INSTANCE
// Not needed instances are managed by DetailedTask/Task.
#elif defined(HAVE_CUDA) // CUDA only when using streams
//______________________________________________________________________
//
cudaStream_t*
#ifdef TASK_MANAGES_EXECSPACE
GPUStreamPool::getCudaStreamFromPool(const Task * task, int device)
#else
GPUStreamPool::getCudaStreamFromPool(const DetailedTask * task, int device)
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
GPUStreamPool::reclaimCudaStreamsIntoPool(intptr_t dTask, Task * task )
#else
GPUStreamPool::reclaimCudaStreamsIntoPool(DetailedTask * task )
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
GPUStreamPool::freeCudaStreamsFromPool()
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
