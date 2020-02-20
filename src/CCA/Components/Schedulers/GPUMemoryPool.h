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
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h> //For myRankThread()
#include <map>
#include <queue>

namespace Uintah {

class GPUStreamPool;

class GPUMemoryPool;

class GPUMemoryPool {

public:

  struct gpuMemoryPoolDevicePtrValue {
    unsigned int timestep;
    size_t       size;
  };
  struct gpuMemoryPoolDevicePtrItem {

    unsigned int  device_id;
    void*         ptr;


    gpuMemoryPoolDevicePtrItem(unsigned int device_id, void* ptr) {
      this->device_id = device_id;
      this->ptr = ptr;
    }

    //This so it can be used in an STL map
    bool operator<(const gpuMemoryPoolDevicePtrItem& right) const {
      if (this->device_id < right.device_id) {
        return true;
      } else if ((this->device_id == right.device_id) && (this->ptr < right.ptr)) {
        return true;
      } else {
        return false;
      }
    }
  };

  struct gpuMemoryPoolDeviceSizeValue {
    void * ptr;
  };

  struct gpuMemoryPoolDeviceSizeItem {

    unsigned int  device_id;
    size_t        deviceSize;

    gpuMemoryPoolDeviceSizeItem(unsigned int device_id, size_t deviceSize) {
     this->device_id = device_id;
     this->deviceSize = deviceSize;
    }
    //This so it can be used in an STL map
    bool operator<(const gpuMemoryPoolDeviceSizeItem& right) const {
      if (this->device_id < right.device_id) {
        return true;
      } else if ((this->device_id == right.device_id) && (this->deviceSize < right.deviceSize)) {
        return true;
      } else {
        return false;
      }
    }
  };

  static void* allocateCudaSpaceFromPool(unsigned int device_id, size_t memSize);

  static bool freeCudaSpaceFromPool(unsigned int device_id, void* addr);

  static void reclaimCudaStreamsIntoPool( DetailedTask * dtask );

  static void freeCudaStreamsFromPool();

  static cudaStream_t* getCudaStreamFromPool( int device );

private:

  //For a given device and address, holds the timestep
  static std::multimap<gpuMemoryPoolDevicePtrItem, gpuMemoryPoolDevicePtrValue> *gpuMemoryPoolInUse;

  static std::multimap<gpuMemoryPoolDeviceSizeItem, gpuMemoryPoolDeviceSizeValue> *gpuMemoryPoolUnused;

  // thread shared data, needs lock protection when accessed

  //Operations within the same stream are ordered (FIFO) and cannot overlap.
  //Operations in different streams are unordered and can overlap
  //For this reason we let each task own a stream, as we want one task to be able to run
  //if it is ready to do work even if another task is not yet ready.  It also enables us
  //to easily determine when a computed variable is "valid" because when that task's stream
  //completes, then we can infer the variable is ready to go.  More about how a task claims a
  //stream can be found in DetailedTasks.cc
  static std::map <unsigned int, std::queue<cudaStream_t*> > * s_idle_streams;

};

} //end namespace

#endif // CCA_COMPONENTS_SCHEDULERS_GPUMEMORYPOOL_H
