/* Copyright (c) 2014 The University of Utah
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
/**
 *  Available debugging flags:
 *    - DEBUG_CUDA_VERBOSE, turns on all CUDA related debugging flags
 *
 *    - DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR, if defined this will output device manager
 *      information. (Everything related to identifying and managing GPU devices)
 *
 *    - DEBUG_EXT_ALLOC_MEM, output all memory related activities
 *
 *    - DEBUG_EXT_ALLOC_CUDA_SHARED_PTR, output information related to the allocation
 *      and destruction of shared pointer objects
 */


#ifdef DEBUG_CUDA_VERBOSE
#define DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
#define DEBUG_EXT_ALLOC_MEM
#define DEBUG_EXT_CUDA_SHARED_PTR
#define DEBUG_EXT_SET_MEM
#endif

//Standard includes
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

//Spatial Operator includes
#include <spatialops/structured/ExternalAllocators.h>

//Boost includes
#ifdef ENABLE_THREADS
#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#endif

#ifndef CUDAMEMORYALLOCATOR_H_
#define CUDAMEMORYALLOCATOR_H_

#define byte char;

namespace ema {
namespace cuda {

/*---- CUDA wrappers with error checking/processing */
void CudaSetDevice(const int device);
void CudaMalloc(void** src, const size_t sz, const unsigned int device);
void CudaHostAlloc(void**, const size_t sz, const unsigned int device);
void CudaFree(void* src, const unsigned int device);
void CudaFreeHost(void* src, const unsigned int device);
void CudaMemcpy(void* src, const void* dest, const unsigned int device, const size_t sz,
                cudaMemcpyKind cmkk);
void CudaStreamSync(cudaStream_t stream );

/* \brief Device management structure for all GPU devices */
class CUDADeviceManager {
    friend class CUDADeviceInterface;
    CUDADeviceManager();

  public:

    ~CUDADeviceManager();

    /** \brief Return reference to the device manager object */
    static CUDADeviceManager& self();

    /** \brief Returns the number of available CUDA capable compute devices */
    int get_device_count() const;

    /** \brief perform synchronization on a stream */
    void sync_stream( cudaStream_t stream );

    /** \brief return sthe best possible device from multiple GPUs */
    int get_best_device() const;

    /** \brief Returns the memory structure associated with device K */
    void get_memory_statistics( CUDAMemStats& cms, const int K = 0 ) const;

    /** \brief Updates the 'device_stats' structures with the most current memory usage statistics
     * Please note that it is possible memory can be allocated from other sources, this
     * is simply to provide a metric for checking relative memory availability.
     * */
    void update_memory_statistics();

    /** \brief output a list of all available CUDA hardware and compute capabilities */
    void print_device_info() const;

  private:
    int device_count;                          ///< Number of CUDA capable devices
    std::vector<cudaDeviceProp*> device_props; ///< Capability information for each compute device
    std::vector<CUDAMemStats*> device_stats;   ///< Global memory statistics for each device


    //TODO-> not sure if this needs a mutex... maybe later (NEED SOME MORE INFO TO IT)
    /**
     *  \class ExecMutex
     *  \brief Scoped lock. An instance should be constructed within any function that touches
     *   thread-shared memory.
     */
    class ExecMutex {
#   ifdef ENABLE_THREADS
      const boost::mutex::scoped_lock lock;
      inline boost::mutex& get_mutex() const {static boost::mutex m; return m;}

    public:
      ExecMutex() : lock( get_mutex() ) {}
      ~ExecMutex() {}
#   else
    public:
      ExecMutex(){}
      ~ExecMutex(){}
#   endif
    };
};

} // End Namespace ema_CUDA
} // End Namespace ema

#endif /* CUDAMEMORYALLOCATOR_H_ */
