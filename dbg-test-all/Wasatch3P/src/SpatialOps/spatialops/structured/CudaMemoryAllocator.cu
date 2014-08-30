/*
 * Copyright (c) 2014 The University of Utah
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

#include <spatialops/structured/CudaMemoryAllocator.h>

namespace ema {
namespace cuda {

void CudaSetDevice(const unsigned int device) {
      #ifdef DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
        std::cout << "CudaSetdevice wrapper called, setting thread device as : " << std::endl;
      #endif
cudaError err;

  if (cudaSuccess != (err = cudaSetDevice(device))) {
    std::ostringstream msg;
    msg << "Failed to set thread device, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - " << cudaGetErrorString(err);
    throw(std::runtime_error(msg.str()));
  }

#ifdef DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
  std::cout << "CudaSetDevice wrapper exiting \n";
#endif
}

/*--------------------------------------------------------------------*/

void CudaMalloc(void** src, const size_t sz, const unsigned int device) {

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMalloc wrapper called (src,size,device) -> (";
  std::cout << src << "," << sz << "," << device << ")" << std::endl;
#endif
  cudaError err;

  CudaSetDevice(device);
  if (cudaSuccess != (err = cudaMalloc(src,sz))) {
    std::ostringstream msg;
    msg << "CudaMalloc failed, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - " << cudaGetErrorString(err);
    throw(std::runtime_error(msg.str()));
  }

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMalloc pointer pointing to allocated memory on Device : " << device
            << "," << " address : " << *src << std::endl;
  std::cout << "CudaMalloc wrapper exiting \n";
#endif
}

void CudaHostAlloc(void** src, const size_t sz, const unsigned int device) {

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaHostAlloc wrapper called (src,size,device) -> (";
  std::cout << src << "," << sz << "," << device << ")" << std::endl;
#endif
  cudaError err;

  CudaSetDevice(device);
  if (cudaSuccess != (err = cudaHostAlloc(src,sz,cudaHostAllocPortable))) {
    std::ostringstream msg;
    msg << "CudaHostAlloc failed, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - " << cudaGetErrorString(err);
    throw(std::runtime_error(msg.str()));
  }

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaHostAlloc pointer to allocated Pinned memory for Device context: " << device
            << "," << " address : " << src << std::hex << " , " << src << std::endl;
  std::cout << "CudaHostAlloc wrapper exiting \n";
#endif
}

/*---------------------------------------------------------------------*/

void CudaFree(void* src, const unsigned int device) {
#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaFree wrapper called (src, device) ->";
  std::cout << "(" << src << "," << device << ")\n";
 #endif
  cudaError err;

  CudaSetDevice(device);
  if (cudaSuccess != (err = cudaFree(src))) {
    std::ostringstream msg;
    msg << "CudaFree failed, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - " << cudaGetErrorString(err);
    throw(std::runtime_error(msg.str()));
  }

#ifdef DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaFree wrapper exiting \n";
#endif
}

void CudaFreeHost(void* src, const unsigned int device) {
#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaFreeHost wrapper called (src, device) ->";
  std::cout << "(" << src << "," << device << ")\n";
#endif
  cudaError err;

  CudaSetDevice(device);
  if (cudaSuccess != (err = cudaFreeHost(src))) {
    std::ostringstream msg;
    msg << "CudaFreeHost failed, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - " << cudaGetErrorString(err);
    throw(std::runtime_error(msg.str()));
  }

#ifdef DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaFreeHost wrapper exiting \n";
#endif
}

/*---------------------------------------------------------------------*/

void CudaMemcpy(void* dest, const void* src, const size_t sz, const unsigned int device,
                const cudaMemcpyKind cmkk, cudaStream_t stream) {
#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMemcpy wrapper called (src,dest,size,device,type,stream) -> ( ";
  std::cout << src << "," << dest << "," << sz << "," << device << "," << cmkk << "," << stream << " )" <<  std::endl;
#endif
  cudaError err;

  CudaSetDevice(device);

  if(stream == NULL){
#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMemcpy wrapper called  (src,dest,type,stream) -> ( ";
  std::cout << src << ", " << dest << ", " << cmkk << " ," << stream << ")" << std::endl;
#endif
   if (cudaSuccess != (err = cudaMemcpy(dest, src, sz, cmkk)) ) {
      std::ostringstream msg;
      msg << "Cuda memcopy (Blocking) failed, at" << __FILE__ << " : " << __LINE__ << std::endl;
      msg << "\t - " << cudaGetErrorString(err);
      throw(std::runtime_error(msg.str()));
   }
  } else{
#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMemcpyAsync wrapper called  (src,dest,type,stream) -> ( ";
  std::cout << src << ", " << dest << ", " << cmkk << " ," << stream << ")" << std::endl;
#endif
   if (cudaSuccess != (err = cudaMemcpyAsync(dest, src, sz, cmkk, stream)) ) {
      std::ostringstream msg;
      msg << "Cuda memcopy (Non-Blocking) failed, at" << __FILE__ << " : " << __LINE__ << std::endl;
      msg << "\t - " << cudaGetErrorString(err);
      throw(std::runtime_error(msg.str()));
   }
#ifndef NDEBUG
   CudaStreamSync(stream);
#endif
  }

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMemcpy wrapper exiting (src,dest,size,device,type,stream) -> ( ";
  std::cout << src << "," << dest << "," << sz << "," << device << "," << cmkk << "," << stream << " )" << std::endl;
#endif
}

/*---------------------------------------------------------------------*/

void CudaStreamSync( const cudaStream_t stream ) {
#ifdef DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaStreamSync wrapper called (stream) : " << stream << std::endl;
#endif
   //Todo Set context for multi-GPUs
   cudaError err;
   if (cudaErrorInvalidResourceHandle == (err = cudaStreamSynchronize(stream)) ) {
      std::ostringstream msg;
      msg << "CudaStreamSynchronize failed - invalid stream, at" << __FILE__ << " : " << __LINE__ << std::endl;
      msg << "\t - " << cudaGetErrorString(err);
      throw(std::runtime_error(msg.str()));
   }
   else if(cudaSuccess != err) {
      std::ostringstream msg;
      msg << "CudaStreamSynchronize failed, at" << __FILE__ << " : " << __LINE__ << std::endl;
      msg << "\t - " << cudaGetErrorString(err);
      throw(std::runtime_error(msg.str()));
   }
#ifdef DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaStreamSync wrapper exiting \n";
#endif
}

/*---------------------------------------------------------------------*/

void CudaMemcpyPeer(void* dest, const int dID, const void* src, const int sID, const size_t sz) {
#ifdef DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMemcpyPeer wrapper called (dest, dID, src, sID, size)-> ( ";
  std::cout << dest << "," << dID << "," << src << "," << sID << "," << sz << " )" << std::endl;
#endif
  cudaError err;

  if (cudaSuccess != (err = cudaMemcpyPeer(dest, dID, src, sID, sz))) {
    std::ostringstream msg;
    msg << "CudaMemcpyPeer failed, at" << __FILE__ << " : " << __LINE__ << std::endl;
    msg << "\t - " << cudaGetErrorString(err);
    throw(std::runtime_error(msg.str()));
  }

#ifdef DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMemcpyPeer wrapper exiting\n";
#endif
}

/*---------------------------------------------------------------------*/

void CudaMemset(void* dest, const int value, const size_t sz, const unsigned int device) {
#ifdef DEBUG_EXT_SET_MEM
  std::cout << "CudaMemset wrapper called (dest, value, size, device)-> ( ";
  std::cout << dest << "," << value << "," << sz << "," << device << " )" << std::endl;
#endif
  cudaError err;

  CudaSetDevice(device);
  if (cudaSuccess != (err = cudaMemset(dest, value, sz))) {
    std::ostringstream msg;
    msg << "Memset failed, at" << __FILE__ << " : " << __LINE__ << std::endl;
    msg << "\t - " << cudaGetErrorString(err);
    throw(std::runtime_error(msg.str()));
  }

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CudaMemset wrapper exiting\n";
#endif
}

/******************* CUDADeviceManager Implementation ******************/

CUDADeviceManager& CUDADeviceManager::self() {
  static CUDADeviceManager CDM;

  return CDM;
}

/*---------------------------------------------------------------------*/

int CUDADeviceManager::get_device_count() const {
  return device_count;
}

int CUDADeviceManager::get_best_device() const {
  int max_SMprocessors = 0, max_device = 0;
  if (device_count > 1) { // multiple GPUs
    for(int device = 0; device < device_count; device++) {
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, device);
      if (max_SMprocessors < properties.multiProcessorCount) {
        max_SMprocessors = properties.multiProcessorCount;
        max_device = device;
      }
    }
   return max_device;
  }
  else { // single GPU
    return max_device;
  }
}

/*---------------------------------------------------------------------*/

void CUDADeviceManager::get_memory_statistics(CUDAMemStats& cms, const int K) const {
  if (K < 0 || K >= device_count) {
    throw std::range_error("Specified device index out of bounds");
  }

  cms.f = device_stats[K]->f;
  cms.t = device_stats[K]->t;
  cms.inuse = device_stats[K]->inuse;
}

/*---------------------------------------------------------------------*/

void CUDADeviceManager::update_memory_statistics() {
#ifdef  DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
  std::cout << "CUDADeviceManager::update_memory_statistics -> called\n";
#endif
  for (int device = 0; device < device_count; ++device) {
    CUDAMemStats& cms = *device_stats[device];
    cudaMemGetInfo(&cms.f, &cms.t);
    cms.inuse = cms.t - cms.f;
  }
}

/*---------------------------------------------------------------------*/

void CUDADeviceManager::print_device_info() const {
  int device = 0;

  for (std::vector<cudaDeviceProp*>::const_iterator cdpit =
      device_props.begin(); cdpit != device_props.end(); ++cdpit) {

    cudaDeviceProp& p = *(*cdpit);

    std::cout << "\n\t[" << device << "] " << p.name << " ("
        << (p.integrated ? "Integrated" : "Discrete") << ")";
    std::cout << ", Compute Capability: " << p.major << "." << p.minor
        << std::endl;
    std::cout << "\n\t Multi-Processors: " << p.multiProcessorCount;
    std::cout << "\n\t  - Clock Rate: " << p.clockRate / 1000 << " MHz";
    std::cout << "\n\t  - Max Threads Per Dim: " << p.maxThreadsDim[0] << " x "
        << p.maxThreadsDim[1] << " x " << p.maxThreadsDim[2];
    std::cout << "\n\t  - Max Grid Size: " << p.maxGridSize[0] << " x "
        << p.maxGridSize[1] << " x " << p.maxGridSize[2];
    std::cout << "\n\t  - Warp Size: " << p.warpSize;
    std::cout << std::endl;
    std::cout << "\n\t Memory";
    std::cout << "\n\t  - Global: " << p.totalGlobalMem / 1000000 << " MB";
    std::cout << "\n\t  - Const: " << p.totalConstMem / 1000 << " KB";
    std::cout << "\n\t  - Shared ( Per Block ): " << p.sharedMemPerBlock / 1000
        << " KB";
    std::cout << "\n\t  - Registers Per Block (32 bit): " << p.regsPerBlock;
    std::cout << std::endl << std::endl;

    device++;
  }
}

/*---------------------------------------------------------------------*/

CUDADeviceManager::~CUDADeviceManager() {
#ifdef DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
  std::cout << "CUDADeviceManager::~CUDADeviceManager() called" << std::endl;
#endif
  // Free device property structures
  for (std::vector<cudaDeviceProp*>::const_iterator cdpit =
      device_props.begin(); cdpit != device_props.end(); ++cdpit) {
    delete (*cdpit);
  }
  for (std::vector<CUDAMemStats*>::const_iterator cmsit = device_stats.begin();
      cmsit != device_stats.end(); ++cmsit) {
    delete (*cmsit);
  }
#ifdef DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
  std::cout << "CUDADeviceManager::~CUDADeviceManager() finished" << std::endl;
#endif
}

/*---------------------------------------------------------------------*/

CUDADeviceManager::CUDADeviceManager() {
  //Gather system parameters
  cudaGetDeviceCount(&device_count);

#ifdef DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
  std::cout << "Initializing CUDADeviceManager, found " << device_count << " devices." << std::endl;
#endif

  for (int device = 0; device < device_count; ++device) {
#ifdef DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
    std::cout << "Processing device (" << device << ")\n";
#endif
    device_props.push_back(new cudaDeviceProp());
    device_stats.push_back(new CUDAMemStats());
    cudaGetDeviceProperties(device_props.back(), device);
  }

#ifdef DEBUG_EXT_ALLOC_CUDA_DEVICE_MNGR
  print_device_info();
#endif
}

/******************* CUDADeviceInterface Implementation *********************/
CUDADeviceInterface::~CUDADeviceInterface() {

}

CUDADeviceInterface::CUDADeviceInterface() {

}

CUDADeviceInterface& CUDADeviceInterface::self() {
  static CUDADeviceInterface CDI;

  return CDI;
}

/*---------------------------------------------------------------------*/

int CUDADeviceInterface::get_device_count() const {
  return CUDADeviceManager::self().get_device_count();
}

int CUDADeviceInterface::get_best_device() const {
  return CUDADeviceManager::self().get_best_device();
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::get_memory_statistics(CUDAMemStats& cms, int K) const {
  CUDADeviceManager::self().get_memory_statistics(cms, K);
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::update_memory_statistics() {
  CUDADeviceManager::self().update_memory_statistics();
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::print_device_info() const {
  CUDADeviceManager::self().print_device_info();
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::sync_stream(const cudaStream_t stream) {
  CudaStreamSync(stream);
}

/*---------------------------------------------------------------------*/

CUDASharedPointer CUDADeviceInterface::get_shared_pointer(unsigned long int N,
    unsigned int K) {
  void* devptr;

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CUDADeviceInterface::get_shared_pointer -> Allocating new shared pointer, " << N << " bytes on device " << K << std::endl;
#endif
  if (K > CUDADeviceManager::self().device_count) {
    std::ostringstream msg;
    msg << "CudaMalloc failed, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - Invalid device index '" << K << "'";
    throw(std::range_error(msg.str()));
  }

  CudaMalloc(&devptr, N, K);

  return CUDASharedPointer(devptr, K);
}

/*---------------------------------------------------------------------*/

void* CUDADeviceInterface::get_raw_pointer(unsigned long int N,
    unsigned int K) {
  void* devptr; //device pointer declaration

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CUDADeviceInterface::get_raw_pointer -> Allocating new raw pointer, " << N << " bytes on device " << K << std::endl;
#endif
  if (K >= CUDADeviceManager::self().device_count) {
    std::ostringstream msg;
    msg << "Failed call to get_raw_pointer, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - Invalid device index '" << K << "'";
    throw(std::range_error(msg.str()));
  }

  CudaMalloc(&devptr, N, K);
  return devptr;
}

void* CUDADeviceInterface::get_pinned_pointer(unsigned long int N,
                                              unsigned int K) {
  void* hostptr; //CPU Pinned pointer declaration

#ifdef  DEBUG_EXT_ALLOC_MEM
  std::cout << "CUDADeviceInterface::get_pinned_pointer -> Allocating new pinned pointer, " << N << " bytes on device context :" << K << std::endl;
#endif
  if (K >= CUDADeviceManager::self().device_count) {
    std::ostringstream msg;
    msg << "Failed call to get_pinned_pointer, at " << __FILE__ << " : " << __LINE__
        << std::endl;
    msg << "\t - Invalid device context '" << K << "'";
    throw(std::range_error(msg.str()));
  }

  CudaHostAlloc(&hostptr, N, K);
  return hostptr;
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::memcpy_to(CUDASharedPointer& dest, const void* src,
    const size_t sz, cudaStream_t stream) {
  CudaMemcpy(dest.ptr_, src, sz, (*dest.deviceID_), cudaMemcpyHostToDevice, stream);
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::memcpy_to(void* dest, const void* src, const size_t sz,
   const unsigned int deviceID, cudaStream_t stream) {
  CudaMemcpy(dest, src, sz, deviceID, cudaMemcpyHostToDevice, stream);
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::memcpy_from(void* dest, const CUDASharedPointer& src,
    const size_t sz, cudaStream_t stream) {
  CudaMemcpy(dest, src.ptr_, sz, (*src.deviceID_), cudaMemcpyDeviceToHost, stream);
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::memcpy_from(void* dest, const void* src, const size_t sz,
    const unsigned int deviceID, cudaStream_t stream) {
	CudaMemcpy(dest, src, sz, deviceID, cudaMemcpyDeviceToHost, stream);
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::memcpy_peer(void* dest, const int dID, const void* src,
		const int sID, const size_t sz) {
  CudaMemcpyPeer(dest, dID, src, sID, sz );
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::memset(void* dest, const int value, const size_t sz,
    const unsigned int deviceID) {
  CudaMemset(dest, value, sz, deviceID);
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::memset(CUDASharedPointer& dest, const int value, const size_t sz) {
  CudaMemset(dest.ptr_, value, sz, (*dest.deviceID_));
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::release(CUDASharedPointer& x) {
  x.detach();
}

/*---------------------------------------------------------------------*/

void CUDADeviceInterface::release(void* x, const unsigned int deviceID) {
  CudaFree(x, deviceID);
}

void CUDADeviceInterface::releaseHost(void* x, const unsigned int deviceID) {
  CudaFreeHost(x, deviceID);
}

/******************* CUDASharedPointer Implementation *********************/
CUDASharedPointer::~CUDASharedPointer() {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
  std::cout << "CUDASharedPointer::~CUDASharedPointer called\n\n";
#endif
  detach();
}

/*---------------------------------------------------------------------*/

CUDASharedPointer::CUDASharedPointer() :
    ptr_(NULL), refCount_(NULL), deviceID_(NULL) {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
  std::cout << "CUDASharedPointer::CUDASharedPointer called\n\n";
#endif
  /** Free floating pointer **/
}

/*---------------------------------------------------------------------*/

CUDASharedPointer::CUDASharedPointer(void* ptr, const int K) :
    ptr_(ptr), refCount_(new int(1)), deviceID_(new int(K)) {
  /** Pointer initialized to wrap a normal void pointer **/
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
  std::cout << "CUDASharedPointer::CUDASharedPointer(void* ptr, int K), called ->";
  std::cout << "(" << ptr << "," << K << ")\n\n";
#endif
}

CUDASharedPointer::CUDASharedPointer(const CUDASharedPointer& x) :
    ptr_(NULL), refCount_(NULL), deviceID_(NULL) {
  (*this) = x;
}

/*---------------------------------------------------------------------*/

CUDASharedPointer& CUDASharedPointer::operator=(void* x) {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
  std::cout << "CUDASharedPointer assigned to void*";
#endif
  if (refCount_ != NULL) {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
    std::cout << "\n\t - Non-zero ref count, decrementing";
#endif
    --(*refCount_);
    if ((*refCount_) == 0) {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
      std::cout << "\n\t - Pointer ref count is zero, freeing host and device memory";
#endif
      delete refCount_;
      refCount_ = NULL;

      CudaFree(ptr_, *deviceID_);
      ptr_ = NULL;

      delete deviceID_;
      deviceID_ = NULL;
    }
  }

  ptr_ = x;
  refCount_ = new int(1);
  deviceID_ = new int();
  cudaGetDevice(deviceID_);

#ifdef DEBUG_EXT_CUDA_SHARED_PTR
  std::cout << "\nCUDASharedPointer returning successfully\n\n";
#endif
  return *this;
}

/*---------------------------------------------------------------------*/

bool CUDASharedPointer::operator==(const CUDASharedPointer& x) const {
  if ((this->ptr_ == x.ptr_) && (this->deviceID_ == x.deviceID_)) {
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------*/

CUDASharedPointer& CUDASharedPointer::operator=(const CUDASharedPointer& x) {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
  std::cout << "CUDASharedPointer assigned to CUDASharedPointer reference";
#endif

  /** Check to make sure we're not self assigning. **/
  if ((*this) == x) { return *this; }

  if (refCount_ != NULL) {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
    std::cout << "\n\t - Non-zero ref count, decrementing";
#endif
    --(*refCount_);
    if ((*refCount_) == 0) {
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
      std::cout << "\n\t - Pointer ref count is zero, freeing host and device memory";
#endif
      delete refCount_;
      refCount_ = NULL;

      CudaFree(ptr_, *deviceID_);
      ptr_ = NULL;

      delete deviceID_;
      deviceID_ = NULL;
    }
  }

  ptr_ = x.ptr_;
  deviceID_ = x.deviceID_;
  refCount_ = x.refCount_;
  ++(*refCount_);
#ifdef DEBUG_EXT_CUDA_SHARED_PTR
  std::cout << "\nCUDASharedPointer returning successfully\n\n";
#endif
  return *this;
}

/*---------------------------------------------------------------------*/

void* CUDASharedPointer::operator->() {
  return ptr_;
}

/*---------------------------------------------------------------------*/

const void* CUDASharedPointer::operator->() const {
  return ptr_;
}

/*---------------------------------------------------------------------*/

bool CUDASharedPointer::isnull() const {
  return (ptr_ == NULL);
}

/*---------------------------------------------------------------------*/

int CUDASharedPointer::get_refcount() const {
  return *refCount_;
}

/*---------------------------------------------------------------------*/

int CUDASharedPointer::get_deviceID() const {
  return *deviceID_;
}

/*---------------------------------------------------------------------*/

CUDASharedPointer& CUDASharedPointer::detach() {
  (*this) = ((void*) NULL);

  return *this;
}

} // End namespace - ema::cuda
} // End namespace - ema
