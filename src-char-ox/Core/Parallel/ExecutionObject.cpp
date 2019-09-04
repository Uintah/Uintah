
#include <Core/Exceptions/InternalError.h>
#include <Core/Parallel/LoopExecution.hpp>
#include <vector>

#include <sci_defs/cuda_defs.h>
#include <sci_defs/kokkos_defs.h>

namespace Uintah {

// Streams should be created, supplied, and managed by the scheduler itself.
// The application developer probably shouldn't be managing his or her own streams.
void ExecutionObject::setStream(void* stream, int deviceID) {

#if defined(HAVE_CUDA)
  //Ignore the non-CUDA case as those streams are pointless.
  m_streams.push_back(stream);
  this->deviceID = deviceID;
#endif
}

void ExecutionObject::setStreams(const std::vector<void*>& streams, int deviceID) {
#if defined(HAVE_CUDA)
  for (auto& stream : streams) {
    m_streams.push_back(stream);
  }
  this->deviceID = deviceID;
#endif //Ignore the non-CUDA case as those streams are pointless.
}

void * ExecutionObject::getStream() const {
  if ( m_streams.size() == 0 ) {
    return nullptr;
  } else {
    return m_streams[0];
  }
}
void * ExecutionObject::getStream(unsigned int i) const {
  if ( i >= m_streams.size() ) {
    SCI_THROW(InternalError("Requested a stream that doesn't exist.", __FILE__, __LINE__));
  } else {
    return m_streams[i];
  }
 }

unsigned int ExecutionObject::getNumStreams() const {
  return m_streams.size();
}

int ExecutionObject::getCudaThreadsPerBlock() const {
  return cuda_threads_per_block;
}

void ExecutionObject::setCudaThreadsPerBlock(int CudaThreadsPerBlock) {
  this->cuda_threads_per_block = CudaThreadsPerBlock;
}

int ExecutionObject::getCudaBlocksPerLoop() const {
  return cuda_blocks_per_loop;
}

void ExecutionObject::setCudaBlocksPerLoop(int CudaBlocksPerLoop) {
  this->cuda_blocks_per_loop = CudaBlocksPerLoop;
}

void ExecutionObject::getTempTaskSpaceFromPool(void** ptr, unsigned int size) const {
  //TODO:
}

} //end namespace Uintah
