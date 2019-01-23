/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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

// This class is crucial for Uintah portability.   It contains additional task parameters to help
// aid execution of this task on different architectures.  At the time of this file's creation a
// Detailed Task supporting portabiltiy consisted of the following:
//                        const PatchSubset* patches,
//                        const MaterialSubset* matls,
//                        DataWarehouse* fromDW,
//                        DataWarehouse* toDW,
//                        UintahParams & uintahParams,
//                        ExecutionObject & execObj
// For Cuda related tasks in particular, this object wraps a Cuda Stream.  Further, this task
// wraps the command line execution arguments given so that a particular task can modify
// these arguments further.
// From the application developer's perpsective, the execObj received should be
// passed into a Uintah parallel loop from LoopExecution.hpp.  The application developer does not
// need to modify this object at all, and likely most of the time won't need to.


#ifndef EXECUTIONOBJECT_H_
#define EXECUTIONOBJECT_H_
#include <Core/Exceptions/InternalError.h>

#include <sci_defs/cuda_defs.h>

#include <vector>

namespace Uintah {

class UintahParams;

template <typename ExecSpace, typename MemSpace>
class ExecutionObject {
public:

  // Streams should be created, supplied, and managed by the scheduler itself.
  // The application developer probably shouldn't be managing his or her own streams.
  void setStream(void* stream, int deviceID){
#if defined(HAVE_CUDA)
    //Ignore the non-CUDA case as those streams are pointless.
    m_streams.push_back(stream);
    this->deviceID = deviceID;
#endif
  }

  void setStreams(const std::vector<void*>& streams, int deviceID){
#if defined(HAVE_CUDA)
  for (auto& stream : streams) {
    m_streams.push_back(stream);
  }
  this->deviceID = deviceID;
#endif //Ignore the non-CUDA case as those streams are pointless.
  }

  void * getStream() const {
    if ( m_streams.size() == 0 ) {
      return nullptr;
    } else {
      return m_streams[0];
    }
  }
  void * getStream(unsigned int i) const {
    if ( i >= m_streams.size() ) {
      SCI_THROW(InternalError("Requested a stream that doesn't exist.", __FILE__, __LINE__));
    } else {
      return m_streams[i];
    }
  }

  unsigned int getNumStreams() const {
    return m_streams.size();
  }

  int getCudaThreadsPerBlock() const {
    return cuda_threads_per_block;
  }

  void setCudaThreadsPerBlock(int CudaThreadsPerBlock) {
    this->cuda_threads_per_block = CudaThreadsPerBlock;
  }

  int getCudaBlocksPerLoop() const {
    return cuda_blocks_per_loop;
  }

  void setCudaBlocksPerLoop(int CudaBlocksPerLoop) {
    this->cuda_blocks_per_loop = CudaBlocksPerLoop;
  }

  //void getTempTaskSpaceFromPool(void** ptr, unsigned int size) const {}
private:
  std::vector<void*> m_streams;
  int deviceID{0};
  int cuda_threads_per_block{-1};
  int cuda_blocks_per_loop{-1};
};

} // end namespace Uintah
#endif /* EXECUTIONOBJECT_H_ */
