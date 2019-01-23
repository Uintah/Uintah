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

// This class acts as an additional Task parameter object to easy add or remove additional parameters with ease.
// At the time of this class's creation, the Task consisted of 5 or 6 parameters (legacy and Kokkos tasks)
// For Kokkos tasks, the parameters are as follows:
//                        const PatchSubset* patches,
//                        const MaterialSubset* matls,
//                        DataWarehouse* fromDW,
//                        DataWarehouse* toDW,
//                        UintahParams & uintahParams,
//                        ExecutionObject & execObj
// The first 4 parameters are rather important parameters that help a task execute.
// uintahParams may contain a collection of additional helpful items inside.


#ifndef UINTAH_HOMEBREW_UINTAH_PARAMS_HPP
#define UINTAH_HOMEBREW_UINTAH_PARAMS_HPP

#include <Core/Grid/TaskStatus.h>

#include <sci_defs/cuda_defs.h>

namespace Uintah {
//----------------------------------------------------------------------------
// Class UintahParams
//----------------------------------------------------------------------------

class DetailedTask;
class ProcessorGroup;

class UintahParams {

public:

  UintahParams() {}

  UintahParams(const UintahParams& obj) = delete;
  UintahParams(UintahParams&& obj) = delete;
  UintahParams& operator=(const UintahParams& obj) = delete;
  UintahParams& operator=(UintahParams&& obj) = delete;

  void setTaskDWs(void * oldTaskGpuDW, void * newTaskGpuDW) {
    this->oldTaskGpuDW = oldTaskGpuDW;
    this->newTaskGpuDW = newTaskGpuDW;
  }

  void setCallBackEvent(CallBackEvent callBackEvent) {
    this->callBackEvent = callBackEvent;
  }

  CallBackEvent getCallBackEvent() const {
    return callBackEvent;
  }

  const ProcessorGroup* getProcessorGroup() const {
    return processorGroup;
  }

  void setProcessorGroup(const ProcessorGroup* processorGroup) {
    this->processorGroup = processorGroup;
  }

  void setStream(void* stream){
#if defined(HAVE_CUDA)
    //Ignore the non-CUDA case as those streams are pointless.
    m_streams.push_back(stream);
#endif
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

private:

  void * oldTaskGpuDW{nullptr};
  void * newTaskGpuDW{nullptr};

  CallBackEvent callBackEvent;
  const ProcessorGroup* processorGroup {nullptr};

  std::vector<void*> m_streams;

};

} //namespace Uintah
#endif //UINTAH_HOMEBREW_UINTAH_PARAMS_HPP
