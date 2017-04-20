/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

// GPU ReductionVariable: in host & device code (HOST_DEVICE == __host__ __device__)

#ifndef UINTAH_CORE_GRID_VARIABLES_GPUREDUCTIONVARIABLE_H
#define UINTAH_CORE_GRID_VARIABLES_GPUREDUCTIONVARIABLE_H

#include <Core/Grid/Variables/GPUReductionVariableBase.h>
#include <sci_defs/cuda_defs.h>

#include <string>

namespace Uintah {

template<class T>
class GPUReductionVariable : public GPUReductionVariableBase {

  friend class UnifiedScheduler; // allow Scheduler access

  public:

    HOST_DEVICE GPUReductionVariable()
    {
      d_data = nullptr;
    }

    HOST_DEVICE virtual ~GPUReductionVariable() {}

    HOST_DEVICE virtual size_t getMemSize()
    {
      return sizeof(T);
    }

    HOST_DEVICE T* getPointer() const
    {
      return d_data;
    }

    HOST_DEVICE void* getVoidPointer() const
	{
	  return d_data;
	}


    HOST_DEVICE void getSizeInfo(std::string& elems, unsigned long& totsize, void* &ptr) const
    {
      elems = "1";
      totsize = sizeof(T);
      ptr = (void*)&d_data;
    }

  private:

    mutable T* d_data;

    HOST_DEVICE virtual void getData(void* &ptr) const
    {
        ptr = (void*)d_data;
    }

    HOST_DEVICE virtual void setData(void* &ptr) const
    {
      d_data = (T*)ptr;
    }

    HOST_DEVICE GPUReductionVariable& operator=(const GPUReductionVariable&);
    HOST_DEVICE GPUReductionVariable(const GPUReductionVariable&);
};

}  // end namespace Uintah

#endif // UINTAH_CORE_GRID_VARIABLES_GPUREDUCTIONVARIABLE_H
