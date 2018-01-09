/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

// GPU PerPatch base class: in host & device code (HOST_DEVICE == __host__ __device__)

#ifndef UINTAH_CORE_GRID_VARIABLES_GPUPERPATCH_H
#define UINTAH_CORE_GRID_VARIABLES_GPUPERPATCH_H

#include <Core/Grid/Variables/GPUPerPatchBase.h>
#include <sci_defs/cuda_defs.h>

#include <string>

namespace Uintah {

template<class T>
class GPUPerPatch : public GPUPerPatchBase {

  friend class UnifiedScheduler; // allow UnifiedScheduler access

  public:

    HOST_DEVICE GPUPerPatch()
    {
      d_value = 0;
    }

    HOST_DEVICE GPUPerPatch(T value)
          : d_value(value)
    {
    }

    HOST_DEVICE virtual ~GPUPerPatch() {};

    HOST_DEVICE virtual size_t getMemSize()
    {
      return sizeof(T);
    }

    HOST_DEVICE T* getPointer() const
    {
      return d_value;
    }

    HOST_DEVICE void* getVoidPointer() const {
      return d_value;
    }

    HOST_DEVICE void getSizeInfo(std::string& elems, unsigned long& totsize, void* &ptr) const
    {
      elems = "1";
      totsize = sizeof(T);
      ptr = (void*)&d_value;
    }


  private:

    mutable T* d_value;

    HOST_DEVICE virtual void getData(void* &ptr) const
    {
        ptr = (void*)d_value;
    }

    HOST_DEVICE virtual void setData(void* &ptr) const
    {
      d_value = (T*)ptr;
    }

    HOST_DEVICE GPUPerPatch<T>& operator=(const GPUPerPatch<T>& copy);
    HOST_DEVICE GPUPerPatch(const GPUPerPatch&);
};


}  // End namespace Uintah

#endif // UINTAH_CORE_GRID_VARIABLES_GPUPERPATCH_H
