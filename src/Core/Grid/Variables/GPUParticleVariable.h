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

// GPU ReductionVariable: in host & device code (HOST_DEVICE == __host__ __device__)

#ifndef UINTAH_GPUPARTICLEVARIABLE_H
#define UINTAH_GPUPARTICLEVARIABLE_H

#include <Core/Grid/Variables/GPUParticleVariableBase.h>
#include <sci_defs/cuda_defs.h>

namespace Uintah {

template<class T>
class GPUParticleVariable : public GPUParticleVariableBase {

  friend class UnifiedScheduler; // allow Scheduler access

  public:

    HOST_DEVICE GPUParticleVariable() {d_data = nullptr; d_size = 0;}
    HOST_DEVICE virtual ~GPUParticleVariable() {}

    HOST_DEVICE virtual size_t getMemSize() {
      return (d_size * sizeof(T));
    }

    HOST_DEVICE T* getPointer() const {
      return d_data;
    }

    HOST_DEVICE void* getVoidPointer() const {
      return d_data;
    }

  private:

    mutable T* d_data;
    mutable size_t d_size;

    HOST_DEVICE virtual void setData(const size_t& size, void* &ptr) const {
      d_data = (T*)ptr;
      d_size = size;
    }

    HOST_DEVICE virtual void getData(size_t& size, void* &ptr) const {
      ptr = (void*)d_data;
      size = d_size;
    }

    HOST_DEVICE GPUParticleVariable& operator=(const GPUParticleVariable&);
    HOST_DEVICE GPUParticleVariable(const GPUParticleVariable&);
};

}  // end namespace Uintah

#endif
