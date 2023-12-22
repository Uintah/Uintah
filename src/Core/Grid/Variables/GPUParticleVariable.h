/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef UINTAH_GPUPARTICLEVARIABLE_H
#define UINTAH_GPUPARTICLEVARIABLE_H

#include <Core/Grid/Variables/GPUParticleVariableBase.h>
#include <sci_defs/gpu_defs.h>

namespace Uintah {

template<class T>
class GPUParticleVariable : public GPUParticleVariableBase {

  friend class KokkosScheduler;   // allow scheduler access
  friend class UnifiedScheduler;  // allow scheduler access

  public:

    GPU_INLINE_FUNCTION GPUParticleVariable() {d_data = nullptr; d_size = 0;}
    GPU_INLINE_FUNCTION virtual ~GPUParticleVariable() {}

    GPU_INLINE_FUNCTION virtual size_t getMemSize() {
      return (d_size * sizeof(T));
    }

    GPU_INLINE_FUNCTION T* getPointer() const {
      return d_data;
    }

    GPU_INLINE_FUNCTION void* getVoidPointer() const {
      return d_data;
    }

  private:

    mutable T* d_data;
    mutable size_t d_size;

    GPU_INLINE_FUNCTION virtual void setData(const size_t& size, void* &ptr) const {
      d_data = (T*)ptr;
      d_size = size;
    }

    GPU_INLINE_FUNCTION virtual void getData(size_t& size, void* &ptr) const {
      ptr = (void*)d_data;
      size = d_size;
    }

    GPU_INLINE_FUNCTION GPUParticleVariable& operator=(const GPUParticleVariable&);
    GPU_INLINE_FUNCTION GPUParticleVariable(const GPUParticleVariable&);
};

}  // end namespace Uintah

#endif
