/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef UINTAH_GPUPARTICLEVARIABLEBASE_H
#define UINTAH_GPUPARTICLEVARIABLEBASE_H

#include <Core/Grid/Variables/GPUVariable.h>
#include <sci_defs/gpu_defs.h>

namespace Uintah {

class GPUParticleVariableBase : public GPUVariable {

  friend class GPUDataWarehouse;  // allow DataWarehouse set/get data members

  public:
    GPU_INLINE_FUNCTION virtual ~GPUParticleVariableBase() {}
    GPU_INLINE_FUNCTION virtual size_t getMemSize() = 0;

  protected:
    GPU_INLINE_FUNCTION GPUParticleVariableBase() {}
    GPU_INLINE_FUNCTION GPUParticleVariableBase(const GPUParticleVariableBase&);

  private:
    GPU_INLINE_FUNCTION virtual void setData(const size_t& size, void* &ptr) const = 0;
    GPU_INLINE_FUNCTION virtual void getData(size_t& size, void* &ptr) const = 0;
    GPU_INLINE_FUNCTION GPUParticleVariableBase& operator=(const GPUParticleVariableBase&);
};

}  // end namespace Uintah

#endif
