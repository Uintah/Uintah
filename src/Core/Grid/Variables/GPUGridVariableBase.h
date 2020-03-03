/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

// GPU GridVariable base class: in host & device code (HOST_DEVICE == __host__ __device__)

#ifndef UINTAH_CORE_GRID_VARIABLES_GPUGRIDVARIABLEBASE_H
#define UINTAH_CORE_GRID_VARIABLES_GPUGRIDVARIABLEBASE_H

#include <Core/Grid/Variables/GPUVariable.h>
#include <sci_defs/cuda_defs.h>

namespace Uintah {

class GPUGridVariableBase : public GPUVariable {

  friend class GPUDataWarehouse;  // allow DataWarehouse set/get data members
  friend class KokkosScheduler;   // allow scheduler access
  friend class UnifiedScheduler;  // allow scheduler access

  public:
    HOST_DEVICE virtual ~GPUGridVariableBase() {}
    HOST_DEVICE virtual size_t getMemSize() = 0;
  protected:
    HOST_DEVICE GPUGridVariableBase() {}
    HOST_DEVICE GPUGridVariableBase(const GPUGridVariableBase&);

  private:
    HOST_DEVICE virtual void getArray3(int3& offset, int3& size, void* &ptr) const = 0;
    HOST_DEVICE virtual void setArray3(const int3& offset, const int3& size, void* &ptr) const = 0;
    HOST_DEVICE GPUGridVariableBase& operator=(const GPUGridVariableBase&);
};

} // end namespace Uintah

#endif // UINTAH_CORE_GRID_VARIABLES_GPUGRIDVARIABLEBASE_H
