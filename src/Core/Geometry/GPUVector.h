/*
 * The MIT License
 *
 * Copyright (c) 2013-2026 The University of Utah
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

#ifndef CORE_GEOMETRY_GPUVECTOR_H
#define CORE_GEOMETRY_GPUVECTOR_H

#include <sci_defs/gpu_defs.h>

namespace Uintah {

class gpuIntVector : public int3 {
  public:
    GPU_INLINE_FUNCTION gpuIntVector() {}
    GPU_INLINE_FUNCTION int& operator[](const int& i) { return (&x)[i]; }
    GPU_INLINE_FUNCTION int& operator[](      int& i) { return (&x)[i]; }
    
    GPU_INLINE_FUNCTION const int& operator[](const int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION const int& operator[](      int& i) const { return (&x)[i]; }
    
    GPU_INLINE_FUNCTION gpuIntVector(const int3& copy):int3(copy) {}
};

//______________________________________________________________________
//
class uInt3 : public uint3 {

  public:
    GPU_INLINE_FUNCTION uInt3() {}
    GPU_INLINE_FUNCTION unsigned int& operator[](const int& i) { return (&x)[i]; }
    GPU_INLINE_FUNCTION unsigned int& operator[](      int& i) { return (&x)[i]; }
    
    GPU_INLINE_FUNCTION const unsigned int& operator[](const int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION const unsigned int& operator[](      int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION uInt3(const uint3& copy):uint3(copy) {}
};

//______________________________________________________________________
//
class Float3 : public float3 {

  public:
    GPU_INLINE_FUNCTION Float3() {}
    GPU_INLINE_FUNCTION float& operator[](const int& i) { return (&x)[i]; }
    GPU_INLINE_FUNCTION float& operator[](      int& i) { return (&x)[i]; }
    
    GPU_INLINE_FUNCTION const float& operator[](const int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION const float& operator[](      int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION Float3(const float3& copy):float3(copy) {}
};

//______________________________________________________________________
//
class gpuVector : public double3 {
  public:
    GPU_INLINE_FUNCTION gpuVector() {}
    GPU_INLINE_FUNCTION double& operator[](const int &i) { return (&x)[i]; }
    GPU_INLINE_FUNCTION double& operator[](      int &i) { return (&x)[i]; }
    
    GPU_INLINE_FUNCTION const double& operator[](const int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION const double& operator[](      int& i) const { return (&x)[i]; }

    GPU_INLINE_FUNCTION gpuVector(const double3& copy) : double3(copy) {}
};

//______________________________________________________________________
//
class gpuPoint : public double3 {

  public:
    GPU_INLINE_FUNCTION gpuPoint() {}
    GPU_INLINE_FUNCTION double& operator[](const int& i) { return (&x)[i]; }
    GPU_INLINE_FUNCTION double& operator[](      int& i) { return (&x)[i]; }
    
    GPU_INLINE_FUNCTION const double& operator[](const int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION const double& operator[](      int& i) const { return (&x)[i]; }
    GPU_INLINE_FUNCTION gpuPoint(const double3& copy) : double3(copy) {}
};

} // end namespace Uintah

#endif // end #ifndef CORE_GEOMETRY_GPUVECTOR_H
