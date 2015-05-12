/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

// GPU Grid Variable: in host & device code (HOST_DEVICE == __host__ __device__)

#ifndef UINTAH_GRID_VARIABLES_GPUGRIDVARIABLE_H
#define UINTAH_GRID_VARIABLES_GPUGRIDVARIABLE_H

#include <Core/Grid/Variables/GPUGridVariableBase.h>
#include <sci_defs/cuda_defs.h>

namespace Uintah {
  
  template<class T> class GPUArray3 {

    public:

      HOST_DEVICE virtual ~GPUArray3(){};

      HOST_DEVICE const T& operator[](const int3& idx) const
      {  //get data from global index
        CHECK_INSIDE(idx, d_offset, d_size)
        return d_data[idx.x - d_offset.x + d_size.x * (idx.y - d_offset.y + (idx.z - d_offset.z) * d_size.y)];
      }

      HOST_DEVICE T& operator[](const int3& idx)
      {  //get data from global index
        CHECK_INSIDE(idx, d_offset, d_size)
        return d_data[idx.x - d_offset.x + d_size.x * (idx.y - d_offset.y + (idx.z - d_offset.z) * d_size.y)];
      }

      HOST_DEVICE const T&
      operator()(const int& x, const int& y, const int& z) const
      {  //get data from global index
        CHECK_INSIDE3(x, y, z, d_offset, d_size)
        return d_data[x - d_offset.x + d_size.x * (y - d_offset.y + (z - d_offset.z) * d_size.y)];
      }

      HOST_DEVICE T& operator()(const int& x, const int& y, const int& z)
      {  //get data from global index
        CHECK_INSIDE3(x, y, z, d_offset, d_size)
        return d_data[x - d_offset.x + d_size.x * (y - d_offset.y + (z - d_offset.z) * d_size.y)];
      }

      HOST_DEVICE T* getPointer() const
      {
        return d_data;
      }

      HOST_DEVICE size_t getMemSize() const
      {
        return d_size.x * d_size.y * d_size.z * sizeof(T);
      }
      
      HOST_DEVICE int3 getLowIndex() const
      {
        return make_int3(d_offset.x, d_offset.y, d_offset.z);
      }
      HOST_DEVICE int3 getHighIndex() const
      {
        return make_int3(d_offset.x+d_size.x, d_offset.y+d_size.y, d_offset.z+d_size.z);
      }
      
       HOST_DEVICE int3 getLowIndex()
      {
        return make_int3(d_offset.x, d_offset.y, d_offset.z);
      }
      HOST_DEVICE int3 getHighIndex()
      {
        return make_int3(d_offset.x+d_size.x, d_offset.y+d_size.y, d_offset.z+d_size.z);
      }

    protected:

      HOST_DEVICE GPUArray3() {};

      HOST_DEVICE void setOffsetSizePtr(const int3& offset, const int3& size, void* &ptr) const
      {
        d_offset = offset;
        d_size = size;
        d_data = (T*)ptr;
      }

      HOST_DEVICE void getOffsetSizePtr(int3& offset, int3& size, void* &ptr) const
      {
        offset = d_offset;
        size = d_size;
        ptr = (void*)d_data;
      }

    private:

      //---------------------------------------------------------------
      // global high = d_offset+d_data
      // global low  = d_offset
      //---------------------------------------------------------------
      mutable T*    d_data;
      mutable int3  d_offset;  //offset from global index to local index
      mutable int3  d_size;    //size of local storage 

      HOST_DEVICE GPUArray3& operator=(const GPUArray3&);
      HOST_DEVICE GPUArray3(const GPUArray3&);
  };

  template<class T> class GPUGridVariable: public GPUGridVariableBase, public GPUArray3<T> {

      friend class UnifiedScheduler;  // allow Scheduler access

    public:

      HOST_DEVICE GPUGridVariable() {}
      HOST_DEVICE virtual ~GPUGridVariable() {}

      HOST_DEVICE virtual size_t getMemSize()
      {
        return GPUArray3<T>::getMemSize();
      }
      
      HOST_DEVICE virtual int3 getLowIndex()
      {
        return GPUArray3<T>::getLowIndex();
      }
      
      HOST_DEVICE virtual int3 getHighIndex()
      {
        return GPUArray3<T>::getHighIndex();
      }
      HOST_DEVICE virtual int3 getLowIndex() const
      {
        return GPUArray3<T>::getLowIndex();
      }
      
      HOST_DEVICE virtual int3 getHighIndex() const
      {
        return GPUArray3<T>::getHighIndex();
      }
    

    private:

      HOST_DEVICE virtual void getArray3(int3& offset, int3& size, void* &ptr) const
      {
        GPUArray3<T>::getOffsetSizePtr(offset, size, ptr);
      }

      HOST_DEVICE virtual void setArray3(const int3& offset, const int3& size, void* &ptr) const
      {
        GPUArray3<T>::setOffsetSizePtr(offset, size, ptr);
      }
  };

} // end namespace Uintah

#endif // UINTAH_GRID_VARIABLES_GPUGRIDVARIABLE_H
