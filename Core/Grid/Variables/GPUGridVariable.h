/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

/* GPU Grid Variable in host&deivce code*/
#ifndef UINTAH_HOMEBREW_GPUGRIDVAR_H
#define UINTAH_HOMEBREW_GPUGRIDVAR_H

#include <sci_defs/cuda_defs.h>

namespace Uintah {
  
  template<class T> class GPUArray3 {
    public:
      HOST_DEVICE virtual ~GPUArray3(){};
      HOST_DEVICE const T& operator[](const int3& idx) const{ //get data from global index
        CHECK_INSIDE(idx,d_offset, d_size)
        return d_data[ idx.x-d_offset.x + d_size.x*(idx.y-d_offset.y + (idx.z-d_offset.z)*d_size.y)];
      }
      HOST_DEVICE T& operator[](const int3& idx){ //get data from global index
        CHECK_INSIDE(idx,d_offset, d_size)
        return d_data[ idx.x-d_offset.x + d_size.x*(idx.y-d_offset.y + (idx.z-d_offset.z)*d_size.y)];
      }
      HOST_DEVICE T*  getPointer() const{
        return d_data;
      }
      HOST_DEVICE size_t getMemSize() const {
        return d_size.x*d_size.y*d_size.z*sizeof(T);
      }
      HOST_DEVICE void setOffsetSizePtr(const int3& offset, const int3& size, void* &ptr){
        d_offset = offset;
        d_size = size;
        d_data = (T*) ptr;
      }
      HOST_DEVICE void getOffsetSizePtr(int3& offset, int3& size, void* &ptr){
        offset = d_offset;
        size = d_size;
        ptr = (void*) d_data;
      }
    protected:
      HOST_DEVICE GPUArray3(){};
    private:
      T*    d_data;
      
      int3  d_offset;  //offset from gobal index to local index
      int3  d_size;    //size of local storage 
      /* global high=d_offset+d_data 
         global low =d_offset */

      HOST_DEVICE GPUArray3& operator=(const GPUArray3&);
      HOST_DEVICE GPUArray3(const GPUArray3&);
  };

  class GPUGridVariableBase {
    friend class GPUDataWarehouse;
    public:
      HOST_DEVICE ~GPUGridVariableBase() {}
      HOST_DEVICE  virtual size_t getMemSize() = 0;
    protected:
      HOST_DEVICE GPUGridVariableBase() {}
    private:
      HOST_DEVICE  virtual void setArray3(const int3& offset, const int3& size, void* &ptr) = 0;
      HOST_DEVICE  virtual void getArray3(int3& offset, int3& size, void* &ptr) = 0;
      HOST_DEVICE GPUGridVariableBase& operator=(const GPUGridVariableBase&);
      HOST_DEVICE GPUGridVariableBase(const GPUGridVariableBase&);
  };

  template<class T> class GPUGridVariable: public GPUGridVariableBase, public GPUArray3<T> {
    public:
      HOST_DEVICE GPUGridVariable() {}
      HOST_DEVICE virtual ~GPUGridVariable() {}
      HOST_DEVICE virtual size_t getMemSize() {
        return GPUArray3<T>::getMemSize();
      }
    private:
      HOST_DEVICE virtual void setArray3(const int3& offset, const int3& size, void* &ptr) {
        GPUArray3<T>::setOffsetSizePtr(offset, size, ptr);
      }
      HOST_DEVICE virtual void getArray3(int3& offset, int3& size, void* &ptr) {
        GPUArray3<T>::getOffsetSizePtr(offset, size, ptr);
      }
  };

} // End namespace Uintah

#endif
