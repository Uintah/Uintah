/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#ifndef UINTAH_GRID_VARIABLES_GPUGRIDVARIABLE_H
#define UINTAH_GRID_VARIABLES_GPUGRIDVARIABLE_H

#include <Core/Grid/Variables/GPUGridVariableBase.h>

#include <cstdio>

namespace Uintah {
  
  template<class T> class GPUArray3 {

    public:

      GPU_INLINE_FUNCTION virtual ~GPUArray3(){};

      GPU_INLINE_FUNCTION const T& operator[](const int3& idx) const
      {  //get data from global index
#if SCI_KOKKOS_ASSERTION_LEVEL >= 3
        checkBounds( idx);
#endif
        return d_data[idx.x - d_offset.x + d_size.x * (idx.y - d_offset.y + (idx.z - d_offset.z) * d_size.y)];
      }

      GPU_INLINE_FUNCTION T& operator[](const int3& idx)
      {  //get data from global index
#if SCI_KOKKOS_ASSERTION_LEVEL >= 3
        checkBounds( idx);
#endif
        return d_data[idx.x - d_offset.x + d_size.x * (idx.y - d_offset.y + (idx.z - d_offset.z) * d_size.y)];
      }

      GPU_INLINE_FUNCTION const T&
      operator()(const int& x, const int& y, const int& z) const
      {  //get data from global index
#if SCI_KOKKOS_ASSERTION_LEVEL >= 3
        checkBounds3(x,y,z);
#endif
        return d_data[x - d_offset.x + d_size.x * (y - d_offset.y + (z - d_offset.z) * d_size.y)];
      }

      GPU_INLINE_FUNCTION T& operator()(const int& x, const int& y, const int& z)
      {  //get data from global index
#if SCI_KOKKOS_ASSERTION_LEVEL >= 3
        checkBounds3(x,y,z);
#endif
        return d_data[x - d_offset.x + d_size.x * (y - d_offset.y + (z - d_offset.z) * d_size.y)];
      }

      GPU_INLINE_FUNCTION const T& operator()(const int& x, const int& y, const int& z, const int& m) const { //get data from global index
        CHECK_INSIDE3(x,y,z,d_offset, d_size)
        return d_data[ x-d_offset.x + d_size.x*(y-d_offset.y + (z-d_offset.z)*d_size.y)];
      }

      GPU_INLINE_FUNCTION T& operator()(const int& x, const int& y, const int& z, const int& m) { //get data from global index
        CHECK_INSIDE3(x,y,z,d_offset, d_size)
        //TODO: Get materials working with the offsets.
        //return d_data[ x-d_offset.x + d_size.x*(y-d_offset.y + (z-d_offset.z)*d_size.y)];
        return d_data[ m * d_size.x * d_size.y * d_size.z + x + d_size.x*(y + (z)*d_size.y)];
      }


      GPU_INLINE_FUNCTION T* getPointer() const {
        return d_data;
      }


      GPU_INLINE_FUNCTION void copyZSliceData(const GPUArray3& copyFromVar);


      GPU_INLINE_FUNCTION size_t getMemSize() const
      {
        return d_size.x * d_size.y * d_size.z * sizeof(T);
      }
      
      GPU_INLINE_FUNCTION int3 getLowIndex() const
      {
        return make_int3(d_offset.x, d_offset.y, d_offset.z);
      }
      GPU_INLINE_FUNCTION int3 getHighIndex() const
      {
        return make_int3(d_offset.x+d_size.x, d_offset.y+d_size.y, d_offset.z+d_size.z);
      }
      
       GPU_INLINE_FUNCTION int3 getLowIndex()
      {
        return make_int3(d_offset.x, d_offset.y, d_offset.z);
      }
      GPU_INLINE_FUNCTION int3 getHighIndex()
      {
        return make_int3(d_offset.x+d_size.x, d_offset.y+d_size.y, d_offset.z+d_size.z);
      }

    protected:

      GPU_INLINE_FUNCTION GPUArray3() {};

      GPU_INLINE_FUNCTION void setOffsetSizePtr(const int3& offset, const int3& size, void* &ptr) const
      {
        d_offset = offset;
        d_size = size;
        d_data = (T*)ptr;
      }

      GPU_INLINE_FUNCTION void getOffsetSizePtr(int3& offset, int3& size, void* &ptr) const
      {
        offset = d_offset;
        size = d_size;
        ptr = (void*)d_data;
      }

      mutable T*    d_data;

    private:

      //---------------------------------------------------------------
      // global high = d_offset+d_data
      // global low  = d_offset
      //---------------------------------------------------------------
      mutable int3  d_offset;  //offset from global index to local index
      mutable int3  d_size;    //size of local storage 

      GPU_INLINE_FUNCTION GPUArray3& operator=(const GPUArray3&);
      GPU_INLINE_FUNCTION GPUArray3(const GPUArray3&);
      
      //__________________________________
      //
      GPU_INLINE_FUNCTION void checkBounds( const int3& idx){
         int3 Lo = getLowIndex();  
         int3 Hi = getHighIndex();                                             
         if (idx.x < Lo.x || idx.y < Lo.y || idx.z < Lo.z ||                                    
             idx.x > Hi.x || idx.y > Hi.y || idx.z > Hi.z){        
                printf ("GPU OUT_OF_BOUND ERROR: pointer: %p (%d, %d, %d) not inside (%d, %d, %d)-(%d, %d, %d) \n",    
                        (void*)d_data, idx.x, idx.y, idx.z, Lo.x, Lo.y, Lo.z, Hi.x, Hi.y, Hi.z);                          
        }
      }

      //__________________________________
      //    const version
      GPU_INLINE_FUNCTION void checkBounds( const int3& idx) const{
         int3 Lo = getLowIndex();  
         int3 Hi = getHighIndex();                                             
         if (idx.x < Lo.x || idx.y < Lo.y || idx.z < Lo.z ||                                    
             idx.x > Hi.x || idx.y > Hi.y || idx.z > Hi.z){        
                printf ("GPU OUT_OF_BOUND ERROR (const) (: pointer: %p (%d, %d, %d) not inside (%d, %d, %d)-(%d, %d, %d) \n",    
                        (void*)d_data, idx.x, idx.y, idx.z, Lo.x, Lo.y, Lo.z, Hi.x, Hi.y, Hi.z);                          
        }
      }
      //__________________________________
      //
      GPU_INLINE_FUNCTION void checkBounds3(const int& x, const int& y, const int& z ){
        int3 idx = make_int3(x,y,z);
        checkBounds(idx);
      }
      //__________________________________
      //    const version
      GPU_INLINE_FUNCTION void checkBounds3(const int& x, const int& y, const int& z )const {
        int3 idx = make_int3(x,y,z);
        checkBounds(idx);
      }
      
      
  };

  template<class T> class GPUGridVariable: public GPUGridVariableBase, public GPUArray3<T> {

    friend class KokkosScheduler;   // allow scheduler access
    friend class UnifiedScheduler;  // allow scheduler access

    public:

      GPU_INLINE_FUNCTION GPUGridVariable() {}
      GPU_INLINE_FUNCTION virtual ~GPUGridVariable() {}

      GPU_INLINE_FUNCTION virtual size_t getMemSize()
      {
        return GPUArray3<T>::getMemSize();
      }
      
      GPU_INLINE_FUNCTION virtual int3 getLowIndex()
      {
        return GPUArray3<T>::getLowIndex();
      }
      
      GPU_INLINE_FUNCTION virtual int3 getHighIndex()
      {
        return GPUArray3<T>::getHighIndex();
      }
      GPU_INLINE_FUNCTION virtual int3 getLowIndex() const
      {
        return GPUArray3<T>::getLowIndex();
      }
      
      GPU_INLINE_FUNCTION virtual int3 getHighIndex() const
      {
        return GPUArray3<T>::getHighIndex();
      }
    

      GPU_INLINE_FUNCTION void* getVoidPointer() const {
        return GPUArray3<T>::d_data;
      }

    private:

      GPU_INLINE_FUNCTION virtual void getArray3(int3& offset, int3& size, void* &ptr) const
      {
        GPUArray3<T>::getOffsetSizePtr(offset, size, ptr);
      }

      GPU_INLINE_FUNCTION virtual void setArray3(const int3& offset, const int3& size, void* &ptr) const
      {
        GPUArray3<T>::setOffsetSizePtr(offset, size, ptr);
      }
  };

} // end namespace Uintah

#endif // UINTAH_GRID_VARIABLES_GPUGRIDVARIABLE_H
