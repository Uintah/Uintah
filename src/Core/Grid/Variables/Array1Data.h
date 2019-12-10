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

#ifndef UINTAH_HOMEBREW_ARRAY1DATA_H
#define UINTAH_HOMEBREW_ARRAY1DATA_H

#include <Core/Util/RefCounted.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

#include <sci_defs/kokkos_defs.h>

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif //UINTAH_ENABLE_KOKKOS

namespace Uintah {

  /**************************************

    CLASS
    Array1Data

    GENERAL INFORMATION

    Array1Data.h

    Steven G. Parker
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


    KEYWORDS
    Array1Data

    DESCRIPTION
    Long description...

    WARNING

   ****************************************/

#ifdef UINTAH_ENABLE_KOKKOS
template <typename T>
using KokkosData = Kokkos::View<T*, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
#endif //UINTAH_ENABLE_KOKKOS

  template<class T> class Array1Data : public RefCounted {
    public:
      Array1Data(const int& size);
      virtual ~Array1Data();

      inline int size() const {
        return d_size;
      }
      void copy(const int& ts, const int& te,
                const Array1Data<T>* from,
                const int& fs, const int& fe);

      void initialize(const T& val, const int& s, const int& e);
#if 0
      inline T& get(const int& idx) {
        CHECKARRAYBOUNDS(idx, 0, d_size);
        return d_data[idx];
      }
#endif
      inline T& get(int i) {
        CHECKARRAYBOUNDS(i, 0, d_size);
        return d_data[i];
      }

      ///////////////////////////////////////////////////////////////////////
      // Return pointer to the data
      // (**WARNING**not complete implementation)
      inline T* getPointer() {
        return d_data;
      }

      ///////////////////////////////////////////////////////////////////////
      // Return const pointer to the data
      // (**WARNING**not complete implementation)
      inline const T* getPointer() const {
        return d_data;
      }

#ifdef UINTAH_ENABLE_KOKKOS
      inline KokkosData<T> getKokkosData() const {
        return KokkosData<T>(d_data, d_size);
      }
#endif //UINTAH_ENABLE_KOKKOS


    private:
      T*    d_data;
      int d_size;

      Array1Data& operator=(const Array1Data&);
      Array1Data(const Array1Data&);
  };

  template<class T>
    void Array1Data<T>::initialize(const T& val,
        const int& lowIndex,
        const int& highIndex)
    {
      CHECKARRAYBOUNDS(lowIndex, 0, d_size);
      CHECKARRAYBOUNDS(highIndex, lowIndex, d_size+1);
      T* d = &d_data[lowIndex];
      int s = highIndex-lowIndex;
      for(int i=0;i<s;i++){
	d[i]=val;
      }
    }

  template<class T>
    void Array1Data<T>::copy(const int& to_lowIndex,
                             const int& to_highIndex,
                             const Array1Data<T>* from,
                             const int& from_lowIndex,
                             const int& from_highIndex)
    {
      CHECKARRAYBOUNDS(to_lowIndex, 0, d_size);
      CHECKARRAYBOUNDS(to_highIndex, to_lowIndex, d_size+1);
      T* dst = &d_data[to_lowIndex];

      CHECKARRAYBOUNDS(from_lowIndex, 0, from->d_size);
      CHECKARRAYBOUNDS(from_highIndex, from_lowIndex,from->d_size+1);
      T* src = &from->d_data[from_lowIndex];

      int s = from_highIndex-from_lowIndex;
      //int s_check = to_highIndex-to_lowIndex;
      // Check to make sure that the two window sizes are the same
      ASSERT(s == to_highIndex-to_lowIndex);
      for(int k=0;k<s;k++) {
	dst[k]=src[k];
      }
    }


  template<class T>
    Array1Data<T>::Array1Data(const int& size)
    : d_size(size)
    {
      long s=d_size;
      if(s){
        d_data=new T[s];
      }
    }

  template<class T>
    Array1Data<T>::~Array1Data()
    {
      if(d_data){
        delete[] d_data;
	d_data = 0;
      }
    }

} // End namespace Uintah

#endif
