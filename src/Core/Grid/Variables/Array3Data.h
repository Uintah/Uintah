/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_ARRAY3DATA_H
#define UINTAH_HOMEBREW_ARRAY3DATA_H

#include <Core/Util/RefCounted.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

    CLASS
    Array3Data

    GENERAL INFORMATION

    Array3Data.h

    Steven G. Parker
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


    KEYWORDS
    Array3Data

    DESCRIPTION
    Long description...

    WARNING

   ****************************************/

  template<class T> class Array3Data : public RefCounted {
    public:
      Array3Data(const IntVector& size);
      virtual ~Array3Data();

      inline IntVector size() const {
        return d_size;
      }
      void copy(const IntVector& ts, const IntVector& te,
                const Array3Data<T>* from,
                const IntVector& fs, const IntVector& fe);

      void initialize(const T& val, const IntVector& s, const IntVector& e);
      inline T& get(const IntVector& idx) {
        CHECKARRAYBOUNDS(idx.x(), 0, d_size.x());
        CHECKARRAYBOUNDS(idx.y(), 0, d_size.y());
        CHECKARRAYBOUNDS(idx.z(), 0, d_size.z());
        return d_data3[idx.z()][idx.y()][idx.x()];
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

      inline T*** get3DPointer() {
        return d_data3;
      }
      inline T*** get3DPointer() const {
        return d_data3;
      }

    private:
      T*    d_data;
      T***  d_data3;
      IntVector d_size;

      Array3Data& operator=(const Array3Data&);
      Array3Data(const Array3Data&);
  };

  template<class T>
    void Array3Data<T>::initialize(const T& val,
        const IntVector& lowIndex,
        const IntVector& highIndex)
    {
      CHECKARRAYBOUNDS(lowIndex.x(), 0, d_size.x());
      CHECKARRAYBOUNDS(lowIndex.y(), 0, d_size.y());
      CHECKARRAYBOUNDS(lowIndex.z(), 0, d_size.z());
      CHECKARRAYBOUNDS(highIndex.x(), lowIndex.x(), d_size.x()+1);
      CHECKARRAYBOUNDS(highIndex.y(), lowIndex.y(), d_size.y()+1);
      CHECKARRAYBOUNDS(highIndex.z(), lowIndex.z(), d_size.z()+1);
      T* d = &d_data3[lowIndex.z()][lowIndex.y()][lowIndex.x()];
      IntVector s = highIndex-lowIndex;
      for(int i=0;i<s.z();i++){
        T* dd=d;
        for(int j=0;j<s.y();j++){
          T* ddd=dd;
          for(int k=0;k<s.x();k++) {
            ddd[k]=val;
          }
          dd+=d_size.x();
        }
        d+=d_size.x()*d_size.y();
      }
    }

  template<class T>
    void Array3Data<T>::copy(const IntVector& to_lowIndex,
                             const IntVector& to_highIndex,
                             const Array3Data<T>* from,
                             const IntVector& from_lowIndex,
                             const IntVector& from_highIndex)
    {
      CHECKARRAYBOUNDS(to_lowIndex.x(), 0, d_size.x());
      CHECKARRAYBOUNDS(to_lowIndex.y(), 0, d_size.y());
      CHECKARRAYBOUNDS(to_lowIndex.z(), 0, d_size.z());
      CHECKARRAYBOUNDS(to_highIndex.x(), to_lowIndex.x(), d_size.x()+1);
      CHECKARRAYBOUNDS(to_highIndex.y(), to_lowIndex.y(), d_size.y()+1);
      CHECKARRAYBOUNDS(to_highIndex.z(), to_lowIndex.z(), d_size.z()+1);
      T* dst = &d_data3[to_lowIndex.z()][to_lowIndex.y()][to_lowIndex.x()];

      CHECKARRAYBOUNDS(from_lowIndex.x(), 0, from->d_size.x());
      CHECKARRAYBOUNDS(from_lowIndex.y(), 0, from->d_size.y());
      CHECKARRAYBOUNDS(from_lowIndex.z(), 0, from->d_size.z());
      CHECKARRAYBOUNDS(from_highIndex.x(), from_lowIndex.x(),
          from->d_size.x()+1);
      CHECKARRAYBOUNDS(from_highIndex.y(), from_lowIndex.y(),
          from->d_size.y()+1);
      CHECKARRAYBOUNDS(from_highIndex.z(), from_lowIndex.z(),
          from->d_size.z()+1);
      T* src = &from->d_data3[from_lowIndex.z()][from_lowIndex.y()][from_lowIndex.x()];

      IntVector s = from_highIndex-from_lowIndex;
      //IntVector s_check = to_highIndex-to_lowIndex;
      // Check to make sure that the two window sizes are the same
      ASSERT(s == to_highIndex-to_lowIndex);
      for(int i=0;i<s.z();i++){
        T* dd=dst;
        T* ss=src;
        for(int j=0;j<s.y();j++){
          T* ddd=dd;
          T* sss=ss;
          for(int k=0;k<s.x();k++)
            ddd[k]=sss[k];
          dd+=d_size.x();
          ss+=from->d_size.x();
        }
        dst+=d_size.x()*d_size.y();
        src+=from->d_size.x()*from->d_size.y();
      }
    }

  template<class T>
    Array3Data<T>::Array3Data(const IntVector& size)
    : d_size(size)
    {
      long s=d_size.x()*d_size.y()*d_size.z();
      if(s){
        d_data=new T[s];
        d_data3=new T**[d_size.z()];
        d_data3[0]=new T*[d_size.z()*d_size.y()];
        d_data3[0][0]=d_data;
        for(int i=1;i<d_size.z();i++){
          d_data3[i]=d_data3[i-1]+d_size.y();
        }
        for(int j=1;j<d_size.z()*d_size.y();j++){
          d_data3[0][j]=d_data3[0][j-1]+d_size.x();
        }
      } else {
        d_data=0;
        d_data3=0;
      }
    }

  template<class T>
    Array3Data<T>::~Array3Data()
    {
      if(d_data){
        delete[] d_data;
        d_data=0;
        delete[] d_data3[0];
        d_data3[0]=0;
        delete[] d_data3;
        d_data3=0;
      }
    }

} // End namespace Uintah

#endif
