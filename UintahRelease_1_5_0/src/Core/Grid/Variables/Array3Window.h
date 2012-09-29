/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_Array3Window_H
#define UINTAH_HOMEBREW_Array3Window_H

#include <Core/Util/RefCounted.h>
#include <Core/Grid/Variables/Array3Data.h>
#include <climits>
#include <Core/Geometry/IntVector.h>

#if SCI_ASSERTION_LEVEL >= 3
// test the range and throw a more informative exception
// instead of testing one index's range
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>
using std::ostringstream;
#endif

/**************************************

CLASS
   Array3Window
   
GENERAL INFORMATION

   Array3Window.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Array3Window

DESCRIPTION

   The Array3Window class supports a windowed access into memory.  For
   multiple patches per process, this allows one large block of memory
   to be accessed by each patch with local 0-based access from the
   patch.  The offset contained herein is a GLOBAL offset.
  
WARNING
  
****************************************/

namespace Uintah {
   template<class T> class Array3Window : public RefCounted {
   public:
      Array3Window(Array3Data<T>*);
      Array3Window(Array3Data<T>*, const IntVector& offset,
                   const IntVector& lowIndex, const IntVector& highIndex);
      virtual ~Array3Window();
      
      inline const Array3Data<T>* getData() const {
         return data;
      }
     
      inline Array3Data<T>* getData() {
         return data;
      }
      
      void copy(const Array3Window<T>*);
      void copy(const Array3Window<T>*, const IntVector& low, const IntVector& high);
      void initialize(const T&);
      void initialize(const T&, const IntVector& s, const IntVector& e);
      inline IntVector getLowIndex() const {
         return lowIndex;
      }
      inline IntVector getHighIndex() const {
         return highIndex;
      }
      inline IntVector getOffset() const {
         return offset;
      }
      inline T& get(const IntVector& idx) {
         ASSERT(data);
#if SCI_ASSERTION_LEVEL >= 3
          // used to be several CHECKARRAYBOUNDS, but its lack of information
          // has bitten me a few times.... BJW
          bool bad = false;
          if (idx.x() < lowIndex.x() || idx.x() >= highIndex.x()) bad = true;
          if (idx.y() < lowIndex.y() || idx.y() >= highIndex.y()) bad = true;
          if (idx.z() < lowIndex.z() || idx.z() >= highIndex.z()) bad = true;
          if (bad) {
            ostringstream ostr;
            ostr << "Index not in range of window (on get): index: " << idx << " window low " 
                 << lowIndex << " window high " << highIndex;
            throw SCIRun::InternalError(ostr.str(), __FILE__, __LINE__);
          }
#endif
         return data->get(idx-offset);
      }
      
      ///////////////////////////////////////////////////////////////////////
      // Return pointer to the data 
      // (**WARNING**not complete implementation)
      inline T* getPointer() {
        return data ? (data->getPointer()) : 0;
      }
      
      ///////////////////////////////////////////////////////////////////////
      // Return const pointer to the data 
      // (**WARNING**not complete implementation)
      inline const T* getPointer() const {
        return (data->getPointer());
      }

      inline T*** get3DPointer() {
        return data ? data->get3DPointer():0;
      }
      inline T*** get3DPointer() const {
        return data ? data->get3DPointer():0;
      }

   private:
      
      Array3Data<T>* data;
      IntVector offset;
      IntVector lowIndex;
      IntVector highIndex;
      Array3Window(const Array3Window<T>&);
      Array3Window<T>& operator=(const Array3Window<T>&);
   };
   
   template<class T>
      void Array3Window<T>::initialize(const T& val)
      {
         data->initialize(val, lowIndex-offset, highIndex-offset);
      }
   
   template<class T>
      void Array3Window<T>::copy(const Array3Window<T>* from)
      {
         data->copy(lowIndex-offset, highIndex-offset, from->data,
                    from->lowIndex-from->offset, from->highIndex-from->offset);
      }
   
   template<class T>
      void Array3Window<T>::copy(const Array3Window<T>* from,
                                 const IntVector& low, const IntVector& high)
      {
         data->copy(low-offset, high-offset, from->data,
                    low-from->offset, high-from->offset);
      }
   
   template<class T>
      void Array3Window<T>::initialize(const T& val,
                                       const IntVector& s,
                                       const IntVector& e)
      {
         CHECKARRAYBOUNDS(s.x(), lowIndex.x(), highIndex.x());
         CHECKARRAYBOUNDS(s.y(), lowIndex.y(), highIndex.y());
         CHECKARRAYBOUNDS(s.z(), lowIndex.z(), highIndex.z());
         CHECKARRAYBOUNDS(e.x(), s.x(), highIndex.x()+1);
         CHECKARRAYBOUNDS(e.y(), s.y(), highIndex.y()+1);
         CHECKARRAYBOUNDS(e.z(), s.z(), highIndex.z()+1);
         data->initialize(val, s-offset, e-offset);
      }
   
   template<class T>
      Array3Window<T>::Array3Window(Array3Data<T>* data)
      : data(data), offset(0,0,0), lowIndex(0,0,0), highIndex(data->size())
      {
         data->addReference();
      }
   
   template<class T>
      Array3Window<T>::Array3Window(Array3Data<T>* data,
                                    const IntVector& offset,
                                    const IntVector& lowIndex,
                                    const IntVector& highIndex)
      : data(data), offset(offset), lowIndex(lowIndex), highIndex(highIndex)
      {
        // null data can be used for a place holder in OnDemandDataWarehouse
        if (data != 0) {
#if SCI_ASSERTION_LEVEL >= 3
          // used to be several CHECKARRAYBOUNDS, but its lack of information
          // has bitten me a few times.... BJW
          IntVector low(lowIndex-offset);
          IntVector high(highIndex-offset);
          bool bad = false;
          if (low.x() < 0 || low.x() >= data->size().x()) bad = true;
          if (low.y() < 0 || low.y() >= data->size().y()) bad = true;
          if (low.z() < 0 || low.z() >= data->size().z()) bad = true;
          if (high.x() < 0 || high.x() > data->size().x()) bad = true;
          if (high.y() < 0 || high.y() > data->size().y()) bad = true;
          if (high.z() < 0 || high.z() > data->size().z()) bad = true;;
          if (bad) {
            ostringstream ostr;
            ostr << "Data not in range of new window: data size: " << data->size() << " window low " 
                 << lowIndex << " window high " << highIndex;
            throw SCIRun::InternalError(ostr.str(), __FILE__, __LINE__);
          }
#endif
          data->addReference();
        }
        else {
          // To use null data, put the offset as {INT_MAX, INT_MAX, INT_MAX}.
          // This way, when null is used accidentally, this assertion will
          // fail while allowing purposeful (and hopefully careful) uses
          // of null data.
          ASSERT(offset == IntVector(INT_MAX, INT_MAX, INT_MAX));
        }
      }
   
   template<class T>
      Array3Window<T>::~Array3Window()
      {
        if(data && data->removeReference())
        {
          delete data;
          data=0;
        }
      }
} // End namespace Uintah

#endif
