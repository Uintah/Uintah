/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef UINTAH_CORE_GRID_VARIABLES_PERPATCH_H
#define UINTAH_CORE_GRID_VARIABLES_PERPATCH_H

#include <Core/Grid/Variables/PerPatchBase.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Malloc/Allocator.h>
#include <stdio.h>
#include <cstring>
#include <memory>

namespace Uintah {

/**************************************

CLASS
   PerPatch
   
   Short description...

GENERAL INFORMATION

   PerPatch.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   PerPatch Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   // Uses C++11's shared_ptr to handle memory management.
   template<class T> class PerPatch : public PerPatchBase {
   public:
      inline PerPatch() : value(std::make_shared<T>()){}
      inline PerPatch(T value) : value(std::make_shared<T>(value)) {}

      virtual void copyPointer(Variable&);

      inline PerPatch(const PerPatch<T>& copy) : value(copy.value) {}

      virtual ~PerPatch();

      static const TypeDescription* getTypeDescription();

      inline operator T () const {
         return *value;
      }

      inline T& get() {
         return *value;
      }

      inline const T& get() const {
         return *value;
      }

      void setData(const T&);
      virtual PerPatchBase* clone() const;
      PerPatch<T>& operator=(const PerPatch<T>& copy);

      virtual void getSizeInfo(std::string& elems, unsigned long& totsize, void*& ptr) const
      {
        elems = "1";
        totsize = getDataSize();
        ptr = getBasePointer();
      }

      virtual size_t getDataSize() const {
        return sizeof(T);
      }

      virtual void* getBasePointer() const {
        return value.get();
        //return (void*)&value;
      }

      virtual bool copyOut(void* dst) const {
        void* src = (void*)(&value);
        size_t numBytes = getDataSize();
        void* retVal = std::memcpy(dst, src, numBytes);
        return (retVal == dst) ? true : false;
      }

   private:

      static TypeDescription* td;
      std::shared_ptr<T> value;
      static Variable* maker();

   }; // end class PerPatch


   template<class T>
   TypeDescription* PerPatch<T>::td = 0;
   
   template<class T>
     Variable*
     PerPatch<T>::maker()
     {
       return scinew PerPatch<T>();
     }


   template<class T>
      const TypeDescription*
      PerPatch<T>::getTypeDescription()
      {
        if(!td){

          // this is a hack to get a non-null perpatch
          // var for some functions the perpatches are used in (i.e., task->computes).
          // Since they're not fully-qualified variables, maker
          // would fail anyway.  And since most instances use Handle, it would be difficult.
          td = scinew TypeDescription(TypeDescription::PerPatch,
                                      "PerPatch", &maker,
                                      fun_getTypeDescription((int*)0));
        }
        return td;
      }


   //Manually list the double basic data type.  If others are needed, list them here too.
   //For everything else, a hacky solution is
   //used which defaults their internal type as if it were int
   template<>
    inline const TypeDescription*
      PerPatch<int>::getTypeDescription()
      {
        if(!td){
          TypeDescription* sub_td;
          sub_td = scinew TypeDescription(TypeDescription::int_type, "int", true, MPI_INT);
          td = scinew TypeDescription(TypeDescription::PerPatch, "PerPatch", &maker, sub_td);
        }
        return td;
      }
   
   template<>
    inline const TypeDescription*
      PerPatch<double>::getTypeDescription()
      {
        if(!td){
          TypeDescription* sub_td;
          sub_td = scinew TypeDescription(TypeDescription::double_type, "double", true, MPI_DOUBLE);
          td = scinew TypeDescription(TypeDescription::PerPatch, "PerPatch", &maker, sub_td);
        }
        return td;
      }


   template<>
    inline const TypeDescription*
      PerPatch<double*>::getTypeDescription()
      {
        if(!td){
          TypeDescription* sub_td;
          sub_td = scinew TypeDescription(TypeDescription::Other, "double*", true, MPI_DOUBLE);
          td = scinew TypeDescription(TypeDescription::PerPatch, "PerPatch", &maker, sub_td);
        }
        return td;
      }

   template<class T>
      PerPatch<T>::~PerPatch()
      {
      }
   
   template<class T>
      PerPatchBase*
      PerPatch<T>::clone() const
      {
         return scinew PerPatch<T>(*this);
      }
   
   template<class T>
      PerPatch<T>&
      PerPatch<T>::operator=(const PerPatch<T>& copy)
      {
         value = copy.value;
         return *this;
      }

   template<class T>
      void
      PerPatch<T>::copyPointer(Variable& copy)
      {
         const PerPatch<T>* c = dynamic_cast<const PerPatch<T>* >(&copy);
         if(!c)
           SCI_THROW(TypeMismatchException("Type mismatch in PerPatch variable", __FILE__, __LINE__));
         *this = *c;
      }

   template<>
      inline void
      PerPatch<double>::copyPointer(Variable& copy)
      {
         const PerPatch<double>* c = dynamic_cast<const PerPatch<double>* >(&copy);
         if(!c)
           SCI_THROW(TypeMismatchException("Type mismatch in PerPatch variable", __FILE__, __LINE__));
         *this = *c;
      }

   template<>
      inline void
      PerPatch<double*>::copyPointer(Variable& copy)
      {
         const PerPatch<double*>* c = dynamic_cast<const PerPatch<double*>* >(&copy);
         if(!c)
           SCI_THROW(TypeMismatchException("Type mismatch in PerPatch variable", __FILE__, __LINE__));
         *this = *c;
      }

   template<class T>
      void
      PerPatch<T>::setData(const T& val)
      {
        value = std::make_shared<T>(val);
      }
} // End namespace Uintah

#endif // UINTAH_CORE_GRID_VARIABLES_PERPATCH_H
