/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_PERPATCH_H
#define UINTAH_HOMEBREW_PERPATCH_H

#include <Core/Grid/Variables/PerPatchBase.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Malloc/Allocator.h>

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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Sole_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class Variable;

   // 'T' should be a Handle to be something that's RefCounted.
   // Otherwise, do your own memory management...
   template<class T> class PerPatch : public PerPatchBase {
   public:
      inline PerPatch() {}
      inline PerPatch(T value) : value(value) {}
      virtual void copyPointer(Variable&);
      inline PerPatch(const PerPatch<T>& copy) : value(copy.value) {}
      virtual ~PerPatch();
      
      static const TypeDescription* getTypeDescription();
      
      inline operator T () const {
         return value;
      }
      inline T& get() {
         return value;
      }
      inline const T& get() const {
         return value;
      }
      void setData(const T&);
      virtual PerPatchBase* clone() const;
      PerPatch<T>& operator=(const PerPatch<T>& copy);
      virtual void getSizeInfo(string& elems, unsigned long& totsize,
                               void*& ptr) const {
        elems="1";
        totsize=sizeof(T);
        ptr=(void*)&value;
      }
   private:
      static TypeDescription* td;
      T value;
      // this function only exists to satisfy the TypeDescription, it will return null.
      static Variable* maker();
   };

   template<class T>
   TypeDescription* PerPatch<T>::td = 0;
   
   template<class T>
     Variable*
     PerPatch<T>::maker()
     {
       return NULL;
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


   template<class T>
      void
      PerPatch<T>::setData(const T& val)
      {
        value = val;
      }
} // End namespace Uintah

#endif
