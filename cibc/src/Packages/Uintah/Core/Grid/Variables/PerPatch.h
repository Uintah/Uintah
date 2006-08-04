#ifndef UINTAH_HOMEBREW_PERPATCH_H
#define UINTAH_HOMEBREW_PERPATCH_H

#include <Packages/Uintah/Core/Grid/Variables/PerPatchBase.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
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
      T value;
      // this function only exists to satisfy the TypeDescription, it will return null.
      static Variable* maker();
   };
   
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
        static TypeDescription* td;
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
