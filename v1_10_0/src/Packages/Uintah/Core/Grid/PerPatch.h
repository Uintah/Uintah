#ifndef UINTAH_HOMEBREW_PERPATCH_H
#define UINTAH_HOMEBREW_PERPATCH_H

#include <Packages/Uintah/Core/Grid/PerPatchBase.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {
  class TypeDescription;
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

   template<class T> class PerPatch : public PerPatchBase {
   public:
      inline PerPatch() {}
      inline PerPatch(T value) : value(value) {}
      virtual void copyPointer(const PerPatchBase&);
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
   };
   
   template<class T>
      const TypeDescription*
      PerPatch<T>::getTypeDescription()
      {
	 return 0;
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
      PerPatch<T>::copyPointer(const PerPatchBase& copy)
      {
         const PerPatch<T>* c = dynamic_cast<const PerPatch<T>* >(&copy);
         if(!c)
	   SCI_THROW(TypeMismatchException("Type mismatch in PerPatch variable"));
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
