#ifndef UINTAH_HOMEBREW_PERPATCH_H
#define UINTAH_HOMEBREW_PERPATCH_H

#include <Packages/Uintah/Core/Grid/PerPatchBase.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
using namespace std;

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

   class Variable;

   template<class T> class PerPatch : public PerPatchBase {
   public:
      inline PerPatch() { data = new T; deleteData = true; }
      inline PerPatch(const T& value) {
        data = new T;
        setData(value);
        deleteData = true;
      }
      virtual void copyPointer(const PerPatchBase&);
      inline PerPatch(const PerPatch<T>& copy) {
        // copy the pointer not the data
        data = copy.data;
      }

      virtual ~PerPatch();
      
      static const TypeDescription* getTypeDescription();
      
      inline operator T () const {
	 return *data;      
      }
      inline T& get() {
	 return *data;
      }
      inline const T& get() const {
	 return *data;
      }
      void setData(const T&);
      virtual PerPatchBase* clone() const;
      PerPatch<T>& operator=(const PerPatch<T>& copy);
      virtual void getSizeInfo(string& elems, unsigned long& totsize,
			       void*& ptr) const {
	elems="1";
	totsize=sizeof(T);
	ptr=(void*)data;
      }
   private:
      T* data;

      //! Try to remember if this PerPatch needs to delete the memory, since we copy the pointer
      //! Don't use RefCounted, as simple and/or pre-existing types probably don't use refcounted.
      mutable deleteData; 
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
          // would fail anyway.
          td = scinew TypeDescription(TypeDescription::PerPatch,
                                      "PerPatch", &maker,
                                      fun_getTypeDescription((int*)0));
        }
        return td;
      }
   
   template<class T>
      PerPatch<T>::~PerPatch()
      {
        if (data && deleteData) delete data;
      }
   
   template<class T>
      PerPatchBase*
      PerPatch<T>::clone() const
      {
        // copy the pointer, not the data
        PerPatch<T> *var  = scinew PerPatch<T>;
        var->data = data;
        deleteData = false;
        return var;
      }
   
   template<class T>
      PerPatch<T>&
      PerPatch<T>::operator=(const PerPatch<T>& copy)
      {
        // copy the pointer, not the data.
        data = copy.data;
        return *this;
      }

   template<class T>
      void
      PerPatch<T>::copyPointer(const PerPatchBase& copy)
      {
         const PerPatch<T>* c = dynamic_cast<const PerPatch<T>* >(&copy);
         if(!c)
	   SCI_THROW(TypeMismatchException("Type mismatch in PerPatch variable"));
         data = c->data;
         deleteData = false;
      }


   template<class T>
      void
      PerPatch<T>::setData(const T& val)
      {
	*data = val;
      }
} // End namespace Uintah

#endif
