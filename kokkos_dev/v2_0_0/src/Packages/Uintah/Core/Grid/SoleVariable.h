#ifndef UINTAH_HOMEBREW_SOLEVARIABLE_H
#define UINTAH_HOMEBREW_SOLEVARIABLE_H

#include <Packages/Uintah/Core/Grid/DataItem.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>

namespace Uintah {
/**************************************

CLASS
   SoleVariable
   
   Short description...

GENERAL INFORMATION

   SoleVariable.h

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

   template<class T> class SoleVariable : public DataItem {
   public:
      inline SoleVariable() {}
      inline SoleVariable(T value) : value(value) {}
      inline SoleVariable(const SoleVariable<T>& copy) : value(copy.value) {}
      virtual ~SoleVariable();
      
      static const TypeDescription* getTypeDescription();
      
      inline operator T () const {
	 return value;
      }
      virtual void get(DataItem&) const;
      virtual SoleVariable<T>* clone() const;
      virtual void allocate(const Patch*);
   private:
      SoleVariable<T>& operator=(const SoleVariable<T>& copy);
      T value;
   };
   
   template<class T>
      const TypeDescription*
      SoleVariable<T>::getTypeDescription()
      {
	 return 0;
      }
   
   template<class T>
      void
      SoleVariable<T>::get(DataItem& copy) const
      {
	 SoleVariable<T>* ref = dynamic_cast<SoleVariable<T>*>(&copy);
	 if(!ref)
	   SCI_THROW(TypeMismatchException("SoleVariable<T>"));
	 *ref = *this;
      }
   
   template<class T>
      SoleVariable<T>::~SoleVariable()
      {
      }
   
   template<class T>
      SoleVariable<T>*
      SoleVariable<T>::clone() const
      {
	 return scinew SoleVariable<T>(*this);
      }
   
   template<class T>
      SoleVariable<T>&
      SoleVariable<T>::operator=(const SoleVariable<T>& copy)
      {
	 value = copy.value;
	 return *this;
      }
   
   template<class T>
      void
      SoleVariable<T>::allocate(const Patch*)
      {
	SCI_THROW(TypeMismatchException("SoleVariable shouldn't use allocate"));
      }
} // End namespace Uintah
   
#endif
