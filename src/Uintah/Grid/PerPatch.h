#ifndef UINTAH_HOMEBREW_PERPATCH_H
#define UINTAH_HOMEBREW_PERPATCH_H

#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>

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

   template<class T> class PerPatch : public DataItem {
   public:
      inline PerPatch() {}
      inline PerPatch(T value) : value(value) {}
      inline PerPatch(const PerPatch<T>& copy) : value(copy.value) {}
      virtual ~PerPatch();
      
      static const TypeDescription* getTypeDescription();
      
      inline operator T () const {
	 return value;
      }
      virtual void get(DataItem&) const;
      virtual void setData(const T&);
      virtual PerPatch<T>* clone() const;
      virtual void allocate(const Patch*);
   private:
      PerPatch<T>& operator=(const PerPatch<T>& copy);
      T value;
   };
   
   template<class T>
      const TypeDescription*
      PerPatch<T>::getTypeDescription()
      {
	 return 0;
      }
   
   template<class T>
      void
      PerPatch<T>::get(DataItem& copy) const
      {
	 PerPatch<T>* ref = dynamic_cast<PerPatch<T>*>(&copy);
	 if(!ref)
	    throw TypeMismatchException("PerPatch<T>");
	 *ref = *this;
      }
   
   template<class T>
      PerPatch<T>::~PerPatch()
      {
      }
   
   template<class T>
      PerPatch<T>*
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
      PerPatch<T>::allocate(const Patch*)
      {
	 throw TypeMismatchException("PerPatch shouldn't use allocate");
      }
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/30 20:19:32  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.3  2000/05/15 19:39:49  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.2  2000/04/26 06:48:52  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/03/22 00:32:13  sparker
// Added Face-centered variable class
// Added Per-patch data class
// Added new task constructor for procedures with arguments
// Use Array3Index more often
//
//
//

#endif
