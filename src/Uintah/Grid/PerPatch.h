#ifndef UINTAH_HOMEBREW_PERPATCH_H
#define UINTAH_HOMEBREW_PERPATCH_H

#include <Uintah/Grid/PerPatchBase.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <SCICore/Malloc/Allocator.h>

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
      void setData(const T&);
      virtual PerPatchBase* clone() const;
      PerPatch<T>& operator=(const PerPatch<T>& copy);
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
            throw TypeMismatchException("Type mismatch in PerPatch variable");
         *this = *c;
      }


   template<class T>
      void
      PerPatch<T>::setData(const T& val)
      {
	value = val;
      }

} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/09/25 18:12:20  sparker
// do not use covariant return types due to problems with g++
// other linux/g++ fixes
//
// Revision 1.4  2000/09/20 15:48:30  sparker
// Added .copy() method to copy one Array3 from another
//
// Revision 1.3  2000/06/30 04:17:25  rawat
// added setData() function in PerPatch.h
//
// Revision 1.2  2000/06/05 19:44:48  guilkey
// Created PerPatchBase, filled in PerPatch class.
//
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
