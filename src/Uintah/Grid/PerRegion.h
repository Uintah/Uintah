#ifndef UINTAH_HOMEBREW_PERREGION_H
#define UINTAH_HOMEBREW_PERREGION_H

#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <iostream> // TEMPORARY

namespace Uintah {

/**************************************

CLASS
   PerRegion
   
   Short description...

GENERAL INFORMATION

   PerRegion.h

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

   template<class T> class PerRegion : public DataItem {
   public:
      inline PerRegion() {}
      inline PerRegion(T value) : value(value) {}
      inline PerRegion(const PerRegion<T>& copy) : value(copy.value) {}
      virtual ~PerRegion();
      
      static const TypeDescription* getTypeDescription();
      
      inline operator T () const {
	 return value;
      }
      virtual void get(DataItem&) const;
      virtual void setData(const T&);
      virtual PerRegion<T>* clone() const;
      virtual void allocate(const Region*);
   private:
      PerRegion<T>& operator=(const PerRegion<T>& copy);
      T value;
   };
   
   template<class T>
      const TypeDescription*
      PerRegion<T>::getTypeDescription()
      {
	 //cerr << "PerRegion::getTypeDescription not done\n";
	 return 0;
      }
   
   template<class T>
      void
      PerRegion<T>::get(DataItem& copy) const
      {
	 PerRegion<T>* ref = dynamic_cast<PerRegion<T>*>(&copy);
	 if(!ref)
	    throw TypeMismatchException("PerRegion<T>");
	 *ref = *this;
      }
   
   template<class T>
      PerRegion<T>::~PerRegion()
      {
      }
   
   template<class T>
      PerRegion<T>*
      PerRegion<T>::clone() const
      {
	 return new PerRegion<T>(*this);
      }
   
   template<class T>
      PerRegion<T>&
      PerRegion<T>::operator=(const PerRegion<T>& copy)
      {
	 value = copy.value;
	 return *this;
      }
   
   template<class T>
      void
      PerRegion<T>::allocate(const Region*)
      {
	 throw TypeMismatchException("PerRegion shouldn't use allocate");
      }
   
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/04/26 06:48:52  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/03/22 00:32:13  sparker
// Added Face-centered variable class
// Added Per-region data class
// Added new task constructor for procedures with arguments
// Use Array3Index more often
//
//
//

#endif
