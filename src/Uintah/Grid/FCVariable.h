
#ifndef UINTAH_HOMEBREW_FCVARIABLE_H
#define UINTAH_HOMEBREW_FCVARIABLE_H

#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <iostream> // TEMPORARY

namespace Uintah {

   class TypeDescription;

/**************************************

CLASS
   FCVariable
   
   Short description...

GENERAL INFORMATION

   FCVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of AFCidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Variable__Cell_Centered

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   template<class T> class FCVariable : public DataItem {
   public:
      FCVariable();
      FCVariable(const FCVariable<T>&);
      virtual ~FCVariable();
      
      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();
      
      //////////
      // Insert Documentation Here:
      virtual void get(DataItem&) const;
      
      //////////
      // Insert Documentation Here:
      virtual FCVariable<T>* clone() const;
      
      //////////
      // Insert Documentation Here:
      virtual void allocate(const Patch*);
      
      FCVariable<T>& operator=(const FCVariable<T>&);
   private:
   };
   
   template<class T>
      const TypeDescription*
      FCVariable<T>::getTypeDescription()
      {
	 return 0;
      }
   
   template<class T>
      FCVariable<T>::~FCVariable()
      {
      }
   
   template<class T>
      void
      FCVariable<T>::get(DataItem& copy) const
      {
	 FCVariable<T>* ref=dynamic_cast<FCVariable<T>*>(&copy);
	 if(!ref)
	    throw TypeMismatchException("FCVariable<T>");
	 *ref = *this;
      }
   
   template<class T>
      FCVariable<T>*
      FCVariable<T>::clone() const
      {
	 return scinew FCVariable<T>(*this);
      }
   
   template<class T>
      FCVariable<T>&
      FCVariable<T>::operator=(const FCVariable<T>& copy)
      {
	 if(this != &copy){
	    std::cerr << "FCVariable<T>::operator= not done!\n";
	 }
	 return *this;
      }
   
   template<class T>
      FCVariable<T>::FCVariable()
      {
	 std::cerr << "FCVariable ctor not done!\n";
      }
   
   template<class T>
      FCVariable<T>::FCVariable(const FCVariable<T>& copy)
      {
	 std::cerr << "FCVariable copy ctor not done!\n";
      }
   
   template<class T>
      void FCVariable<T>::allocate(const Patch*)
      {
	 std::cerr << "FCVariable::allocate not done!\n";
      }
   
} // end namespace Uintah

//
// $Log$
// Revision 1.4  2000/05/30 20:19:28  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.3  2000/05/15 19:39:47  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.2  2000/04/26 06:48:47  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/03/22 00:32:12  sparker
// Added Face-centered variable class
// Added Per-patch data class
// Added new task constructor for procedures with arguments
// Use Array3Index more often
//
//
//

#endif

