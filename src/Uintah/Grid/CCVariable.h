#ifndef UINTAH_HOMEBREW_CCVARIABLE_H
#define UINTAH_HOMEBREW_CCVARIABLE_H

#include <Uintah/Grid/CCVariableBase.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <iostream> // TEMPORARY

namespace Uintah {

   class TypeDescription;

/**************************************

CLASS
   CCVariable
   
   Short description...

GENERAL INFORMATION

   CCVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Variable__Cell_Centered

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   template<class T> class CCVariable : public CCVariableBase {
   public:
      CCVariable();
      CCVariable(const CCVariable<T>&);
      virtual ~CCVariable();
      
      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();
      
      //////////
      // Insert Documentation Here:
      virtual CCVariable<T>* clone() const;
      
      //////////
      // Insert Documentation Here:
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex);
      
      //////////
      // Insert Documentation Here:
      void initialize(const T& value);
      
      CCVariable<T>& operator=(const CCVariable<T>&);
   private:
   };
   
   template<class T>
      const TypeDescription*
      CCVariable<T>::getTypeDescription()
      {
	 //cerr << "CCVariable::getTypeDescription not done\n";
	 return 0;
      }
   
   template<class T>
      CCVariable<T>::~CCVariable()
      {
      }
   
   template<class T>
      CCVariable<T>*
      CCVariable<T>::clone() const
      {
	 return new CCVariable<T>(*this);
      }
   
   template<class T>
      CCVariable<T>&
      CCVariable<T>::operator=(const CCVariable<T>& copy)
      {
	 if(this != &copy){
	    std::cerr << "CCVariable<T>::operator= not done!\n";
	 }
	 return *this;
      }
   
   template<class T>
      CCVariable<T>::CCVariable()
      {
	 std::cerr << "CCVariable ctor not done!\n";
      }
   
   template<class T>
      CCVariable<T>::CCVariable(const CCVariable<T>& copy)
      {
	 std::cerr << "CCVariable copy ctor not done!\n";
      }
   
   template<class T>
      void CCVariable<T>::allocate(const IntVector& lowIndex,
				   const IntVector& highIndex)
      {
	 std::cerr << "CCVariable::allocate not done!\n";
      }

   template<class T>
      void initialize(const T& value) {
	 std::cerr << "CCVariable::initialize!\n";
      }
   
} // end namespace Uintah

//
// $Log$
// Revision 1.6  2000/05/11 20:10:21  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.5  2000/05/10 20:31:14  tan
// Added initialize member function. Currently nothing in the function,
// just to make the complilation work.
//
// Revision 1.4  2000/04/26 06:48:47  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif

