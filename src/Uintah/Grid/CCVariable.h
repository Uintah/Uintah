#ifndef UINTAH_HOMEBREW_CCVARIABLE_H
#define UINTAH_HOMEBREW_CCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/CCVariableBase.h>
#include <Uintah/Exceptions/TypeMismatchException.h>

#include <SCICore/Exceptions/InternalError.h>

#include <iostream> // TEMPORARY

namespace Uintah {

   using SCICore::Exceptions::InternalError;

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

template<class T> 
class CCVariable : public Array3<T>, public CCVariableBase {
   public:
      CCVariable();
      CCVariable(const CCVariable<T>&);
      virtual ~CCVariable();
      
      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();
      
      virtual void copyPointer(const CCVariableBase&);

      //////////
      // Insert Documentation Here:
      virtual CCVariable<T>* clone() const;
      
      //////////
      // Insert Documentation Here:
      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex);
      
      //////////
      // Insert Documentation Here:
      void copyRegion(CCVariableBase* src,
		      const IntVector& lowIndex,
		      const IntVector& highIndex);

      //////////
      // Insert Documentation Here:
      void initialize(const T& value);
      
      //////////
      // Insert Documentation Here:
      T& operator[](const IntVector& idx) const;
      

      CCVariable<T>& operator=(const CCVariable<T>&);
   private:
   };
   
   template<class T>
      const TypeDescription*
      CCVariable<T>::getTypeDescription()
      {
	 // Dd: Whis isn't td a class variable and does it
	 // need to be deteleted in the destructor?
	std::cerr << "getting type description from CC var\n";

	 // Dd: Copied this from NC Var... don't know if it is 
	 // correct.
	 static TypeDescription* td = 0;
	 if(!td){
	    td = new TypeDescription(TypeDescription::CCVariable,
				     "CCVariable",
				     fun_getTypeDescription((T*)0));
	 }
	 return td;
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
	 if(getWindow())
	    throw InternalError("Allocating a CCvariable that "
				"is apparently already allocated!");
	 resize(lowIndex, highIndex);
      }

   template<class T>
      void CCVariable<T>::initialize(const T& value) {
	 std::cerr << "CCVariable::initialize!\n";
      }
      
   template<class T>
      void CCVariable<T>::copyPointer(const CCVariableBase&) {
	 std::cerr << "CCVariable::copyPointer!\n";
      }
   
   template<class T>
      void CCVariable<T>::copyRegion(CCVariableBase* src,
		      const IntVector& lowIndex,
		      const IntVector& highIndex) {
	 std::cerr << "CCVariable::copyRegion!\n";
      }

   template<class T>
      T& CCVariable<T>::operator[](const IntVector& idx) const {
	 std::cerr << "CCVariable::operator[]!\n";
      }

} // end namespace Uintah

//
// $Log$
// Revision 1.11  2000/05/28 17:25:55  dav
// adding code. someone should check to see if i did it corretly
//
// Revision 1.10  2000/05/15 19:39:46  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.9  2000/05/15 19:09:43  tan
// CCVariable<T>::operator[] is needed.
//
// Revision 1.8  2000/05/12 18:12:37  sparker
// Added CCVariableBase.cc to sub.mk
// Fixed copyPointer and other CCVariable methods - still not implemented
//
// Revision 1.7  2000/05/12 01:48:34  tan
// Put two empty functions copyPointer and copyRegion just to make the
// compiler work.
//
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

