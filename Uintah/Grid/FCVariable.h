
#ifndef UINTAH_HOMEBREW_FCVARIABLE_H
#define UINTAH_HOMEBREW_FCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/FCVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Grid/TypeUtils.h>
#include <Uintah/Interface/InputContext.h>
#include <Uintah/Interface/OutputContext.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Malloc/Allocator.h>
#include <unistd.h>
#include <errno.h>

#include <iostream> // TEMPORARY

using namespace Uintah;

namespace Uintah {
   using SCICore::Exceptions::ErrnoException;
   using SCICore::Exceptions::InternalError;
   using SCICore::Geometry::Vector;

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
   Variable__Face_Centered

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T> 
class FCVariable : public Array3<T>, public FCVariableBase {
   public:
      FCVariable();
      FCVariable(const FCVariable<T>&);
      virtual ~FCVariable();
      
      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();
      
      virtual void copyPointer(const FCVariableBase&);

      //////////
      // Insert Documentation Here:
      virtual FCVariable<T>* clone() const;

 virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex);
      
      //////////
      // Insert Documentation Here:
      void copyPatch(FCVariableBase* src,
		      const IntVector& lowIndex,
		      const IntVector& highIndex);

      FCVariable<T>& operator=(const FCVariable<T>&);
     
     // Replace the values on the indicated face with value
      void fillFace(Patch::FaceType face, const T& value)
	{ 
	  IntVector low = getLowIndex();
	  IntVector hi = getHighIndex();
	  switch (face) {
	  case Patch::xplus:
	    for (int j = low.y(); j<hi.y(); j++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(hi.x()-1,j,k)] = value;
	      }
	    }
	    break;
	  case Patch::xminus:
	    for (int j = low.y(); j<hi.y(); j++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(low.x(),j,k)] = value;
	      }
	    }
	    break;
	  case Patch::yplus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(i,hi.y()-1,k)] = value;
	      }
	    }
	    break;
	  case Patch::yminus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(i,low.y(),k)] = value;
	      }
	    }
	    break;
	  case Patch::zplus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int j = low.y(); j<hi.y(); j++) {
		(*this)[IntVector(i,j,hi.z()-1)] = value;
	      }
	    }
	    break;
	  case Patch::zminus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int j = low.y(); j<hi.y(); j++) {
		(*this)[IntVector(i,j,low.z())] = value;
	      }
	    }
	    break;
	  }

	};
     
      // Use to apply symmetry boundary conditions.  On the
      // indicated face, replace the component of the vector
      // normal to the face with 0.0
      void fillFaceNormal(Patch::FaceType face)
	{
	  IntVector low = getLowIndex();
	  IntVector hi = getHighIndex();
	  switch (face) {
	  case Patch::xplus:
	    for (int j = low.y(); j<hi.y(); j++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(hi.x()-1,j,k)] =
		  Vector(0.0,(*this)[IntVector(hi.x()-1,j,k)].y(),
			 (*this)[IntVector(hi.x()-1,j,k)].z());
	      }
	    }
	    break;
	  case Patch::xminus:
	    for (int j = low.y(); j<hi.y(); j++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(low.x(),j,k)] = 
		  Vector(0.0,(*this)[IntVector(low.x(),j,k)].y(),
			 (*this)[IntVector(low.x(),j,k)].z());
	      }
	    }
	    break;
	  case Patch::yplus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(i,hi.y()-1,k)] =
		  Vector((*this)[IntVector(i,hi.y()-1,k)].x(),0.0,
			 (*this)[IntVector(i,hi.y()-1,k)].z());
	      }
	    }
	    break;
	  case Patch::yminus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(i,low.y(),k)] =
		  Vector((*this)[IntVector(i,low.y(),k)].x(),0.0,
			 (*this)[IntVector(i,low.y(),k)].z());
	      }
	    }
	    break;
	  case Patch::zplus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int j = low.y(); j<hi.y(); j++) {
		(*this)[IntVector(i,j,hi.z()-1)] =
		  Vector((*this)[IntVector(i,j,hi.z()-1)].x(),
			 (*this)[IntVector(i,j,hi.z()-1)].y(),0.0);
	      }
	    }
	    break;
	  case Patch::zminus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int j = low.y(); j<hi.y(); j++) {
		(*this)[IntVector(i,j,low.z())] =
		  Vector((*this)[IntVector(i,j,low.z())].x(),
			 (*this)[IntVector(i,j,low.z())].y(),0.0);
	      }
	    }
	    break;
	  }
	};
     
      virtual void emit(OutputContext&);
      virtual void read(InputContext&);
      static TypeDescription::Register registerMe;

   private:
   };
      template<class T>
      TypeDescription::Register
	FCVariable<T>::registerMe(getTypeDescription());

   template<class T>
      const TypeDescription*
      FCVariable<T>::getTypeDescription()
      {
	 // Dd: Whis isn't td a class variable and does it
	 // need to be deleted in the destructor?
	std::cerr << "getting type description from FC var\n";

	 // Dd: Copied this from NC Var... don't know if it is 
	 // correct.
	static TypeDescription* td;
	if(!td){
	  td = scinew TypeDescription(TypeDescription::FCVariable,
				   "FCVariable",
				   fun_getTypeDescription((T*)0));
	}
	return td;
      }
   
   template<class T>
      FCVariable<T>::~FCVariable()
      {
      }
   
   template<class T>
      FCVariable<T>*
      FCVariable<T>::clone() const
      {
	 return scinew FCVariable<T>(*this);
      }
   template<class T>
     void
     FCVariable<T>::copyPointer(const FCVariableBase& copy)
     {
       const FCVariable<T>* c = dynamic_cast<const FCVariable<T>* >(&copy);
       if(!c)
	 throw TypeMismatchException("Type mismatch in FC variable");
       *this = *c;
     }

 
   template<class T>
      FCVariable<T>&
      FCVariable<T>::operator=(const FCVariable<T>& copy)
      {
	 if(this != &copy){
	    Array3<T>::operator=(copy);
	 }
	 return *this;
      }
   
   template<class T>
      FCVariable<T>::FCVariable()
      {
	//	 std::cerr << "FCVariable ctor not done!\n";
      }
   
   template<class T>
      FCVariable<T>::FCVariable(const FCVariable<T>& copy)
      : Array3<T>(copy)
      {
	//	 std::cerr << "FCVariable copy ctor not done!\n";
      }
   
   template<class T>
      void FCVariable<T>::allocate(const IntVector& lowIndex,
				   const IntVector& highIndex)
      {
	 if(getWindow())
	    throw InternalError("Allocating a FCvariable that "
				"is apparently already allocated!");
	 resize(lowIndex, highIndex);
      }

   template<class T>
      void
      FCVariable<T>::copyPatch(FCVariableBase* srcptr,
				const IntVector& lowIndex,
				const IntVector& highIndex)
      {
	 const FCVariable<T>* c = dynamic_cast<const FCVariable<T>* >(srcptr);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in FC variable");
	 const FCVariable<T>& src = *c;
	 for(int i=lowIndex.x();i<highIndex.x();i++)
	    for(int j=lowIndex.y();j<highIndex.y();j++)
	       for(int k=lowIndex.z();k<highIndex.z();k++)
		  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
      }
   
   template<class T>
      void
      FCVariable<T>::emit(OutputContext& oc)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 if(td->isFlat()){
	    // This could be optimized...
	    IntVector l(getLowIndex());
	    IntVector h(getHighIndex());
	    for(int x=l.x();x<h.x();x++){
	       for(int y=l.y();y<h.y();y++){
		  size_t size = sizeof(T)*(h.z()-l.z());
		  ssize_t s=write(oc.fd, &(*this)[IntVector(x,y,l.z())], size);
		  if(size != s)
		     throw ErrnoException("FCVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      void
      FCVariable<T>::read(InputContext& oc)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 if(td->isFlat()){
	    // This could be optimized...
	    IntVector l(getLowIndex());
	    IntVector h(getHighIndex());
	    for(int x=l.x();x<h.x();x++){
	       for(int y=l.y();y<h.y();y++){
		  size_t size = sizeof(T)*(h.z()-l.z());
		  ssize_t s=::read(oc.fd, &(*this)[IntVector(x,y,l.z())], size);
		  if(size != s)
		     throw ErrnoException("FCVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/06/14 21:59:35  jas
// Copied CCVariable stuff to make FCVariables.  Implementation is not
// correct for the actual data storage and iteration scheme.
//
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

