#ifndef UINTAH_HOMEBREW_CCVARIABLE_H
#define UINTAH_HOMEBREW_CCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/CCVariableBase.h>
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
      void copyPatch(CCVariableBase* src,
		      const IntVector& lowIndex,
		      const IntVector& highIndex);

      CCVariable<T>& operator=(const CCVariable<T>&);
     
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
	CCVariable<T>::registerMe(getTypeDescription());

   template<class T>
      const TypeDescription*
      CCVariable<T>::getTypeDescription()
      {
	static TypeDescription* td;
	if(!td){
	  td = scinew TypeDescription(TypeDescription::CCVariable,
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
	 return scinew CCVariable<T>(*this);
      }
   template<class T>
     void
     CCVariable<T>::copyPointer(const CCVariableBase& copy)
     {
       const CCVariable<T>* c = dynamic_cast<const CCVariable<T>* >(&copy);
       if(!c)
	 throw TypeMismatchException("Type mismatch in CC variable");
       *this = *c;
     }

 
   template<class T>
      CCVariable<T>&
      CCVariable<T>::operator=(const CCVariable<T>& copy)
      {
	 if(this != &copy){
	    Array3<T>::operator=(copy);
	 }
	 return *this;
      }
   
   template<class T>
      CCVariable<T>::CCVariable()
      {
	//	 std::cerr << "CCVariable ctor not done!\n";
      }
   
   template<class T>
      CCVariable<T>::CCVariable(const CCVariable<T>& copy)
      : Array3<T>(copy)
      {
	//	 std::cerr << "CCVariable copy ctor not done!\n";
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
      void
      CCVariable<T>::copyPatch(CCVariableBase* srcptr,
				const IntVector& lowIndex,
				const IntVector& highIndex)
      {
	 const CCVariable<T>* c = dynamic_cast<const CCVariable<T>* >(srcptr);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in CC variable");
	 const CCVariable<T>& src = *c;
	 for(int i=lowIndex.x();i<highIndex.x();i++)
	    for(int j=lowIndex.y();j<highIndex.y();j++)
	       for(int k=lowIndex.z();k<highIndex.z();k++)
		  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
      }
   
   template<class T>
      void
      CCVariable<T>::emit(OutputContext& oc)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 if(td->isFlat()){
	    // This could be optimized...
	    IntVector l(getLowIndex());
	    IntVector h(getHighIndex());
	    for(int z=l.z();z<h.z();z++){
	       for(int y=l.y();y<h.y();y++){
		  size_t size = sizeof(T)*(h.x()-l.x());
		  ssize_t s=write(oc.fd, &(*this)[IntVector(l.x(),y,z)], size);
		  if(size != s)
		     throw ErrnoException("CCVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      void
      CCVariable<T>::read(InputContext& oc)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 if(td->isFlat()){
	    // This could be optimized...
	    IntVector l(getLowIndex());
	    IntVector h(getHighIndex());
	    for(int z=l.z();z<h.z();z++){
	       for(int y=l.y();y<h.y();y++){
		  size_t size = sizeof(T)*(h.x()-l.x());
		  ssize_t s=::read(oc.fd, &(*this)[IntVector(l.x(),y,z)], size);
		  if(size != s)
		     throw ErrnoException("CCVariable::emit (write call)", errno);
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
// Revision 1.18  2000/06/22 21:56:30  sparker
// Changed variable read/write to fortran order
//
// Revision 1.17  2000/06/15 21:57:16  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.16  2000/06/13 21:28:30  jas
// Added missing TypeUtils.h for fun_forgottherestofname and copy constructor
// was wrong for CellIterator.
//
// Revision 1.15  2000/06/03 05:29:44  sparker
// Changed reduction variable emit to require ostream instead of ofstream
// emit now only prints number without formatting
// Cleaned up a few extraneously included files
// Added task constructor for an non-patch-based action with 1 argument
// Allow for patches and actions to be null
// Removed back pointer to this from Task::Dependency
//
// Revision 1.14  2000/06/01 22:04:23  tan
// Using operator[](const IntVector&) and void initialize(const T&)
// from Array3.
//
// Revision 1.13  2000/05/31 04:01:50  rawat
// partially completed CCVariable implementation
//
// Revision 1.12  2000/05/30 20:19:27  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
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
// Put two empty functions copyPointer and copyPatch just to make the
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

