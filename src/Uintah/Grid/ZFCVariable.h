
#ifndef UINTAH_HOMEBREW_ZFCVARIABLE_H
#define UINTAH_HOMEBREW_ZFCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/ZFCVariableBase.h>
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
   ZFCVariable
   
   Short description...

GENERAL INFORMATION

   ZFCVariable.h

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
class ZFCVariable : public Array3<T>, public ZFCVariableBase {
   public:
      ZFCVariable();
      ZFCVariable(const ZFCVariable<T>&);
      virtual ~ZFCVariable();
      
      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();
      
      virtual void copyPointer(const ZFCVariableBase&);

      //////////
      // Insert Documentation Here:
      virtual ZFCVariableBase* clone() const;

      virtual void allocate(const IntVector& lowIndex,
			    const IntVector& highIndex);
      
      //////////
      // Insert Documentation Here:
      void copyPatch(ZFCVariableBase* src,
		      const IntVector& lowIndex,
		      const IntVector& highIndex);

      ZFCVariable<T>& operator=(const ZFCVariable<T>&);
      virtual void* getBasePointer();
      virtual const TypeDescription* virtualGetTypeDescription() const;
      virtual void getSizes(IntVector& low, IntVector& high,
			   IntVector& siz) const;

     
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
	  case Patch::numFaces:
	    break;
	  }

	};
     
      // Set the Neumann BC condition using a 1st order approximation
      void fillFaceFlux(Patch::FaceType face, const T& value, const Vector& dx)
	{ 
	  IntVector low = getLowIndex();
	  IntVector hi = getHighIndex();
	  switch (face) {
	  case Patch::xplus:
	    for (int j = low.y(); j<hi.y(); j++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(hi.x()-1,j,k)] = 
		   (*this)[IntVector(hi.x()-2,j,k)] - value*dx.x();
	      }
	    }
	    break;
	  case Patch::xminus:
	    for (int j = low.y(); j<hi.y(); j++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(low.x(),j,k)] = 
		  (*this)[IntVector(low.x()+1,j,k)] - value * dx.x();
	      }
	    }
	    break;
	  case Patch::yplus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(i,hi.y()-1,k)] = 
		  (*this)[IntVector(i,hi.y()-2,k)] - value * dx.y();
	      }
	    }
	    break;
	  case Patch::yminus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(i,low.y(),k)] = 
		  (*this)[IntVector(i,low.y()+1,k)] - value * dx.y();
	      }
	    }
	    break;
	  case Patch::zplus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int j = low.y(); j<hi.y(); j++) {
		(*this)[IntVector(i,j,hi.z()-1)] = 
		  (*this)[IntVector(i,j,hi.z()-2)] - value * dx.z();
	      }
	    }
	    break;
	  case Patch::zminus:
	    for (int i = low.x(); i<hi.x(); i++) {
	      for (int j = low.y(); j<hi.y(); j++) {
		(*this)[IntVector(i,j,low.z())] = 
		  (*this)[IntVector(i,j,low.z()+1)] -  value * dx.z();
	      }
	    }
	    break;
	  case Patch::numFaces:
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
   static Variable* maker();
   };
      template<class T>
      TypeDescription::Register
	ZFCVariable<T>::registerMe(getTypeDescription());

   template<class T>
      const TypeDescription*
      ZFCVariable<T>::getTypeDescription()
      {
	 static TypeDescription* td;
	 if(!td){
	    td = scinew TypeDescription(TypeDescription::ZFCVariable,
					"ZFCVariable", &maker,
					fun_getTypeDescription((T*)0));
	 }
	 return td;
      }
   
   template<class T>
      const TypeDescription*
      ZFCVariable<T>::virtualGetTypeDescription() const
      {
	 return getTypeDescription();
      }

   template<class T>
      void
      ZFCVariable<T>::getSizes(IntVector& low, IntVector& high, IntVector& siz) const
      {
	 low=getLowIndex();
	 high=getHighIndex();
	 siz=size();
      }

   
   template<class T>
      Variable*
      ZFCVariable<T>::maker()
      {
	 return scinew ZFCVariable<T>();
      }
   
   template<class T>
      ZFCVariable<T>::~ZFCVariable()
      {
      }
   
   template<class T>
      ZFCVariableBase*
      ZFCVariable<T>::clone() const
      {
	 return scinew ZFCVariable<T>(*this);
      }
   template<class T>
     void
     ZFCVariable<T>::copyPointer(const ZFCVariableBase& copy)
     {
       const ZFCVariable<T>* c = dynamic_cast<const ZFCVariable<T>* >(&copy);
       if(!c)
	 throw TypeMismatchException("Type mismatch in FC variable");
       *this = *c;
     }

 
   template<class T>
      ZFCVariable<T>&
      ZFCVariable<T>::operator=(const ZFCVariable<T>& copy)
      {
	 if(this != &copy){
	    Array3<T>::operator=(copy);
	 }
	 return *this;
      }
   
   template<class T>
      ZFCVariable<T>::ZFCVariable()
      {
	//	 std::cerr << "ZFCVariable ctor not done!\n";
      }
   
   template<class T>
      ZFCVariable<T>::ZFCVariable(const ZFCVariable<T>& copy)
      : Array3<T>(copy)
      {
	//	 std::cerr << "ZFCVariable copy ctor not done!\n";
      }
   
   template<class T>
      void ZFCVariable<T>::allocate(const IntVector& lowIndex,
				   const IntVector& highIndex)
      {
	 if(getWindow())
	    throw InternalError("Allocating a ZFCVariable that "
				"is apparently already allocated!");
	 resize(lowIndex, highIndex);
      }

   template<class T>
      void
      ZFCVariable<T>::copyPatch(ZFCVariableBase* srcptr,
				const IntVector& lowIndex,
				const IntVector& highIndex)
      {
	 const ZFCVariable<T>* c = dynamic_cast<const ZFCVariable<T>* >(srcptr);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in FC variable");
	 const ZFCVariable<T>& src = *c;
	 for(int i=lowIndex.x();i<highIndex.x();i++)
	    for(int j=lowIndex.y();j<highIndex.y();j++)
	       for(int k=lowIndex.z();k<highIndex.z();k++)
		  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
      }
   
   template<class T>
      void
      ZFCVariable<T>::emit(OutputContext& oc)
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
		  if((ssize_t)size != s)
		     throw ErrnoException("ZFCVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }
   
   template<class T>
      void*
      ZFCVariable<T>::getBasePointer()
      {
	 return getPointer();
      }

   template<class T>
      void
      ZFCVariable<T>::read(InputContext& oc)
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
		  if((ssize_t)size != s)
		     throw ErrnoException("ZFCVariable::emit (write call)", errno);
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
// Revision 1.1  2000/11/28 03:47:26  jas
// Added FCVariables for the specific faces X,Y,and Z.
//
//

#endif

