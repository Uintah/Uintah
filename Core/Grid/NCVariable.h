#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/NCVariableBase.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/TypeUtils.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/Grid/TypeUtils.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>
#include <unistd.h>
#include <errno.h>

namespace Uintah {

using namespace SCIRun;

class TypeDescription;

/**************************************

CLASS
   NCVariable
   
GENERAL INFORMATION

   NCVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NCVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class NCVariable : public Array3<T>, public NCVariableBase {
   public:
     
     NCVariable();
     NCVariable(const NCVariable<T>&);
     virtual ~NCVariable();
     
     //////////
     // Insert Documentation Here:
     static const TypeDescription* getTypeDescription();
     
     virtual void copyPointer(const NCVariableBase&);
     
     //////////
     // Insert Documentation Here:
     virtual NCVariableBase* clone() const;
     
     //////////
     // Insert Documentation Here:
     virtual void allocate(const IntVector& lowIndex,
			   const IntVector& highIndex);
     
     virtual void copyPatch(NCVariableBase* src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex);
     NCVariable<T>& operator=(const NCVariable<T>&);
     
     virtual void* getBasePointer();
     virtual const TypeDescription* virtualGetTypeDescription() const;

     virtual void getSizes(IntVector& low, IntVector& high,
			   IntVector& siz, IntVector& strides) const;
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
	 default:
	     throw InternalError("Illegal FaceType in NCVariable::fillFace");
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
	 default:
	     throw InternalError("Illegal FaceType in NCVariable::fillFaceNormal");
         }
      };
     
      virtual void emit(OutputContext&);
      virtual void read(InputContext&);
      static TypeDescription::Register registerMe;
   private:
   static Variable* maker();
   };
   
   template<class T>
      TypeDescription::Register NCVariable<T>::registerMe(getTypeDescription());

   template<class T>
      const TypeDescription*
      NCVariable<T>::getTypeDescription()
      {
	 static TypeDescription* td;
	 if(!td){
	    td = scinew TypeDescription(TypeDescription::NCVariable,
					"NCVariable", &maker,
					fun_getTypeDescription((T*)0));
	 }
	 return td;
      }
   
   template<class T>
      Variable*
      NCVariable<T>::maker()
      {
	 return scinew NCVariable<T>();
      }
   
   template<class T>
      NCVariable<T>::~NCVariable()
      {
      }
   
   template<class T>
      NCVariableBase*
      NCVariable<T>::clone() const
      {
	 NCVariable<T>* tmp=scinew NCVariable<T>(*this);
	 return tmp;
      }
   
   template<class T>
      void
      NCVariable<T>::copyPointer(const NCVariableBase& copy)
      {
	 const NCVariable<T>* c = dynamic_cast<const NCVariable<T>* >(&copy);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in NC variable");
	 *this = *c;
      }

   template<class T>
      NCVariable<T>&
      NCVariable<T>::operator=(const NCVariable<T>& copy)
      {
	 if(this != &copy){
	    Array3<T>::operator=(copy);
	 }
	 return *this;
      }
   
   template<class T>
      NCVariable<T>::NCVariable()
      {
      }
   
   template<class T>
      NCVariable<T>::NCVariable(const NCVariable<T>& copy)
      : Array3<T>(copy)
      {
      }
   
   template<class T>
      void
      NCVariable<T>::allocate(const IntVector& lowIndex,
			      const IntVector& highIndex)
      {
	 if(getWindow())
	    throw InternalError("Allocating an NCvariable that "
				"is apparently already allocated!");
	 resize(lowIndex, highIndex);
      }
   template<class T>
      void
      NCVariable<T>::copyPatch(NCVariableBase* srcptr,
				const IntVector& lowIndex,
				const IntVector& highIndex)
      {
	 const NCVariable<T>* c = dynamic_cast<const NCVariable<T>* >(srcptr);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in NC variable");
	 const NCVariable<T>& src = *c;
	 for(int i=lowIndex.x();i<highIndex.x();i++)
	    for(int j=lowIndex.y();j<highIndex.y();j++)
	       for(int k=lowIndex.z();k<highIndex.z();k++)
		  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
      }
   
   template<class T>
      void
      NCVariable<T>::emit(OutputContext& oc)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 if(td->isFlat()){
	    // This could be optimized...
	    IntVector l(getLowIndex());
	    IntVector h(getHighIndex());
	    for(int z=l.z();z<h.z();z++){
	       for(int y=l.y();y<h.y();y++){
		  ssize_t size = (ssize_t)(sizeof(T)*(h.x()-l.x()));
		  ssize_t s=write(oc.fd, &(*this)[IntVector(l.x(),y,z)], size);
		  if(size != s)
		     throw ErrnoException("NCVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      void*
      NCVariable<T>::getBasePointer()
      {
	 return getPointer();
      }

   template<class T>
      void
      NCVariable<T>::read(InputContext& oc)
      {
	 const TypeDescription* td = fun_getTypeDescription((T*)0);
	 if(td->isFlat()){
	    // This could be optimized...
	    IntVector l(getLowIndex());
	    IntVector h(getHighIndex());
	    for(int z=l.z();z<h.z();z++){
	       for(int y=l.y();y<h.y();y++){
		  ssize_t size = (ssize_t)(sizeof(T)*(h.x()-l.x()));
		  ssize_t s=::read(oc.fd, &(*this)[IntVector(l.x(),y,z)], size);
		  if(size != s)
		     throw ErrnoException("NCVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      const TypeDescription*
      NCVariable<T>::virtualGetTypeDescription() const
      {
	 return getTypeDescription();
      }
   
   template<class T>
      void
      NCVariable<T>::getSizes(IntVector& low, IntVector& high, IntVector& siz,
			      IntVector& strides) const
      {
	 low=getLowIndex();
	 high=getHighIndex();
	 siz=size();
	 strides = IntVector(sizeof(T), (int)(sizeof(T)*siz.x()),
			     (int)(sizeof(T)*siz.y()*siz.x()));
      }
} // End namespace Uintah

#endif
