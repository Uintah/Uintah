#ifndef UINTAH_HOMEBREW_SFCYVARIABLE_H
#define UINTAH_HOMEBREW_SFCYVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/SFCYVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
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

using namespace Uintah;

namespace Uintah {
   using SCICore::Exceptions::ErrnoException;
   using SCICore::Exceptions::InternalError;
   using SCICore::Geometry::Vector;

   class TypeDescription;

/**************************************

CLASS
   SFCYVariable
   
GENERAL INFORMATION

   SFCYVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCYVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class SFCYVariable : public Array3<T>, public SFCYVariableBase {
   public:
     
     SFCYVariable();
     SFCYVariable(const SFCYVariable<T>&);
     virtual ~SFCYVariable();
     
     virtual void rewindow(const IntVector& low, const IntVector& high);
     //////////
     // Insert Documentation Here:
     static const TypeDescription* getTypeDescription();
     
     virtual void copyPointer(const SFCYVariableBase&);
     
     //////////
     // Insert Documentation Here:
     virtual SFCYVariableBase* clone() const;
     
     //////////
     // Insert Documentation Here:
     virtual void allocate(const IntVector& lowIndex,
			   const IntVector& highIndex);
     
     virtual void copyPatch(SFCYVariableBase* src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex);
     SFCYVariable<T>& operator=(const SFCYVariable<T>&);
     
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
      TypeDescription::Register SFCYVariable<T>::registerMe(getTypeDescription());

   template<class T>
      const TypeDescription*
      SFCYVariable<T>::getTypeDescription()
      {
	 static TypeDescription* td;
	 if(!td){
	    td = scinew TypeDescription(TypeDescription::SFCYVariable,
				     "SFCYVariable", &maker,
				     fun_getTypeDescription((T*)0));
	 }
	 return td;
      }
   
   template<class T>
      Variable*
      SFCYVariable<T>::maker()
      {
	 return scinew SFCYVariable<T>();
      }
   
   template<class T>
      SFCYVariable<T>::~SFCYVariable()
      {
      }
   
   template<class T>
      SFCYVariableBase*
      SFCYVariable<T>::clone() const
      {
	 return scinew SFCYVariable<T>(*this);
      }
   
   template<class T>
      void
      SFCYVariable<T>::copyPointer(const SFCYVariableBase& copy)
      {
	 const SFCYVariable<T>* c = dynamic_cast<const SFCYVariable<T>* >(&copy);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in SFCY variable");
	 *this = *c;
      }

   template<class T>
      SFCYVariable<T>&
      SFCYVariable<T>::operator=(const SFCYVariable<T>& copy)
      {
	 if(this != &copy){
	    Array3<T>::operator=(copy);
	 }
	 return *this;
      }
   
   template<class T>
      SFCYVariable<T>::SFCYVariable()
      {
      }
   
   template<class T>
      SFCYVariable<T>::SFCYVariable(const SFCYVariable<T>& copy)
      : Array3<T>(copy)
      {
      }
   
   template<class T>
      void
      SFCYVariable<T>::allocate(const IntVector& lowIndex,
			      const IntVector& highIndex)
      {
	if(getWindow())
	  throw InternalError("Allocating an SFCYvariable that "
			      "is apparently already allocated!");
	 resize(lowIndex, highIndex);
      }
   template<class T>
      void SFCYVariable<T>::rewindow(const IntVector& low,
				   const IntVector& high) {
      Array3<T> newdata;
      newdata.resize(low, high);
      newdata.copy(*this, low, high);
      resize(low, high);
      Array3<T>::operator=(newdata);
   }
   template<class T>
      void
      SFCYVariable<T>::copyPatch(SFCYVariableBase* srcptr,
				const IntVector& lowIndex,
				const IntVector& highIndex)
      {
	 const SFCYVariable<T>* c = dynamic_cast<const SFCYVariable<T>* >(srcptr);
	 if(!c)
	    throw TypeMismatchException("Type mismatch in SFCY variable");
	 const SFCYVariable<T>& src = *c;
	 for(int i=lowIndex.x();i<highIndex.x();i++)
	    for(int j=lowIndex.y();j<highIndex.y();j++)
	       for(int k=lowIndex.z();k<highIndex.z();k++)
		  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
      }
   
   template<class T>
      void
      SFCYVariable<T>::emit(OutputContext& oc)
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
		     throw ErrnoException("SFCYVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      void*
      SFCYVariable<T>::getBasePointer()
      {
	 return getPointer();
      }

   template<class T>
      void
      SFCYVariable<T>::read(InputContext& oc)
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
		     throw ErrnoException("SFCYVariable::emit (write call)", errno);
		  oc.cur+=size;
	       }
	    }
	 } else {
	    throw InternalError("Cannot yet write non-flat objects!\n");
	 }
      }

   template<class T>
      const TypeDescription*
      SFCYVariable<T>::virtualGetTypeDescription() const
      {
	 return getTypeDescription();
      }
   
   template<class T>
     void
     SFCYVariable<T>::getSizes(IntVector& low, IntVector& high, 
			       IntVector& siz) const
     {
       low = getLowIndex();
       high = getHighIndex();
       siz = size();
     }

} // end namespace Uintah

//
// $Log$
// Revision 1.9  2000/12/05 16:05:48  jas
// Added BC fillFace and other BC related things needed for ICE.
//
// Revision 1.8  2000/10/12 20:05:37  sparker
// Removed print statement from FCVariable
// Added rewindow to SFC{X,Y,Z}Variables
// Uncommented assertion in CCVariable
//
// Revision 1.7  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.6  2000/09/25 18:12:20  sparker
// do not use covariant return types due to problems with g++
// other linux/g++ fixes
//
// Revision 1.5  2000/09/25 14:41:32  rawat
// added mpi support for cell centered and staggered cell variables
//
// Revision 1.3  2000/08/08 01:32:47  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.2  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.1  2000/06/27 23:18:18  rawat
// implemented Staggered cell variables. Modified Patch.cc to get ghostcell
// and staggered cell indexes.
//
//

#endif
