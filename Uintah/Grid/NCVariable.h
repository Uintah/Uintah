#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/NCVariableBase.h>
#include <Uintah/Grid/EmitUtils.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Interface/OutputContext.h>
#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/Region.h>
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

   template<class T> class NCVariable : public Array3<T>, public NCVariableBase{
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
     virtual NCVariable<T>* clone() const;
     
     //////////
     // Insert Documentation Here:
     virtual void allocate(const IntVector& lowIndex,
			   const IntVector& highIndex);
     
     virtual void copyRegion(NCVariableBase* src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex);
     NCVariable<T>& operator=(const NCVariable<T>&);
     
     // Replace the values on the indicated face with value
     void fillFace(Region::FaceType face, Vector value)
       { 
	 IntVector low = getLowIndex();
	 IntVector hi = getHighIndex();
	 switch (face) {
	 case Region::xplus:
	   for (int j = low.y(); j<hi.y(); j++) {
	     for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(hi.x()-1,j,k)] = value;
	     }
	   }
	   break;
	 case Region::xminus:
	   for (int j = low.y(); j<hi.y(); j++) {
	     for (int k = low.z(); k<hi.z(); k++) {
	       (*this)[IntVector(low.x(),j,k)] = value;
	     }
	   }
	   break;
	 case Region::yplus:
	   for (int i = low.x(); i<hi.x(); i++) {
	     for (int k = low.z(); k<hi.z(); k++) {
	       (*this)[IntVector(i,hi.y()-1,k)] = value;
	     }
	   }
	   break;
	 case Region::yminus:
	   for (int i = low.x(); i<hi.x(); i++) {
	     for (int k = low.z(); k<hi.z(); k++) {
	       (*this)[IntVector(i,low.y(),k)] = value;
	     }
	   }
	   break;
	 case Region::zplus:
	   for (int i = low.x(); i<hi.x(); i++) {
	     for (int j = low.y(); j<hi.y(); j++) {
	       (*this)[IntVector(i,j,hi.z()-1)] = value;
	     }
	   }
	   break;
	 case Region::zminus:
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
     void fillFaceNormal(Region::FaceType face)
       {
	 IntVector low = getLowIndex();
	 IntVector hi = getHighIndex();
	 switch (face) {
	 case Region::xplus:
	   for (int j = low.y(); j<hi.y(); j++) {
	     for (int k = low.z(); k<hi.z(); k++) {
		(*this)[IntVector(hi.x()-1,j,k)] =
		Vector(0.0,(*this)[IntVector(hi.x()-1,j,k)].y(),
				(*this)[IntVector(hi.x()-1,j,k)].z());
	     }
	   }
	   break;
	 case Region::xminus:
	   for (int j = low.y(); j<hi.y(); j++) {
	     for (int k = low.z(); k<hi.z(); k++) {
	       (*this)[IntVector(low.x(),j,k)] = 
		Vector(0.0,(*this)[IntVector(low.x(),j,k)].y(),
				(*this)[IntVector(low.x(),j,k)].z());
	     }
	   }
	   break;
	 case Region::yplus:
	   for (int i = low.x(); i<hi.x(); i++) {
	     for (int k = low.z(); k<hi.z(); k++) {
	       (*this)[IntVector(i,hi.y()-1,k)] =
		Vector((*this)[IntVector(i,hi.y()-1,k)].x(),0.0,
				(*this)[IntVector(i,hi.y()-1,k)].z());
	     }
	   }
	   break;
	 case Region::yminus:
	   for (int i = low.x(); i<hi.x(); i++) {
	     for (int k = low.z(); k<hi.z(); k++) {
	       (*this)[IntVector(i,low.y(),k)] =
		Vector((*this)[IntVector(i,low.y(),k)].x(),0.0,
				(*this)[IntVector(i,low.y(),k)].z());
	     }
	   }
	   break;
	 case Region::zplus:
	   for (int i = low.x(); i<hi.x(); i++) {
	     for (int j = low.y(); j<hi.y(); j++) {
	       (*this)[IntVector(i,j,hi.z()-1)] =
		Vector((*this)[IntVector(i,j,hi.z()-1)].x(),
				(*this)[IntVector(i,j,hi.z()-1)].y(),0.0);
	     }
	   }
	   break;
	 case Region::zminus:
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
   private:
   };
   
   template<class T>
      const TypeDescription*
      NCVariable<T>::getTypeDescription()
      {
	 static TypeDescription* td;
	 if(!td)
	    td = new TypeDescription(false, TypeDescription::Node);
	 return td;
      }
   
   template<class T>
      NCVariable<T>::~NCVariable()
      {
      }
   
   template<class T>
      NCVariable<T>*
      NCVariable<T>::clone() const
      {
	 return new NCVariable<T>(*this);
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
	    throw InternalError("Allocating an NCvariable that is apparently already allocated!");
	 resize(lowIndex, highIndex);
      }
   template<class T>
      void
      NCVariable<T>::copyRegion(NCVariableBase* srcptr,
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
	 T* t=0;
	 if(isFlat(*t)){
	    // This could be optimized...
	    IntVector l(getLowIndex());
	    IntVector h(getHighIndex());
	    for(int x=l.x();x<h.x();x++){
	       for(int y=l.y();y<h.y();y++){
		  size_t size = sizeof(T)*(h.z()-l.z());
		  ssize_t s=write(oc.fd, &(*this)[IntVector(x,y,l.z())], size);
		  if(size != s)
		     throw ErrnoException("NCVariable::emit (write call)", errno);
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
// Revision 1.20  2000/05/19 00:37:29  guilkey
// Tested and fixed fillFaceNormal.
//
// Revision 1.19  2000/05/18 23:01:28  guilkey
// Filled in fillFaceNormal.  Haven't tested it yet, but will soon.
//
// Revision 1.18  2000/05/17 00:37:16  guilkey
// Fixed fillFace function.
//
// Revision 1.17  2000/05/15 19:39:48  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.16  2000/05/10 20:03:00  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.15  2000/05/10 19:56:46  guilkey
// D'ohhh!!!  Got a little carried away with my last "correction".  fillFace
// should be right now.
//
// Revision 1.14  2000/05/10 19:39:48  guilkey
// Fixed the fillFace function, it was previously reading and writing out
// of bounds.
//
// Revision 1.13  2000/05/09 23:43:07  jas
// Filled in fillFace.  It is probably slow as mud but hopefully the gist
// is right.
//
// Revision 1.12  2000/05/09 03:24:39  jas
// Added some enums for grid boundary conditions.
//
// Revision 1.11  2000/05/07 06:02:12  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.10  2000/05/04 19:06:47  guilkey
// Added the beginnings of grid boundary conditions.  Functions still
// need to be filled in.
//
// Revision 1.9  2000/05/02 06:07:21  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.8  2000/04/26 06:48:49  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/20 18:56:30  sparker
// Updates to MPM
//
// Revision 1.6  2000/04/12 23:00:48  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.5  2000/04/11 07:10:50  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.4  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.3  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
