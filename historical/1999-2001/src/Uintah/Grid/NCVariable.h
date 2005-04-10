
#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/NCVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Grid/TypeUtils.h>
#include <Uintah/Interface/InputContext.h>
#include <Uintah/Interface/OutputContext.h>
#include <Uintah/Grid/TypeUtils.h>
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
     
     virtual void allocate(const Patch* patch)
     { allocate(patch->getNodeLowIndex(), patch->getNodeHighIndex()); }
   
     virtual void copyPatch(NCVariableBase* src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex);
     NCVariable<T>& operator=(const NCVariable<T>&);
     
     virtual void* getBasePointer();
     virtual const TypeDescription* virtualGetTypeDescription() const;

     virtual void getSizes(IntVector& low, IntVector& high,
			   IntVector& siz, IntVector& strides) const;
     // Replace the values on the indicated face with value
     void fillFace(Patch::FaceType face, const T& value, 
		   IntVector offset = IntVector(0,0,0))
       { 
	 IntVector low,hi;
	 low = getLowIndex() + offset;
	 hi = getHighIndex() - offset;
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
     void fillFaceNormal(Patch::FaceType face, 
			 IntVector offset = IntVector(0,0,0))
       {
	 IntVector low,hi;
	 low = getLowIndex() + offset;
	 hi = getHighIndex() - offset;
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
   
} // end namespace Uintah

//
// $Log$
// Revision 1.34  2000/12/23 00:32:47  witzel
// Added emit(OutputContext), read(InputContext), and allocate(Patch*) as
// pure virtual methods to class Variable and did any needed implementations
// of these in sub-classes.
//
// Revision 1.33  2000/12/20 20:45:13  jas
// Added methods to retriever the interior cell index and use those for
// filling in the bcs for either the extraCells layer or the regular
// domain depending on what the offset is to fillFace and friends.
// MPM requires bcs to be put on the actual boundaries and ICE requires
// bcs to be put in the extraCells.
//
// Revision 1.32  2000/12/10 09:06:16  sparker
// Merge from csafe_risky1
//
// Revision 1.31.2.2  2000/10/10 05:28:08  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.31.2.1  2000/09/29 06:12:29  sparker
// Added support for sending data along patch edges
//
// Revision 1.31  2000/09/28 23:22:01  jas
// Added (int) to remove g++ warnings for STL size().  Reordered initialization
// to coincide with *.h declarations.
//
// Revision 1.30  2000/09/25 20:37:42  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.29  2000/09/25 18:12:19  sparker
// do not use covariant return types due to problems with g++
// other linux/g++ fixes
//
// Revision 1.28  2000/08/08 01:32:46  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.27  2000/07/31 17:45:44  kuzimmer
// Added files and modules for Field Extraction from uda
//
// Revision 1.26  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.25  2000/06/22 21:56:30  sparker
// Changed variable read/write to fortran order
//
// Revision 1.24  2000/05/30 20:19:29  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.23  2000/05/28 17:25:54  dav
// adding code. someone should check to see if i did it corretly
//
// Revision 1.22  2000/05/21 08:19:09  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.21  2000/05/20 08:09:22  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
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
// Made patches have a single uniform index space - still needs work
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
