#ifndef UINTAH_HOMEBREW_ARRAY3DATA_H
#define UINTAH_HOMEBREW_ARRAY3DATA_H

#include "RefCounted.h"
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Malloc/Allocator.h>

/**************************************

CLASS
   Array3Data
   
GENERAL INFORMATION

   Array3Data.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3Data

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

namespace Uintah {
   using SCICore::Geometry::IntVector;
   template<class T> class Array3Data : public RefCounted {
   public:
      Array3Data(const IntVector& size);
      virtual ~Array3Data();
      
      inline IntVector size() const {
	 return d_size;
      }
      void initialize(const T& val, const IntVector& s, const IntVector& e);
      inline T& get(const IntVector& idx) {
	 CHECKARRAYBOUNDS(idx.x(), 0, d_size.x());
	 CHECKARRAYBOUNDS(idx.y(), 0, d_size.y());
	 CHECKARRAYBOUNDS(idx.z(), 0, d_size.z());
	 return d_data3[idx.x()][idx.y()][idx.z()];
      }
   private:
      T*    d_data;
      T***  d_data3;
      IntVector d_size;
      
      Array3Data& operator=(const Array3Data&);
      Array3Data(const Array3Data&);
   };
   
   template<class T>
      void Array3Data<T>::initialize(const T& val,
				     const IntVector& lowIndex,
				     const IntVector& highIndex)
      {
	 CHECKARRAYBOUNDS(lowIndex.x(), 0, d_size.x());
	 CHECKARRAYBOUNDS(lowIndex.y(), 0, d_size.y());
	 CHECKARRAYBOUNDS(lowIndex.z(), 0, d_size.z());
	 CHECKARRAYBOUNDS(highIndex.x(), lowIndex.x(), d_size.x()+1);
	 CHECKARRAYBOUNDS(highIndex.y(), lowIndex.y(), d_size.y()+1);
	 CHECKARRAYBOUNDS(highIndex.z(), lowIndex.z(), d_size.z()+1);
	 T* d = &d_data3[lowIndex.x()][lowIndex.y()][lowIndex.z()];
	 IntVector s = highIndex-lowIndex;
	 for(int i=0;i<s.x();i++){
	    T* dd=d;
	    for(int j=0;j<s.y();j++){
	       T* ddd=dd;
	       for(int k=0;k<s.z();k++)
		  ddd[k]=val;
	       dd+=d_size.z();
	    }
	    d+=d_size.z()*d_size.y();
	 }
      }
   
   template<class T>
      Array3Data<T>::Array3Data(const IntVector& size)
      : d_size(size)
      {
	 if(d_size.x() && d_size.y() && d_size.z())
	    d_data=scinew T[d_size.x()*d_size.y()*d_size.z()];
	 else
	    d_data=0;
	 d_data3=scinew T**[d_size.x()];
	 d_data3[0]=scinew T*[d_size.x()*d_size.y()];
	 d_data3[0][0]=d_data;
	 for(int i=1;i<d_size.x();i++){
	    d_data3[i]=d_data3[i-1]+d_size.y();
	 }
	 for(int j=1;j<d_size.x()*d_size.y();j++){
	    d_data3[0][j]=d_data3[0][j-1]+d_size.z();
	 }
      }
   
   template<class T>
      Array3Data<T>::~Array3Data()
      {
	 if(d_data){
	    delete[] d_data;
	    delete[] d_data3[0];
	    delete[] d_data3;
	 }
      }
}


//
// $Log$
// Revision 1.9  2000/05/30 20:19:27  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.8  2000/05/10 20:02:58  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.7  2000/05/05 04:10:04  guilkey
// Added an include file to fix a compilation problem.  Thanks Steve.
//
// Revision 1.6  2000/04/26 06:48:46  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/03/22 23:41:27  sparker
// Working towards getting arches to compile/run
//
// Revision 1.4  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.3  2000/03/21 01:29:42  dav
// working to make MPM stuff compile successfully
//
// Revision 1.2  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
