#ifndef cMATRIX_H
#define cMATRIX_H 1

/*
 *  cMatrix.h : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <CoreDatatypes/Datatype.h>
#include <Containers/LockingHandle.h>


namespace SCICore {
  namespace Math {
    class Complex;
  }
}

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Math::Complex;
using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class cMatrix;
typedef LockingHandle<cMatrix> cMatrixHandle;

class cVector;
class cMatrix:public Datatype{
  
public:
 virtual cVector operator*( cVector &V)=0;

 virtual void mult(cVector& V,cVector& tmp)=0;
 virtual Complex& get(int row, int col)=0;

// Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
 
  
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:34  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:01  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:52  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:24  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
