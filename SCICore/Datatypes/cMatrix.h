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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <complex>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class cMatrix;
typedef LockingHandle<cMatrix> cMatrixHandle;

class cVector;
class SCICORESHARE cMatrix:public Datatype{
  
public:
    typedef std::complex<double> Complex;
 virtual cVector operator*( cVector &V)=0;

 virtual void mult(cVector& V,cVector& tmp)=0;
 virtual Complex& get(int row, int col)=0;

// Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
 
  
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:35  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/25 03:48:48  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
