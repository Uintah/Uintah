
/*
 *  VoidStar.h: Just has a rep member -- other trivial classes can inherit
 *		from this, rather than having a full-blown datatype and data-
 *		port for every little thing that comes along...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_CoreDatatypes_VoidStar_h
#define SCI_CoreDatatypes_VoidStar_h 1

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Multitask/ITC.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class VoidStar;
typedef LockingHandle<VoidStar> VoidStarHandle;

class SCICORESHARE VoidStar : public Datatype {
protected:
    VoidStar();
public:
    VoidStar(const VoidStar& copy);
    virtual ~VoidStar();
    virtual VoidStar* clone()=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:00  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:33  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:00  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:51  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:23  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
