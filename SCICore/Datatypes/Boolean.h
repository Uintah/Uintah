
/*
 *  sciBoolean.h: Specification of a range [x..y]
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_CoreDatatypes_sciBoolean_h
#define SCI_CoreDatatypes_sciBoolean_h 1

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class sciBoolean;
typedef LockingHandle<sciBoolean> sciBooleanHandle;

class SCICORESHARE sciBoolean : public Datatype {
public:
    int value;
    sciBoolean(int value);
    virtual ~sciBoolean();
    sciBoolean(const sciBoolean&);
    virtual sciBoolean* clone() const;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:19  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:46  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:35  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:03  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//

#endif
