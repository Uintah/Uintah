
/*
 *  Interval.h: Specification of a range [x..y]
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_Datatypes_Interval_h
#define SCI_Datatypes_Interval_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class Interval;
typedef LockingHandle<Interval> IntervalHandle;

class SCICORESHARE Interval : public Datatype {
public:
    double low, high;
    Interval(double low, double high);
    virtual ~Interval();
    Interval(const Interval&);
    virtual Interval* clone() const;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:34  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:22  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:48  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:39  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:08  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif
