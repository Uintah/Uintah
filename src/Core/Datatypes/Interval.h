
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

#ifndef SCI_CoreDatatypes_Interval_h
#define SCI_CoreDatatypes_Interval_h 1

#include <CoreDatatypes/Datatype.h>
#include <Containers/LockingHandle.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class Interval;
typedef LockingHandle<Interval> IntervalHandle;

class Interval : public Datatype {
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

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
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
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif
