//static char *id="@(#) $Id$";

/*
 *  Interval.cc: Specification of a range [x..y]
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <CoreDatatypes/Interval.h>
#include <Containers/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

namespace SCICore {
namespace CoreDatatypes {

static Persistent* maker()
{
    return scinew Interval(0, 1);
}

PersistentTypeID Interval::type_id("Interval", "Datatype", maker);

Interval::Interval(double low, double high)
: low(low), high(high)
{
}

Interval::Interval(const Interval& c)
: low(c.low), high(c.high)
{
}

Interval::~Interval()
{
}

Interval* Interval::clone() const
{
    return scinew Interval(*this);
}

#define INTERVAL_VERSION 1

void Interval::io(Piostream& stream)
{
    stream.begin_class("Interval", INTERVAL_VERSION);
    stream.io(low);
    stream.io(high);
    stream.end_class();
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:22  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:07  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//
