//static char *id="@(#) $Id$";

/*
 *  sciBoolean.cc: All this for true and false...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <SCICore/CoreDatatypes/Boolean.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream.h>

namespace SCICore {
namespace CoreDatatypes {

static Persistent* maker()
{
    return scinew sciBoolean(0);
}

PersistentTypeID sciBoolean::type_id("Boolean", "Datatype", maker);

sciBoolean::sciBoolean(int value)
: value(value)
{
}

sciBoolean::sciBoolean(const sciBoolean& c)
: value(c.value)
{
}

sciBoolean::~sciBoolean()
{
}

sciBoolean* sciBoolean::clone() const
{
    return scinew sciBoolean(*this);
}

#define BOOLEAN_VERSION 1

void sciBoolean::io(Piostream& stream)
{
    stream.begin_class("Boolean", BOOLEAN_VERSION);
    stream.io(value);
    stream.end_class();
}

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
// Revision 1.1  1999/04/25 04:07:02  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//
