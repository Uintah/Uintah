//static char *id="@(#) $Id$";

/*
 *  Datatype.cc: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/CoreDatatypes/Datatype.h>

namespace SCICore {
namespace CoreDatatypes {

Mutex generation_lock;
int current_generation;

Datatype::Datatype() {
    ref_cnt=0;
    generation_lock.lock();
    generation=++current_generation;
    generation_lock.unlock();
}

Datatype::~Datatype() {
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:45  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:20  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:06  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
