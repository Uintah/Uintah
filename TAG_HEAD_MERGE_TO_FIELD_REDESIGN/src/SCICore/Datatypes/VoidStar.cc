//static char *id="@(#) $Id$";

/*
 *  VoidStar.cc: Just has a rep member -- other trivial classes can inherit
 *		 from this, rather than having a full-blown datatype and data-
 *		 port for every little thing that comes along...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <SCICore/Datatypes/VoidStar.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
namespace Datatypes {

PersistentTypeID VoidStar::type_id("VoidStar", "Datatype", 0);

VoidStar::VoidStar()
{
}

VoidStar::VoidStar(const VoidStar& /*copy*/)
{
    NOT_FINISHED("VoidStar::VoidStar");
}

VoidStar::~VoidStar()
{
}

#define VoidStar_VERSION 2
void VoidStar::io(Piostream& stream) {
    int version=stream.begin_class("VoidStar", VoidStar_VERSION);
    if (version < 2) {
	if (stream.reading()) {
	    int rep;
	    PersistentSpace::Pio(stream, rep);
	}
    }
    stream.end_class();
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/09/08 02:26:49  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/25 03:48:47  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:00  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:33  mcq
// Initial commit
//
// Revision 1.2  1999/05/06 19:56:00  dav
// added back .h files
//
// Revision 1.1  1999/04/25 04:07:22  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

