//static char *id="@(#) $Id$";

/*
 *  ColumnMatrixPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Datatypes/ColumnMatrixPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

template<> clString SimpleIPort<ColumnMatrixHandle>::port_type("ColumnMatrix");
template<> clString SimpleIPort<ColumnMatrixHandle>::port_color("dodgerblue4");

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/08/30 20:19:22  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
// Revision 1.3  1999/08/25 03:48:19  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:46  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

