//static char *id="@(#) $Id$";

/*
 *  ScaledBoxWidgetDataPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Datatypes/ScaledBoxWidgetDataPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

clString SimpleIPort<ScaledBoxWidgetDataHandle>::port_type("Data");
clString SimpleIPort<ScaledBoxWidgetDataHandle>::port_color("black");

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:23  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:11  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:49  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//
