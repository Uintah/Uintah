
/*
 *  ScaledBoxWidgetDataPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScaledBoxWidgetDataHandlePort_h
#define SCI_project_ScaledBoxWidgetDataHandlePort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <PSECore/Datatypes/ScaledBoxWidgetData.h>

namespace PSECore {
namespace Datatypes {

typedef SimpleIPort<ScaledBoxWidgetDataHandle> ScaledBoxWidgetDataIPort;
typedef SimpleOPort<ScaledBoxWidgetDataHandle> ScaledBoxWidgetDataOPort;

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
// Revision 1.1  1999/07/27 16:55:50  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:03  dav
// added back PSECore .h files
//
// Revision 1.2  1999/04/27 23:18:37  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif
