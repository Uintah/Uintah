
/*
 *  ImagePort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ImagePort_h
#define SCI_project_ImagePort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/Image.h>

namespace SCIRun {
namespace Datatypes {

using namespace PSECore::Datatypes;
using namespace SCICore::Datatypes;

typedef SimpleIPort<ImageHandle> ImageIPort;
typedef SimpleOPort<ImageHandle> ImageOPort;

} // End namespace Datatypes
} // End namespace SCIRun

//
// $Log$
// Revision 1.3  1999/08/31 08:55:30  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.2  1999/08/25 03:48:54  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/07/27 16:58:47  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 22:25:36  dav
// trying to update all
//
// Revision 1.1  1999/04/29 21:50:58  dav
// moved ImagePort datatype out of common and into scirun
//
// Revision 1.2  1999/04/27 23:18:35  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
