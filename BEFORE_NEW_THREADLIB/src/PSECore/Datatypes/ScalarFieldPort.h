
/*
 *  ScalarField.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldPort_h
#define SCI_project_ScalarFieldPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/ScalarField.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

typedef SimpleIPort<ScalarFieldHandle> ScalarFieldIPort;
typedef SimpleOPort<ScalarFieldHandle> ScalarFieldOPort;

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:22  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:10  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:49  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:02  dav
// added back PSECore .h files
//
// Revision 1.2  1999/04/27 23:18:36  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif
