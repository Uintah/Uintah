
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

#include <CommonDatatypes/SimplePort.h>
#include <CoreDatatypes/ScalarField.h>

namespace PSECommon {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

typedef SimpleIPort<ScalarFieldHandle> ScalarFieldIPort;
typedef SimpleOPort<ScalarFieldHandle> ScalarFieldOPort;

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:49  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:02  dav
// added back PSECommon .h files
//
// Revision 1.2  1999/04/27 23:18:36  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif
