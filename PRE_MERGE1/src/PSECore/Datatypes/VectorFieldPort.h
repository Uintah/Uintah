
/*
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldPort_h
#define SCI_project_VectorFieldPort_h 1

#include <CommonDatatypes/SimplePort.h>
#include <CoreDatatypes/VectorField.h>

namespace PSECommon {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

typedef SimpleIPort<VectorFieldHandle> VectorFieldIPort;
typedef SimpleOPort<VectorFieldHandle> VectorFieldOPort;

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:51  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:04  dav
// added back PSECommon .h files
//
// Revision 1.2  1999/04/27 23:18:37  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif
