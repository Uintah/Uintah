
/*
 *  BooleanPort.h
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_BooleanPort_h
#define SCI_project_BooleanPort_h 1

#include <CommonDatatypes/SimplePort.h>
#include <CoreDatatypes/Boolean.h>

namespace PSECommon {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

typedef SimpleIPort<sciBooleanHandle> sciBooleanIPort;
typedef SimpleOPort<sciBooleanHandle> sciBooleanOPort;

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:45  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:16:59  dav
// added back PSECommon .h files
//
// Revision 1.2  1999/04/25 04:36:35  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif
