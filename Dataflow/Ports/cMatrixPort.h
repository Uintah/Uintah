
/*
 *  cMatrixPort.h
 *
 *  Written by:
 *   Leonid Zhukov
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_cMatrixPort_h
#define SCI_project_cMatrixPort_h 1

#include <CommonDatatypes/SimplePort.h>
#include <CoreDatatypes/cMatrix.h>

namespace PSECommon {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

typedef SimpleIPort<cMatrixHandle> cMatrixIPort;
typedef SimpleOPort<cMatrixHandle> cMatrixOPort;

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
// Revision 1.2  1999/04/27 23:18:38  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif
