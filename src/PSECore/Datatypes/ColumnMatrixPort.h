
/*
 *  ColumnMatrixPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColumnMatrixPort_h
#define SCI_project_ColumnMatrixPort_h 1

#include <PSECore/CommonDatatypes/SimplePort.h>
#include <SCICore/CoreDatatypes/ColumnMatrix.h>

namespace PSECore {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

typedef SimpleIPort<ColumnMatrixHandle> ColumnMatrixIPort;
typedef SimpleOPort<ColumnMatrixHandle> ColumnMatrixOPort;

} // End namespace CommonDatatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:46  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:00  dav
// added back PSECore .h files
//
// Revision 1.2  1999/04/27 23:18:34  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif

