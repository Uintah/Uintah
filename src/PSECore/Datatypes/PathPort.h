
/*
 *  PathPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Novemeber 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_PSECore_Datatypes_PathPort_h
#define SCI_PSECore_Datatypes_PathPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/Path.h>

namespace PSECore {
namespace Datatypes {

using SCICore::Datatypes::PathHandle;

typedef SimpleIPort<PathHandle> PathIPort;
typedef SimpleOPort<PathHandle> PathOPort;

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.1.2.1  2000/09/28 03:14:20  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.1  2000/07/19 06:35:50  samsonov
// PathPort datatype moved from DaveW
//
// Revision 1.1  1999/12/02 21:57:30  dmw
// new camera path datatypes and modules
//
//

#endif
