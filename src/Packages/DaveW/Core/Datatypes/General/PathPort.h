
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

#ifndef SCI_DaveW_Datatypes_PathPort_h
#define SCI_DaveW_Datatypes_PathPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <DaveW/Datatypes/General/Path.h>

namespace DaveW {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<PathHandle> PathIPort;
typedef SimpleOPort<PathHandle> PathOPort;

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/12/02 21:57:30  dmw
// new camera path datatypes and modules
//
//

#endif
