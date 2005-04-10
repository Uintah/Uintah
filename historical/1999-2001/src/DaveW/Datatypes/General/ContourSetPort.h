
/*
 *  ContourSetPort.h: The ContourSetPort datatype
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_ContourSetPort_h
#define SCI_DaveW_Datatypes_ContourSetPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <DaveW/Datatypes/General/ContourSet.h>

namespace DaveW {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<ContourSetHandle> ContourSetIPort;
typedef SimpleOPort<ContourSetHandle> ContourSetOPort;

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 05:27:35  dmw
// more DaveW datatypes...
//
//

#endif
