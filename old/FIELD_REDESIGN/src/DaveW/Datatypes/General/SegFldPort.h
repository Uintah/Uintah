
/*
 *  SegFld.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_SegFldPort_h
#define SCI_DaveW_Datatypes_SegFldPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <DaveW/Datatypes/General/SegFld.h>

namespace DaveW {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<SegFldHandle> SegFldIPort;
typedef SimpleOPort<SegFldHandle> SegFldOPort;

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/25 03:47:34  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/23 02:53:00  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:05  dmw
// Added and updated DaveW Datatypes/Modules
//
//

#endif
