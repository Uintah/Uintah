
/*
 *  SigmaSetPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_SigmaSetPort_h
#define SCI_DaveW_Datatypes_SigmaSetPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <DaveW/Datatypes/General/SigmaSet.h>

namespace DaveW {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<SigmaSetHandle> SigmaSetIPort;
typedef SimpleOPort<SigmaSetHandle> SigmaSetOPort;

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/25 03:47:35  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/23 02:53:01  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:07  dmw
// Added and updated DaveW Datatypes/Modules
//
//

#endif
