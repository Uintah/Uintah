
/*
 *  PIDLObjectPort.h
 *  $Id$
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_project_PIDLObjectPort_h
#define SCI_project_PIDLObjectPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <Uintah/Datatypes/Particles/PIDLObject.h>

namespace Uintah {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<PIDLObjectHandle> PIDLObjectIPort;
typedef SimpleOPort<PIDLObjectHandle> PIDLObjectOPort;

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/10/07 02:08:24  sparker
// use standard iostreams and complex type
//
//

#endif
