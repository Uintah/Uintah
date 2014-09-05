
/*
 *  ParticleSetPort.h
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_project_ParticleSetPort_h
#define SCI_project_ParticleSetPort_h 1

#include <PSECore/Datatypes/SimplePort.h>

#include <Uintah/Datatypes/Particles/ParticleSet.h>

namespace Uintah {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<ParticleSetHandle> ParticleSetIPort;
typedef SimpleOPort<ParticleSetHandle> ParticleSetOPort;

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.3  1999/08/25 03:49:03  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:09  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:59:01  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 23:18:40  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif
