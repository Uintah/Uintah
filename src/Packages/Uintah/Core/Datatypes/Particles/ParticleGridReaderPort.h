#ifndef SCI_project_ParticleGridReaderPort_h
#define SCI_project_ParticleGridReaderPort_h 1

/*----------------------------------------------------------------------
CLASS
    ParticleGridReaderPort

    
OVERVIEW TEXT
    


KEYWORDS

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 8, 1999
----------------------------------------------------------------------*/

#include <PSECore/Datatypes/SimplePort.h>

#include <Uintah/Datatypes/Particles/ParticleGridReader.h>

namespace Uintah {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<ParticleGridReaderHandle> ParticleGridReaderIPort;
typedef SimpleOPort<ParticleGridReaderHandle> ParticleGridReaderOPort;

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.3  1999/08/25 03:49:02  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:59:00  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 23:18:39  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif
