
/*
 *  Datatype.h: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Datatype_h
#define SCI_project_Datatype_h 1

#include <SCICore/share/share.h>

#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Thread/Mutex.h>

namespace SCICore {
namespace Datatypes {

using SCICore::PersistentSpace::Persistent;
using SCICore::Containers::clString;

class SCICORESHARE Datatype : public Persistent {
public:
    int ref_cnt;
    SCICore::Thread::Mutex lock;
    int generation;
    Datatype();
    Datatype(const Datatype&);
    virtual ~Datatype();
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/28 17:54:36  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/25 03:48:32  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:45  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:20  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:47  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:37  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:06  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif /* SCI_project_Datatype_h */
