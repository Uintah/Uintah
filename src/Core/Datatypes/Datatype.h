
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

#include <Persistent/Persistent.h>
#include <Multitask/ITC.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::PersistentSpace::Persistent;
using SCICore::Multitask::Mutex;
using SCICore::Containers::clString;

class Datatype : public Persistent {
public:
    int ref_cnt;
    Mutex lock;
    int generation;
    Datatype();
    virtual ~Datatype();
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
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
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif /* SCI_project_Datatype_h */
