
/*
 *  AllocOS.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Malloc_AllocOS_h
#define Malloc_AllocOS_h 1

#include <stdlib.h>

namespace SCICore {
namespace Malloc {

struct OSHunk {
    static OSHunk* alloc(size_t size, bool returnable);
    static void free(OSHunk*);
    void* data;
    OSHunk* next;

    int ninuse;
    size_t spaceleft;
    void* curr;
    size_t len;
    bool returnable;
    double align;
};

} // End namespace Malloc
} // End namespace SCICore

//
// $Log$
// Revision 1.1.2.2  2000/10/26 17:38:00  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.3  2000/08/01 00:00:24  sparker
// Added a double to the AllocOS struct to ensure that the memory will
// properly get aligned on a double-word boundary.
//
// Revision 1.2  2000/07/27 07:41:48  sparker
// Distinguish between "returnable" chunks and non-returnable chucks of memory
// Make malloc get along with SGI's MPI
//
// Revision 1.1  1999/07/27 16:56:58  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:52:10  dav
// adding .h files back to src tree
//
// Revision 1.1  1999/05/05 21:05:20  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:24  dav
// Import sources
//
//

#endif
