
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

#endif
