//static char *id="@(#) $Id$";

/*
 *  malloc.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Malloc/AllocPriv.h>

// irix64 KCC stuff
#include <strings.h>

#ifdef __sun
  #include <string.h>
  #define bzero(p,sz)  memset(p,0, sz);
#else
  #ifndef __linux
    #include <bstring.h>
  #endif
#endif

extern "C" {
void* malloc(size_t size);
void free(void* ptr);
void* calloc(size_t, size_t);
void* realloc(void* p, size_t s);
void* memalign(size_t, size_t);
void* valloc(size_t);
}

namespace SCICore {
namespace Malloc {

void* malloc(size_t size)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "Unknown - malloc");
}

void free(void* ptr)
{
    default_allocator->free(ptr);
}

void* calloc(size_t n, size_t s)
{
    size_t tsize=n*s;
    void* p=malloc(tsize);
    bzero(p, tsize);
    return p;
}

void* realloc(void* p, size_t s)
{
    return default_allocator->realloc(p, s);
}

void* memalign(size_t, size_t)
{
    AllocError("memalign not finished");
    return 0;
}

void* valloc(size_t)
{
    AllocError("valloc not finished");
    return 0;
}

} // End namespace Malloc
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/23 06:30:37  sparker
// Linux port
// Added X11 configuration options
// Removed many warnings
//
// Revision 1.2  1999/08/17 06:39:31  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:59  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:24  dav
// Import sources
//
//
