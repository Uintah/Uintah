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
#include <sci_defs.h>

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

using namespace SCICore::Malloc;

#ifndef DISABLE_SCI_MALLOC

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

void* memalign(size_t alignment, size_t size)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->memalign(alignment, size, "Unknown - memalign");
}

void* valloc(size_t size)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->memalign(getpagesize(), size,
				       "Unknown - valloc");
}

#endif

//
// $Log$
// Revision 1.8  2000/09/14 15:51:02  sparker
// Include sci_defs.h for DISABLE_SCI_MALLOC
//
// Revision 1.7  2000/09/14 15:34:21  sparker
// Use --disable-sci-malloc configure flag
//
// Revision 1.6  2000/02/24 06:04:55  sparker
// 0xffff5a5a (NaN) is now the fill pattern
// Added #if 1 to malloc/new.cc to make it easier to turn them on/off
//
// Revision 1.5  1999/09/30 00:33:46  sparker
// Added support for memalign (got orphaned in move from SCIRun)
//
// Revision 1.4  1999/09/01 05:34:30  sparker
// malloc/free shouldn't be in the SCICore::Malloc namespace
//
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
