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

#include <Core/Malloc/Allocator.h>
#include <Core/Malloc/AllocPriv.h>
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

#ifdef __GNUG__
#define THROWCLAUSE throw()
#else
#define THROWCLAUSE
#endif

extern "C" {
void* malloc(size_t size) THROWCLAUSE;
void free(void* ptr) THROWCLAUSE;
void* calloc(size_t, size_t) THROWCLAUSE;
void* realloc(void* p, size_t s) THROWCLAUSE;
void* memalign(size_t, size_t) THROWCLAUSE;
void* valloc(size_t) THROWCLAUSE;
}

using namespace SCIRun;

#ifndef DISABLE_SCI_MALLOC

void* malloc(size_t size) THROWCLAUSE
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "Unknown - malloc");
}

void free(void* ptr) THROWCLAUSE
{
    default_allocator->free(ptr);
}

void* calloc(size_t n, size_t s) THROWCLAUSE
{
    size_t tsize=n*s;
    void* p=malloc(tsize);
    bzero(p, tsize);
    return p;
}

void* realloc(void* p, size_t s) THROWCLAUSE
{
    return default_allocator->realloc(p, s);
}

void* memalign(size_t alignment, size_t size) THROWCLAUSE
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->memalign(alignment, size, "Unknown - memalign");
}

void* valloc(size_t size) THROWCLAUSE
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->memalign(getpagesize(), size,
				       "Unknown - valloc");
}

#endif
