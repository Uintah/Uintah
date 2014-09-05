/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

#if defined(__sun)
#include <string.h>
#define bzero(p,sz)  memset(p,0, sz);
#elif defined(__linux) || defined(__sgi) || defined(__digital__) || defined(_AIX) || defined(__APPLE__)
#else
#error "Need bcopy define for this architecture"
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

static const char* default_tag = "Unknown - malloc";
namespace SCIRun {
const char* AllocatorSetDefaultTagMalloc(const char* tag)
{
  const char* old = default_tag;
  default_tag=tag;
  return old;
}
}

#ifndef DISABLE_SCI_MALLOC

void* malloc(size_t size) THROWCLAUSE
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, default_tag);
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
    if(!default_allocator)
	MakeDefaultAllocator();
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
