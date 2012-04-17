/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <sci_defs/malloc_defs.h>

#include <Core/Malloc/Allocator.h>


#include <Core/Malloc/AllocPriv.h>
#include <Core/Malloc/mem_init.h>

#if defined(__sun) || defined(_WIN32)
#  include <cstring>
#  define bzero(p,sz)  memset(p,0, sz);
#elif defined(__linux) || defined(__sgi) || defined(__digital__) || defined(_AIX) || defined(__APPLE__) || defined(__CYGWIN__)
//  do nothing
#else
#  error "Need bcopy define for this architecture"
#endif

#ifndef _WIN32
// irix64 KCC stuff
#  include <strings.h>
#  ifdef __GNUG__
#    define THROWCLAUSE throw()
#  else
#    define THROWCLAUSE
#  endif
#else
#  define THROWCLAUSE
#endif

#ifdef MALLOC_TRACE
#  include "MallocTraceOff.h"
#endif

#ifdef MALLOC_TRACE
#  include "MallocTraceOn.h"
#endif

#ifndef DISABLE_SCI_MALLOC

using namespace SCIRun;

static const char* default_malloc_tag = "Unknown - malloc";
extern int default_tag_line_number;  // defined in new.cc

namespace SCIRun {

  const char*
  AllocatorSetDefaultTagMalloc(const char* tag)
  {
    const char* old = default_malloc_tag;
    default_malloc_tag=tag;
    return old;
  }

  void
  AllocatorResetDefaultTagMalloc()
  {
    default_malloc_tag = "Unknown - malloc";
  }

} // end namespace SCIRun

void*
malloc(size_t size) THROWCLAUSE
{
  if(!default_allocator)
    MakeDefaultAllocator();
  void* mem=default_allocator->alloc(size, default_malloc_tag, default_tag_line_number);
#ifdef INITIALIZE_MEMORY
  for(unsigned int i=0;i<size;i++) {
    static_cast<unsigned char*>(mem)[i] = MEMORY_INIT_NUMBER;
  }
#endif
  return mem;
}

void
free(void* ptr) THROWCLAUSE
{
  default_allocator->free(ptr);
}

void*
calloc(size_t n, size_t s) THROWCLAUSE
{
  size_t tsize=n*s;
  void* p=malloc(tsize);
  bzero(p, tsize);
  return p;
}

void*
realloc(void* p, size_t s) THROWCLAUSE
{
  if(!default_allocator) {
    MakeDefaultAllocator();
  }
  return default_allocator->realloc(p, s);
}

void*
memalign(size_t alignment, size_t size) THROWCLAUSE
{
  if(!default_allocator) {
    MakeDefaultAllocator();
  }
  return default_allocator->memalign(alignment, size, "Unknown - memalign");
}

void*
valloc(size_t size) THROWCLAUSE
{
  if(!default_allocator) {
    MakeDefaultAllocator();
  }
  return default_allocator->memalign(getpagesize(), size, "Unknown - valloc");
}

#endif
