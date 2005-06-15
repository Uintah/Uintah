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
 *  new.cc: ?
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
#include <new>

using namespace SCIRun;

#ifdef __sgi

// This is ugly, but necessary, since --LANG:std remaps the mangling
// of the global operator new.  This provides the "old" operator new,
// since the compiler insists on generating both.
extern "C" {
    void* __nw__GUi(size_t size)
    {
	if(!default_allocator)
	    MakeDefaultAllocator();
	return default_allocator->alloc(size, "unknown - operator new", 0);
    }

    void* __nwa__GUi(size_t size)
    {
	if(!default_allocator)
	    MakeDefaultAllocator();
	return default_allocator->alloc(size, "unknown - operator new", 0);
    }
}
#endif

static const char* default_new_tag = "Unknown - operator new";
static const char* default_new_array_tag = "Unknown - operator new[]";
namespace SCIRun {
const char* AllocatorSetDefaultTagNew(const char* tag)
{
  const char* old = default_new_tag;
  default_new_tag=tag;
  return old;
}

void AllocatorResetDefaultTagNew()
{
  default_new_tag = "Unknown - operator new";
  default_new_array_tag = "Unknown - operator new[]";
}

const char* AllocatorSetDefaultTag(const char* tag)
{
  AllocatorSetDefaultTagMalloc(tag);
  return AllocatorSetDefaultTagNew(tag);
}

void AllocatorResetDefaultTag()
{
  AllocatorResetDefaultTagMalloc();
  AllocatorResetDefaultTagNew();
}
}

#ifndef DISABLE_SCI_MALLOC



void* operator new(size_t size) throw(std::bad_alloc)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, default_new_tag, 0);
}

void* operator new[](size_t size) throw(std::bad_alloc)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, default_new_array_tag, 0);
}

void* operator new(size_t size, const std::nothrow_t&) throw()
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - nothrow operator new", 0);
}

void* operator new[](size_t size, const std::nothrow_t&) throw()
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - nothrow operator new[]", 0);
}

void operator delete(void* ptr) throw()
{
    if(!default_allocator)
	MakeDefaultAllocator();
    default_allocator->free(ptr);
}

void operator delete[](void* ptr) throw()
{
    if(!default_allocator)
	MakeDefaultAllocator();
    default_allocator->free(ptr);
}

void* operator new(size_t size, Allocator* a, char* tag, int linenum)
{
    if(!a){
	if(!default_allocator)
	    MakeDefaultAllocator();
	a=default_allocator;
    }
    return a->alloc(size, tag, linenum);
}

void* operator new[](size_t size, Allocator* a, char* tag, int linenum)
{
    if(!a){
	if(!default_allocator)
	    MakeDefaultAllocator();
	a=default_allocator;
    }
    return a->alloc(size, tag, linenum);
}
#else

void* operator new(size_t size, Allocator*, char*, int)
{
    return new char[size];
}

void* operator new[](size_t size, Allocator*, char*, int)
{
    return new char[size];
}

#endif
