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

#include <Malloc/Allocator.h>
#include <Malloc/AllocPriv.h>
#include <new>

using namespace SemotusVisum::Malloc;

#ifndef DISABLE_SCI_MALLOC

#ifdef __sgi

// This is ugly, but necessary, since --LANG:std remaps the mangling
// of the global operator new.  This provides the "old" operator new,
// since the compiler insists on generating both.
extern "C" {
    void* __nw__GUi(size_t size)
    {
	if(!default_allocator)
	    MakeDefaultAllocator();
	return default_allocator->alloc(size, "unknown - operator new");
    }

    void* __nwa__GUi(size_t size)
    {
	if(!default_allocator)
	    MakeDefaultAllocator();
	return default_allocator->alloc(size, "unknown - operator new");
    }
}
#endif

void* operator new(size_t size) throw(std::bad_alloc)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - operator new");
}

void* operator new[](size_t size) throw(std::bad_alloc)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - operator new[]");
}

void* operator new(size_t size, const std::nothrow_t&) throw()
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - nothrow operator new");
}

void* operator new[](size_t size, const std::nothrow_t&) throw()
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - nothrow operator new[]");
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

void* operator new(size_t size, Allocator* a, char* tag)
{
    if(!a){
	if(!default_allocator)
	    MakeDefaultAllocator();
	a=default_allocator;
    }
    return a->alloc(size, tag);
}

// Dd: Assuming _BOOL should be defined...
//#ifdef _BOOL
void* operator new[](size_t size, Allocator* a, char* tag)
{
    if(!a){
	if(!default_allocator)
	    MakeDefaultAllocator();
	a=default_allocator;
    }
    return a->alloc(size, tag);
}
//#endif
#else

void* operator new(size_t size, Allocator* a, char* tag)
{
    return new char[size];
}

// Dd: Assuming _BOOL should be defined...
//#ifdef _BOOL
void* operator new[](size_t size, Allocator* a, char* tag)
{
    return new char[size];
}

#endif
