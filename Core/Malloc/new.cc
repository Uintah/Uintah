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

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Malloc/AllocPriv.h>
#include <new>
#include <sci_defs.h>

using namespace SCICore::Malloc;

#ifndef DISABLE_SCI_MALLOC

#ifdef __sgi

// This is ugly, but necessary, since --LANG:std remaps the mangling
// of the global operator new.  This provides the "old" operator new,
// since the compiler insists on generating both.
//
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
