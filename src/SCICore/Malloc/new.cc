//static char *id="@(#) $Id$";

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

using namespace SCICore::Malloc;

#if 1

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


//
// $Log$
// Revision 1.4  2000/02/24 06:04:55  sparker
// 0xffff5a5a (NaN) is now the fill pattern
// Added #if 1 to malloc/new.cc to make it easier to turn them on/off
//
// Revision 1.3  1999/08/31 23:26:13  sparker
// Updates to fix operator new support on SGI with -LANG:std
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

