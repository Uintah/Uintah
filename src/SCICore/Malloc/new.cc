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

using namespace SCICore::Malloc;

void* operator new(size_t size)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - operator new");
}

void operator delete(void* ptr)
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

//
// $Log$
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

