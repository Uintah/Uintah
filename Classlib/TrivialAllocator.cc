

/*
 *  TrivialAllocator.h:  Template class for allocating small objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/TrivialAllocator.h>
#include <Malloc/Allocator.h>

#define PAGESIZE 8176  // Leave some room for malloc's overhead

TrivialAllocator::TrivialAllocator(unsigned int size)
: freelist(0), chunklist(0), size(size)
{
    nalloc=(PAGESIZE-sizeof(List))/size;
    alloc_size=nalloc*size+sizeof(List);
}

TrivialAllocator::~TrivialAllocator()
{
    for(List* p=chunklist;p!=0;){
	List* tofree=p;
	p=p->next;
	delete[] tofree;
    }
}

