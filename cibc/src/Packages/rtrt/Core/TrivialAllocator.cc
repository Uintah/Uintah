

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

#include <Packages/rtrt/Core/TrivialAllocator.h>
#include <stdlib.h>

#define PAGESIZE (256*1024-64)  // Leave some room for malloc's overhead

TrivialAllocator::TrivialAllocator(unsigned int size)
: freelist(0), chunklist(0), size(size), lock("Trivial allocator lock")
{
    nalloc=(PAGESIZE-sizeof(List))/size;
    alloc_size=nalloc*size+sizeof(List);
    if(getenv("SCI_TA_DISABLE"))
	ta_disable=1;
    else
	ta_disable=0;
}

TrivialAllocator::~TrivialAllocator()
{
    for(List* p=chunklist;p!=0;){
	List* tofree=p;
	p=p->next;
	delete[] tofree;
    }
}

