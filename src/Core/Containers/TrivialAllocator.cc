
/*
 *  TrivialAllocator.cc:  Template class for allocating small objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Containers/TrivialAllocator.h>
#include <Core/Malloc/Allocator.h>
#include <stdlib.h>

namespace SCIRun {

const int PAGESIZE = 64*1024-64;  // Leave some room for malloc's overhead

TrivialAllocator::TrivialAllocator(unsigned int _size)
: freelist(0), chunklist(0), lock("TrivialAllocator lock")
{
    int word_size=sizeof(void*);
    size=word_size*((_size+(word_size-1))/word_size);
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

} // End namespace SCIRun

