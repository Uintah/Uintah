

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

#define PAGESIZE 8176

TrivialAllocator::TrivialAllocator(unsigned int size)
: size(size), freelist(0), chunklist(0)
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

void* TrivialAllocator::alloc()
{
    List* p=freelist;
    if(!p){
	List* q=(List*)new char[alloc_size];
	q->next=chunklist;
	chunklist=q;
	q++;
	p=q;
	for(int i=1;i<nalloc;i++){
	    List* n=(List*)(((char*)q)+size);
	    q->next=n;
	    q=n;
	}
	q->next=0;
    }
    freelist=p->next;
    return p;
}

void TrivialAllocator::free(void* rp)
{
    List* p=(List*)rp;
    p->next=freelist;
    freelist=p;
}
