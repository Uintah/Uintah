
/*
 *  TrivialAllocator.h:  class for allocating small objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_Classlib_TrivialAllocator_h
#define sci_Classlib_TrivialAllocator_h 1

#include <Core/Thread/Mutex.h>

//namespace rtrt {

using SCIRun::Mutex;

class TrivialAllocator {
    struct List {
	List* next;
	void* pad; // For 8 byte alignment
    };
    List* freelist;
    List* chunklist;
    unsigned int nalloc;
    unsigned int alloc_size;
    unsigned int size;
    Mutex lock;
    int ta_disable;
public:
    TrivialAllocator(unsigned int size);
    ~TrivialAllocator();

    inline void* alloc();
    inline void free(void*);
};

inline void* TrivialAllocator::alloc()
{
    if(ta_disable){
	return new char[size];
    }
    //lock.lock();
    List* p=freelist;
    if(!p){
	List* q=(List*)new char[alloc_size];
	q->next=chunklist;
	chunklist=q;
	q++;
	p=q;
	for(unsigned int i=1;i<nalloc;i++){
	    List* n=(List*)(((char*)q)+size);
	    q->next=n;
	    q=n;
	}
	q->next=0;
    }
    freelist=p->next;
    //lock.unlock();
    return p;
}

inline void TrivialAllocator::free(void* rp)
{
    if(ta_disable){
	delete[] rp;
	return;
    }
    //lock.lock();
    List* p=(List*)rp;
    p->next=freelist;
    freelist=p;
    //lock.unlock();
}

//} // end namespace rtrt

#endif
