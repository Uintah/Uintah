/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  TrivialAllocator.h:  class for allocating small objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 */

#ifndef sci_Containers_TrivialAllocator_h
#define sci_Containers_TrivialAllocator_h 1

#include <Core/Thread/Mutex.h>
#include <Core/Malloc/Allocator.h>



#include <Core/Containers/share.h>

namespace SCIRun {

class TrivialAllocator {
    struct List {
	List* next;
	void* pad; // For 8 byte alignment
    };
    List* freelist;
    List* chunklist;
    unsigned long nalloc;
    unsigned long alloc_size;
    unsigned long size;
    Mutex lock;
    int ta_disable;
public:
    SCISHARE TrivialAllocator(unsigned int size);
    SCISHARE ~TrivialAllocator();

    inline void* alloc();
#ifdef MALLOC_TRACE
#  include "MallocTraceOff.h"
#endif 
    inline void free(void*);
#ifdef MALLOC_TRACE
#  include "MallocTraceOn.h"
#endif 
};

inline void* TrivialAllocator::alloc()
{
    if(ta_disable){
	return new char[size];
    }
    lock.lock();
    List* p=freelist;
    if(!p){
	List* q=(List*)scinew char[alloc_size];
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
    lock.unlock();
    return p;
}

#ifdef MALLOC_TRACE
#  include "MallocTraceOff.h"
#endif 

inline void TrivialAllocator::free(void* rp)
{
#ifdef MALLOC_TRACE
#  include "MallocTraceOn.h"
#endif 
    if(ta_disable){
	delete[] (char*)rp;
	return;
    }
    lock.lock();
    List* p=(List*)rp;
    p->next=freelist;
    freelist=p;
    lock.unlock();
}

} // End namespace SCIRun

#endif
