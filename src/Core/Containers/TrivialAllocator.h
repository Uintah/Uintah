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

#ifndef sci_Containers_TrivialAllocator_h
#define sci_Containers_TrivialAllocator_h 1

#include <Core/share/share.h>

#include <Core/Thread/Mutex.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

class SCICORESHARE TrivialAllocator {
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

inline void TrivialAllocator::free(void* rp)
{
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
