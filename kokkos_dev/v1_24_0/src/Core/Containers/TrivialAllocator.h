/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
