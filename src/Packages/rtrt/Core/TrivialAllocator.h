/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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
      // I think this should be cast to char* since the memory was
      // allocated as an array of chars. --James Bigler
	delete[] (char*)(rp);
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
