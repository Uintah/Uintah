
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

#include <SCICore/share/share.h>

#include <SCICore/Thread/Mutex.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace Containers {

class SCICORESHARE TrivialAllocator {
    struct List {
	List* next;
	void* pad; // For 8 byte alignment
    };
    List* freelist;
    List* chunklist;
    unsigned int nalloc;
    unsigned int alloc_size;
    unsigned int size;
    SCICore::Thread::Mutex lock;
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

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/28 17:54:35  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/23 06:30:34  sparker
// Linux port
// Added X11 configuration options
// Removed many warnings
//
// Revision 1.2  1999/08/17 06:38:39  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:15  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:45  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:34  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif
