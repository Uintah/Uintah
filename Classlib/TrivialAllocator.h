
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

#include <Multitask/ITC.h>

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
public:
    TrivialAllocator(unsigned int size);
    ~TrivialAllocator();

    void* alloc();
    void free(void*);
};

#endif
