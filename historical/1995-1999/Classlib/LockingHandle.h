
/*
 *  LockingHandle.h: Smart Pointers..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Classlib_LockingHandle_h
#define SCI_Classlib_LockingHandle_h 1

#include <Classlib/Assert.h>
class Piostream;

template<class T>
class LockingHandle {
    T* rep;
public:
    LockingHandle();
    LockingHandle(T*);
    LockingHandle(const LockingHandle<T>&);
    LockingHandle<T>& operator=(const LockingHandle<T>&);
    LockingHandle<T>& operator=(T*);
    ~LockingHandle();

    void detach();

    inline T* operator->() const {
	ASSERT(rep != 0);
	return rep;
    }
    inline T* get_rep() const {
	return rep;
    }

    friend void Pio(Piostream&, LockingHandle<T>& data);
};

template<class T> void Pio(Piostream&, LockingHandle<T>& data);

#endif
