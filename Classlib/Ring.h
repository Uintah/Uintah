
/*
 *  Ring.h: A static-length ring buffer
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1996
 *
 *  Copyright (C) 1996 SCI Group
 */


#ifndef SCI_Classlib_Ring_h
#define SCI_Classlib_Ring_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <Classlib/Array1.h>

template<class T> class Ring {
    Array1<T> data;
    int _head;
    int _tail;
    int _size;
public:
    inline int size() {return _size;}
    inline int head() {return _head;}
    inline int tail() {return _tail;}
    Ring(int s);
    ~Ring();
    inline T pop() {T item=data[_head]; _head=(_head+1)%_size; return item;}
    inline T top() {return data[_head];}
    inline void push(T item) {data[_tail]=item; _tail=(_tail+1)%_size;}
    inline void swap(T item) {int i=(_tail-1)%_size; T tmp=data[i]; data[i]=item; data[_tail]=tmp; _tail=(_tail+1)%_size;}
};

#endif
