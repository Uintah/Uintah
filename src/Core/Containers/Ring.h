
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


#ifndef SCI_Containers_Ring_h
#define SCI_Containers_Ring_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <Containers/Array1.h>

namespace SCICore {
namespace Containers {

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

} // End namespace Containers
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included Ring.cc
//

namespace SCICore {
namespace Containers {

template<class T> Ring<T>::Ring(int s)
: _head(0), _tail(0), _size(s)
{
    data.resize(s);
}

template<class T> Ring<T>::~Ring()
{
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:14  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:44  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:33  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif
