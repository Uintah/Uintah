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

#include <Core/Containers/Array1.h>

namespace SCIRun {

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

template<class T> Ring<T>::Ring(int s)
: _head(0), _tail(0), _size(s)
{
    data.resize(s);
}

template<class T> Ring<T>::~Ring()
{
}

} // End namespace SCIRun


#endif
