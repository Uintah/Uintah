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

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  using std::vector;


template<class T> class Ring {
    vector<T> data;
    int head_;
    int tail_;
    int size_;
public:
    inline int size() {return size_;}
    inline int head() {return head_;}
    inline int tail() {return tail_;}
    Ring(int s);
    ~Ring();
    inline T pop() {T item=data_[head_]; head_=(head_+1)%size_; return item;}
    inline T top() {return data_[head_];}
    inline void push(T item) {data_[tail_]=item; tail_=(tail_+1)%size_;}
    inline void swap(T item) {int i=(tail_-1)%size_; T tmp=data_[i]; data_[i]=item; data_[tail_]=tmp; tail_=(tail_+1)%size_;}
};

template<class T> Ring<T>::Ring(int s)
  : data_(s), head_(0), tail_(0), size_(s)
{
}

template<class T> Ring<T>::~Ring()
{
}

} // End namespace SCIRun


#endif
