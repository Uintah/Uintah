/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */


#ifndef SCI_Containers_Ring_h
#define SCI_Containers_Ring_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <vector>

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
