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
 *  Stack.h: A simple LIFO
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Stack_h
#define SCI_Containers_Stack_h 1

#include <Core/Containers/Array1.h>

namespace SCIRun {

template<class T> class Stack {
    Array1<T> stack;
    int sp;
    int growsize;
public:
    Stack(int initial_alloc=0, int growsize=100);
    ~Stack();
    inline void push(const T& item) {
	if(sp >= stack.size())
	    stack.grow(growsize);
	stack[sp++]=item;
    }
    inline T pop() {
	return stack[--sp];
    }
    void dup();
    void yank(int);
    void remove_all();
    inline const T& top() const {
	return stack[sp-1];
    }
    inline T& top() {
	return stack[sp-1];
    }
    inline int size() const {
	return sp;
    }
    inline int empty() const {
	return sp==0;
    }

    // Accesses the nth element of the stack (0=bottom)
    inline const T& operator[](int n) const {
	return stack[n];
    }
};

} // End namespace SCIRun

////////////////////////////////////////////////////////////
// Start of included Stack.cc

#include <Core/Util/Assert.h>

namespace SCIRun {

template<class T>
Stack<T>::Stack(int initial_alloc, int growsize)
: stack(0, initial_alloc), sp(0), growsize(growsize)
{
}

template<class T>
Stack<T>::~Stack()
{
}

template<class T>
void Stack<T>::dup()
{
    push(top());
}

template<class T>
void Stack<T>::yank(int n)
{
    stack.remove(n);
    sp--;
}

template<class T>
void Stack<T>::remove_all()
{
    sp=0;
}

} // End namespace SCIRun


#endif
