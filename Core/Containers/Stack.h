
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

#include <Containers/Array1.h>

namespace SCICore {
namespace Containers {

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
    inline T& top() const {
	return stack[sp-1];
    }
    inline int size() const {
	return sp;
    }
    inline int empty() const {
	return sp==0;
    }

    // Accesses the nth element of the stack (0=bottom)
    inline T& operator[](int n) const {
	return stack[n];
    }
};

} // End namespace Containers
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included Stack.cc
//

#include <Util/Assert.h>

namespace SCICore {
namespace Containers {

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
// Revision 1.1  1999/05/05 21:04:34  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
