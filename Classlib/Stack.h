
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

#ifndef SCI_Classlib_Stack_h
#define SCI_Classlib_Stack_h 1

#include <Classlib/Array1.h>

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

#endif
