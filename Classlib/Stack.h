
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

#ifdef __GNUG__
#pragma interface
#endif

template<class T> class Stack {
    Array1<T> stack;
    int sp;
    int growsize;
public:
    Stack(int initial_alloc=0, int growsize=100);
    ~Stack();
    void push(const T&);
    void dup();
    T pop();
    void yank(int);
    void remove_all();
    inline T& top() const;
    inline int size() const;
    inline int empty() const;

    // Accesses the nth element of the stack (0=bottom)
    inline T& operator[](int n) const {
	return stack[n];
    }

};

template<class T>
inline
T& Stack<T>::top() const
{
    return stack[sp-1];
}

template<class T>
inline
int Stack<T>::size() const
{
    return sp;
}

template<class T>
inline
int Stack<T>::empty() const
{
    return sp==0;
}

template<class T>
inline void Stack<T>::push(const T& item)
{
    if(sp >= stack.size())
	stack.grow(growsize);
    stack[sp++]=item;
}

template<class T>
inline T Stack<T>::pop()
{
    return stack[--sp];
}

#endif
