
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
public:
    Stack(int initial_alloc=0);
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
	ASSERTL3(n>=0 && n<stack.size());
	return stack[n];
    }
};

template<class T>
inline
T& Stack<T>::top() const
{
   ASSERT(stack.size() > 0);
   return stack[stack.size()-1];
}

template<class T>
inline
int Stack<T>::size() const
{
    return stack.size();
}

template<class T>
inline
int Stack<T>::empty() const
{
    return (stack.size()==0);
}

#endif
