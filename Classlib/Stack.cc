
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

#include <Classlib/Stack.h>
#include <Classlib/Assert.h>

#ifdef __GNUG__
#pragma interface
#endif

template<class T>
Stack<T>::Stack(int initial_alloc, int growsize)
: stack(0, initial_alloc), growsize(growsize), sp(0)
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
