
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
Stack<T>::Stack(int initial_alloc)
: stack(0, initial_alloc)
{
}

template<class T>
Stack<T>::~Stack()
{
}

template<class T>
void Stack<T>::push(const T& item)
{
   stack.add(item);
}

template<class T>
void Stack<T>::dup()
{
   stack.add(stack[stack.size()-1]);
}

template<class T>
T Stack<T>::pop()
{
    ASSERT(stack.size() != 0);
    T item(stack[stack.size()-1]);
    stack.remove(stack.size()-1);
    return item;
}

template<class T>
void Stack<T>::yank(int n)
{
   stack.remove(n);
}

template<class T>
void Stack<T>::remove_all()
{
   stack.remove_all();
}

