
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
Stack<T>::Stack()
{
    stack_top=0;
    _size=0;
}

template<class T>
Stack<T>::~Stack()
{
    StackNode<T>* p=stack_top;
    while(p){
	StackNode<T>* next=p->next;
	delete p;
	p=next;
    }
}

template<class T>
void Stack<T>::push(const T& item)
{
    stack_top=new StackNode<T>(item, stack_top);
    _size++;
}

template<class T>
void Stack<T>::dup()
{
    ASSERT(stack_top != 0);
    stack_top=new StackNode<T>(stack_top->item, stack_top);
    _size++;
}

template<class T>
T Stack<T>::pop()
{
    ASSERT(stack_top != 0);
    StackNode<T>* p=stack_top;
    stack_top=stack_top->next;
    T item(p->item);
    delete p;
    _size--;
    return item;
}

template<class T>
void Stack<T>::yank(int n)
{
    StackNode<T>* p=stack_top;
    for(int i=1;i<n;i++){
	ASSERT(p != 0);
	p=p->next;
    }
    ASSERT(p != 0);
    StackNode<T>* yy=p->next;
    ASSERT(p->next != 0);
    p->next=p->next->next;
    _size--;
    delete yy;
}

template<class T>
T& Stack<T>::top()
{
    ASSERT(stack_top != 0);
    return stack_top->item;
}

template<class T>
int Stack<T>::size()
{
    return _size;
}
