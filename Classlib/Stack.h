
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

#ifdef __GNUG__
#pragma interface
#endif

template<class T> class Stack;

template<class T> class StackNode {
    T item;
    StackNode* next;
    inline StackNode(const T& item, StackNode* next) : item(item), next(next){}
    friend class Stack<T>;
};

template<class T> class Stack {
    StackNode<T>* stack_top;
    int _size;
public:
    Stack();
    ~Stack();
    void push(const T&);
    void dup();
    T pop();
    void yank(int);
    T& top();
    int size();
};

#endif
