//static char *id="@(#) $Id$";

/*
  stack.h
  Templated stack implementation

  Philip Sutton


  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __TBON_STACK_H__
#define __TBON_STACK_H__

#include <iostream>

namespace Phil {
namespace Modules {

template <class T>
class Stack {
public:
  Stack( int size=100, int growsize=20 );
  ~Stack();

  void push( T item );
  T& pop();
  int empty() { return (top == 0); }
  void reset() { top = 0; }
  void print() {
    cout << "top = " << top << endl;
    for( int i = top-1; i >= 0; i-- )
      objs[i].print();
  }
protected:
private:
  T* objs;
  int top;
  int nalloc;
  int gsize;
}; // class Stack

template <class T>
Stack<T>::Stack( int size, int growsize ) : nalloc(size), gsize(growsize) {
  top = 0;
  objs = new T[nalloc];
} // Stack

template <class T>
Stack<T>::~Stack() {
  delete [] objs;
} // ~Stack

template <class T>
void
Stack<T>::push( T item ) {
  if( top == nalloc ) {
    T* newobjs = new T[nalloc+gsize];
    for( int i = 0; i < top; i++ )
      newobjs[i] = objs[i];
    delete [] objs;
    objs = newobjs;
    nalloc += gsize;
  }
  objs[top++] = item;
} // push

template <class T>
T&
Stack<T>::pop() {
  if( top == 0 ) {
    cerr << "Error: stack underflow" << endl;
    return;
  }
  return objs[--top];
} // pop


} // end namespace Modules
} // end namespace Phil

#endif

//
// $Log$
// Revision 1.2  2000/02/04 22:15:09  psutton
// fixed ID problem
//
// Revision 1.1  2000/02/04 21:11:59  psutton
// initial revision
//
//
