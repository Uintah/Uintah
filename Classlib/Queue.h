
/*
 *  Queue.h: A simple FIFO
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Classlib_Queue_h
#define SCI_Classlib_Queue_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <Classlib/Assert.h>

template<class T> class Queue;

template<class T> class QueueNode {
  T item;
  QueueNode* next;
  inline QueueNode(const T& item, QueueNode* next) : item(item), next(next){}
  friend class Queue<T>;
};

template<class T> class Queue {
    QueueNode<T>* head;
    QueueNode<T>* tail;
    int _length;
public:
    Queue();
    ~Queue();
    void append(const T&);
    T pop();
    int is_empty();
    int length();
    static void test_rigorous(RigorousTest* __test);

};

#endif







