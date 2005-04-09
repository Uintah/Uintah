
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

#include <Classlib/Assert.h>

template<class T> class Queue;

template<class T> class QueueNode {
  T item;
  QueueNode* next;
  inline QueueNode(const T& item, QueueNode* next) : item(item), next(next){}
  friend class Queue<T>;
};

/**************************************

CLASS
   Queue
   
KEYWORDS
   Queue

DESCRIPTION
    Queue.h: A simple FIFO
  
    Written by:
     Steven G. Parker
     Department of Computer Science
     University of Utah
     June 1994
  
    Copyright (C) 1994 SCI Group
PATTERNS
   
WARNING
  
****************************************/

class RigorousTest;

template<class T> class Queue {
    QueueNode<T>* head;
    QueueNode<T>* tail;
    int _length;
public:
    //////////
    //Create a new Queue object
    Queue();

    //////////
    //Class destructor
    ~Queue();

    //////////
    //Append to the Queue
    void append(const T&);

    //////////
    //Pop the first element from the Queue
    T pop();

    //////////
    //Returns 1 if the Queue is empty, 0 if there is still data in the Queue
    bool is_empty();

    //////////
    //Returns the legnth of the Queue
    int length();

    //////////
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);

};

#endif
