/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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


#ifndef SCI_Containers_Queue_h
#define SCI_Containers_Queue_h 1

#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>

namespace SCIRun {

class RigorousTest;
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
  int is_empty();

  //////////
  //Returns the legnth of the Queue
  int length();

  //////////
  //Rigorous Tests
  static void test_rigorous(RigorousTest* __test);

};

template<class T> Queue<T>::Queue()
  : head(0), tail(0), _length(0)
{
}

template<class T> Queue<T>::~Queue()
{
  QueueNode<T>* p=head;
  while(p){
    QueueNode<T>* todelete=p;
    p=p->next;
    delete todelete;
  }
}

template<class T> void Queue<T>::append(const T& item)
{
  QueueNode<T>* p=scinew QueueNode<T>(item, 0);
  if(tail){
    tail->next=p;
    tail=p;
  } else {
    head=tail=p;
  }
  _length++;
}

template<class T>
T
Queue<T>::pop()
{
  ASSERT(head != 0);
  T item=head->item;
  QueueNode<T>* oldhead=head;
  head=head->next;
  if(!head)tail=0;
  delete oldhead;
  _length--;
  return item;
}

template<class T>
int
Queue<T>::is_empty()
{
  return head==0;
}

template<class T> 
int
Queue<T>::length()
{
  return _length;
}

} // End namespace SCIRun


#endif
