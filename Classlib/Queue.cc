
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

#include <Classlib/Queue.h>
#include <Malloc/Allocator.h>

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

template<class T> T Queue<T>::pop()
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
bool
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


#include <Tester/RigorousTest.h>
//#include <iostream.h>


