
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

#ifdef __GNUG__
#pragma interface
#endif

template<class T> Queue<T>::Queue()
{
    head=tail=0;
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
    QueueNode<T>* p=new QueueNode<T>(item, 0);
    if(tail){
	tail->next=p;
	tail=p;
    } else {
	head=tail=p;
    }
}

template<class T> T Queue<T>::pop()
{
    ASSERT(head != 0);
    T item=head->item;
    QueueNode<T>* oldhead=head;
    head=head->next;
    if(!head)tail=0;
    delete oldhead;
    return item;
}

template<class T> Queue<T>::is_empty()
{
    return head==0;
}
