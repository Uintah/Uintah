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
 *  FLPQueue.h: A fixed length priority queue
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_Containers_FLPQueue_h
#define SCI_Containers_FLPQueue_h 1

#include <sci_config.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


template<class T> class FLPQueue;
template<class T> class FLPQueueNode;
template<class T>
void Pio(Piostream& stream, FLPQueueNode<T>& n);
template<class T>
void Pio(Piostream& stream, FLPQueue<T>& n);

template<class T> class FLPQueueNode {
  T item;
  double w;
  FLPQueueNode* next;
  FLPQueueNode* prev;
  inline FLPQueueNode(const T& item, FLPQueueNode* next, FLPQueueNode* prev, double w) : item(item), next(next), prev(prev), w(w){}
  inline FLPQueueNode(const T& item, double w): item(item), next(0), prev(0), w(w){}
  friend class FLPQueue<T>;
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, FLPQueueNode<T>&);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, FLPQueue<T>&);
};

template<class T> class FLPQueue {
    FLPQueueNode<T>* head;
    FLPQueueNode<T>* tail;
    int _length;
    int _size;
public:
    FLPQueue(int size);
    ~FLPQueue();
    T pop(double &w);
    int is_empty();
    int length();
    void sanity_check();
    // insert retuns true if something was bumped from the q, and false if not
    int insert(const T& item, double weight, int& caused_bump, T& bumped);
    void update_weight(const T&, double weight);
    friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, FLPQueue<T>&);
};

template<class T> FLPQueue<T>::FLPQueue(int size)
: head(0), tail(0), _length(0), _size(size)
{
}

template<class T> FLPQueue<T>::~FLPQueue()
{
}

template<class T> int FLPQueue<T>::insert(const T& item, double w, 
					  int& caused_bump, T& bumped)
{
//    cerr << "Putting in weight: "<<w<<"... ";
//    if (item.i == 9 && item.j == 7 && item.k == 7)
//	cerr << "Inserting ("<<item.i<<", "<<item.j<<", "<<item.k<<")  l="<<_length<<"\n";
//	ASSERT(0);
    caused_bump=0;
    FLPQueueNode<T>* link=scinew FLPQueueNode<T>(item, 0, 0, w);
    if (_length<_size || w<tail->w) {

	if (!head) {
	    head=tail=link;
	} else {
	    if (w<head->w) {
		link->next = head;
		head->prev = link;
		head = link;
	    } else {
		FLPQueueNode<T>* tmp=head;
		for(;;){
		    if (!tmp->next) {
			link->prev = tmp;
			tmp->next = link;
			tail = link;
			break;
		    } else {			
			if (tmp->next->w > w) {
			    link->next = tmp->next;
			    tmp->next->prev = link;
			    tmp->next = link;
			    link->prev = tmp;
			    break;
			}
		    }
		    tmp = tmp->next;
		}
	    }
	}
//	cerr << "w="<<link->w<<"\n";
	if (_length<_size){_length++; sanity_check(); return 1;}
	caused_bump=1;
	bumped = tail->item;

//	cerr << "Bumping ("<<bumped.i<<", "<<bumped.j<<", "<<bumped.k<<")\n";
	tail=tail->prev;
	delete(tail->next);
	tail->next=0;
//	tail->prev->next = 0;
//	delete(tail);
	sanity_check();
	return 1;
    } else {
	sanity_check();
        return 0;
    }
}

template<class T> void FLPQueue<T>::update_weight(const T& item, double w) {
    FLPQueueNode<T>* tmp = head;
    while (tmp != 0 && !item.isequal(tmp->item)) tmp=tmp->next;
    ASSERT(tmp != 0);
    if (tmp->prev && w < tmp->prev->w) {
	FLPQueueNode<T> *link = tmp;
	link->item = item;
	link->w = w;
	// new path is closer - need to move this link in pq
	if (tmp->next) {
	    tmp->next->prev = tmp->prev; 
	    tmp = tmp->prev;
	    tmp->next = tmp->next->next;
	} else {
	    tmp = tmp->prev; 
	    tmp->next = 0; 
	    tail=tmp;
	}
	while (tmp->prev && w<tmp->prev->w) 
	    tmp=tmp->prev;
	if (!tmp->prev) {
	    link->next=head;
	    link->prev=0;
	    head->prev=link;
	    head = link;
	} else {
	    link->prev=tmp->prev;
	    link->next=tmp;
	    tmp->prev->next=link;
	    tmp->prev=link;
	}
    }
    sanity_check();
}

template<class T> T FLPQueue<T>::pop(double &w)
{
    ASSERT(head != 0);
    T item=head->item;
//    if (item.i == 8 && item.j == 11 && item.k == 8)
//	cerr << "Popping ("<<item.i<<", "<<item.j<<", "<<item.k<<")\n";
    w=head->w;
    FLPQueueNode<T>* oldhead=head;
    head=head->next;
    if(!head)tail=0;
    else head->prev=0;
    delete oldhead;
    _length--;
    return item;
}

template<class T>
int
FLPQueue<T>::is_empty()
{
    return head==0;
}

template<class T>
int
FLPQueue<T>::length()
{
    return _length;
}


template<class T> void FLPQueue<T>::sanity_check() {
    return;
}

#if 0
template<class T> void FLPQueue<T>::sanity_check() {
    if (head == tail && tail == 0) return;
    FLPQueueNode<T>* tmp=head;
    for (int i=0; i<_length-1; i++) {
	ASSERT(tmp); 
	tmp=tmp->next;
    }
    ASSERT(tmp==tail);
    for (i=0; i<_length-1; i++) {
	ASSERT(tmp); 
	tmp=tmp->prev;
    }
    ASSERT(tmp==head);
    ASSERT(head->prev == 0);
    ASSERT(tail->next == 0);
}
#endif

#define FLPQUEUE_VERSION 1

template<class T>
void Pio(Piostream& stream, FLPQueue<T>& q)
{
    if (stream.reading()) return;
    /* int version= */stream.begin_class("FLPQueue", FLPQUEUE_VERSION);
    int length=q._length;
    Pio(stream, length);
    FLPQueueNode<T> *tmp=q.head;
    for(int i=0;i<q._length;i++, tmp=tmp->next)
        Pio(stream, *tmp);
    stream.end_class();
}

#define FLPQUEUENODE_VERSION 1

template<class T>
void Pio(Piostream& stream, FLPQueueNode<T>& n)
{
    if (stream.reading()) return;
    /* int version= */stream.begin_class("FLPQueueNode", FLPQUEUENODE_VERSION);
    Pio(stream, n.w);
    int i=(int) n.prev;
    Pio(stream, i);
    i=(int)&n;
    Pio(stream, i);
    i=(int)n.next;
    Pio(stream, i);
    Pio(stream, n.item);
    stream.end_class();
}

} // End namespace SCIRun


#endif
