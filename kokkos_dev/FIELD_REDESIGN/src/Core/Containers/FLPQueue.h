
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
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace Containers {

using namespace SCICore::PersistentSpace;

template<class T> class FLPQueue;
template<class T> class FLPQueueNode;
template<class T>
void Pio(Piostream& stream, Containers::FLPQueueNode<T>& n);
template<class T>
void Pio(Piostream& stream, Containers::FLPQueue<T>& n);

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
void Pio(Piostream& stream, Containers::FLPQueue<T>& q)
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
void Pio(Piostream& stream, Containers::FLPQueueNode<T>& n)
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

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/09/08 02:26:45  sparker
// Various #include cleanups
//
// Revision 1.5  1999/08/30 20:19:26  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
// Revision 1.4  1999/08/19 05:30:54  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
// Revision 1.3  1999/08/18 20:20:19  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:38:35  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:12  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:35  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:42  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:30  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
