
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

#include <SCICore/Util/Assert.h>

namespace SCICore {

namespace Tester {
  class RigorousTest;
}

namespace Containers {

using SCICore::Tester::RigorousTest;

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

} // End namespace Containers
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included Queue.cc
//

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Tester/RigorousTest.h>

namespace SCICore {
namespace Containers {

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


#include <SCICore/Tester/RigorousTest.h>


template<class T>
void Queue<T>::test_rigorous(RigorousTest* __test)
{
  // Dd: I don't understand how this test_rigorous is supposed to work
  //     It is getting instantiated (at least under g++) for T == char *
  //     This causes the code not to compile because we are tryping
  //     to compare "char *"s (returned from pop() with ints.
#if 0
    //Test the queue when holding ints
    Queue x;
    
    int cnt=0;

    int z;
    for( z=1;z<=1000;z++ )
    {
	x.append(z);
	++cnt;
	TEST(x.length()==z);
	TEST(x.is_empty()==0);
    }

    for( z=1;z<=1000;z++ )
    {
	TEST(x.is_empty()==0);
	TEST(x.pop()==z);
	--cnt;
	TEST(x.length()==cnt);
    }

    TEST(x.is_empty()==1);
    TEST(x.length()==0);

    //Test the queue when holding floats

    Queue<float> f;

    TEST(f.is_empty()==1);

    cnt=0;
    float fcnt;

    for( fcnt=1.1;fcnt<=1000.1;fcnt+=1.0 )
    {
	f.append(fcnt);
	TEST(f.is_empty()==0);
	++cnt;
	TEST(f.length()==cnt);
    }

    for( fcnt=1.1;fcnt<=1000.1;fcnt+=1.0 )
    {
	TEST(f.pop()==fcnt);
	--cnt;
	TEST(f.length()==cnt);
    }

    TEST(f.length()==0);
    TEST(f.is_empty()==1);

    //Test the queue with char* variables
    Queue<char*> cq;
    
    TEST(cq.length()==0);
    TEST(cq.is_empty()==1);
    
    cq.append("Hello");
    
    TEST(cq.length()==1);
    TEST(cq.is_empty()==0);

    cq.append("There");

    TEST(cq.length()==2);
    TEST(cq.is_empty()==0);

    TEST(cq.pop()=="Hello");
    
    TEST(cq.length()==1);
    TEST(cq.is_empty()==0);

    TEST(cq.pop()=="There");
     
    TEST(cq.length()==0);
    TEST(cq.is_empty()==1);
#endif
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/19 23:52:59  sparker
// Removed extraneous includes of iostream.h  Fixed a few NotFinished.h
// problems.  May have broken KCC support.
//
// Revision 1.2  1999/08/17 06:38:37  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:13  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:36  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:44  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:32  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif
