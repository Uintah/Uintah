
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

#ifdef __GNUG__
#pragma interface
#endif

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

template<class T> Queue<T>::is_empty()
{
    return head==0;
}

template<class T> Queue<T>::length()
{
    return _length;
}


#include <Tester/RigorousTest.h>
//#include <iostream.h>


template<class T>
void Queue<T>::test_rigorous(RigorousTest* __test)
{
    //Test the queue when holding ints
    Queue x;
    
    int cnt=0;

    for (int z=1;z<=1000;z++)
    {
	x.append(z);
	++cnt;
	TEST(x.length()==z);
	TEST(x.is_empty()==0);
    }
    
    

    for (z=1;z<=1000;z++)
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

    for(float fcnt=1.1;fcnt<=1000.1;fcnt+=1.0)
    {
	f.append(fcnt);
	TEST(f.is_empty()==0);
	++cnt;
	TEST(f.length()==cnt);
    }

    for(fcnt=1.1;fcnt<=1000.1;fcnt+=1.0)
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


}













