
/*
 *  Mailbox: Threadsafe FIFO
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_Mailbox_h
#define SCICore_Thread_Mailbox_h

/**************************************
 
CLASS
   Mailbox
   
KEYWORDS
   Thread, FIFO
   
DESCRIPTION
   A thread-safe, fixed-length FIFO queue which allows multiple
   concurrent senders and receivers.  Multiple threads send <b>Item</b>s
   to the mailbox, and multiple thread may receive <b>Item</b>s from the
   mailbox.  Items are typically pointers to a message structure.
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Thread.h>
#include <vector>

namespace SCICore {
    namespace Thread {
	template<class Item> class Mailbox {
	public:
	    //////////
	    // Create a mailbox with a maximum queue size of <i>size</i>
	    // items. If size is zero, then the mailbox will use
	    // <i>rendevous semantics</i>, where a sender will block
	    // until a reciever is waiting for the item.  The item will
	    // be handed off synchronously. <i>name</i> should be a
	    // static string which describes the primitive for debugging
	    // purposes.
	    Mailbox(const char* name, int size);
    
	    //////////
	    // Destroy the mailbox.  All items in the queue are silently
	    // dropped.
	    ~Mailbox();
    
	    //////////
	    // Puts <i>msg</i> in the queue.  If the queue is full, the
	    // thread will be blocked until there is room in the queue.
	    // Messages from the same thread will be placed in the
	    // queue in a first-in/first out order. Multiple threads may
	    // call <i>send</i> concurrently, and the messages will be
	    // placed in the queue in an arbitrary order.
	    void send(const Item& msg);
    
	    //////////
	    // Attempt to send <i>msg</i> to the queue.  If the queue is
	    // full, the thread will not be blocked, and <i>trySend</i>
	    // will return false.  Otherwise, <i>trySend</i> will return
	    // true.  This may never complete if the reciever only uses
	    // <i>tryRecieve</i>.  
	    bool trySend(const Item& msg);
    
	    //////////
	    // Receive an item from the queue.  If the queue is empty,
	    // the thread will block until another thread sends an item.
	    // Multiple threads may call <i>recieve</i> concurrently, but
	    // no guarantee is made as to which thread will recieve the
	    // next token.  However, implementors should give preference
	    // to the thread that has been waiting the longest.
	    Item receive();

	    //////////
	    // Attempt to recieve <i>item</i> from the mailbox.  If the
	    // queue is empty, the thread is blocked and <i>tryRecieve</i>
	    // will return false.  Otherwise, <i>tryRecieve</i> returns true.
	    bool tryReceive(Item& item);
    
	    //////////
	    // Return the maximum size of the mailbox queue, as given in the
	    // constructor.
	    int size() const;

	    //////////
	    // Return the number of items currently in the queue.
	    int numItems() const;

	private:
	    const char* d_name;
	    Mutex d_mutex;
	    vector<Item> d_ring_buffer;
	    int d_head;
	    int d_len;
	    int d_max;
	    ConditionVariable d_empty;
	    ConditionVariable d_full;
	    Semaphore d_rendezvous;
	    int d_send_wait;
	    int d_recv_wait;
	    inline int ringNext(int inc);

	    // Cannot copy them
	    Mailbox(const Mailbox<Item>&);
	    Mailbox<Item> operator=(const Mailbox<Item>&);
	};
    }
}

template<class Item> inline
int
SCICore::Thread::Mailbox<Item>::ringNext(int inc)
{
    return d_max==0?0:((d_head+inc)%d_max);
}

template<class Item>
SCICore::Thread::Mailbox<Item>::Mailbox(const char* name, int size)
    : d_name(name), d_mutex("Mailbox lock"), d_ring_buffer(size==0?1:size),
      d_empty("Mailbox empty condition"), d_full("Mailbox full condition"),
      d_rendezvous("Mailbox rendezvous semaphore", 0)
{
    d_head=0;
    d_len=0;
    d_send_wait=0;
    d_recv_wait=0;
    d_max=size;
}

template<class Item>
SCICore::Thread::Mailbox<Item>::~Mailbox()
{
}

template<class Item>
void
SCICore::Thread::Mailbox<Item>::send(const Item& msg)
{
    int s=Thread::couldBlock(d_name);
    d_mutex.lock();
    // See if the message buffer is full...
    int rmax=d_max==0?1:d_max;
    while(d_len == rmax){
        d_send_wait++;
        d_full.wait(d_mutex);
        d_send_wait--;
    }
    d_ring_buffer[ringNext(d_len)]=msg;
    d_len++;
    if(d_recv_wait)
        d_empty.conditionSignal();
    d_mutex.unlock();
    if(d_max==0)
        d_rendezvous.down();
    Thread::couldBlockDone(s);
}

template<class Item>
bool
SCICore::Thread::Mailbox<Item>::trySend(const Item& msg)
{
    d_mutex.lock();
    // See if the message buffer is full...
    int rmax=d_max==0?1:d_max;
    if(d_len == rmax){
        d_mutex.unlock();
        return false;
    }
    if(d_max == 0 && d_recv_wait==0){
        // No receivers waiting, so rendezvous will fail. Return now.
        d_mutex.unlock();
        return false;
    }

    d_ring_buffer[ringNext(d_len)]=msg;
    d_len++;
    if(d_recv_wait)
        d_empty.conditionSignal();
    d_mutex.unlock();
    if(d_max==0)
        d_rendezvous.down();  // Won't block for long, since a receiver
                            // will wake us up
    return true;
}

template<class Item>
Item
SCICore::Thread::Mailbox<Item>::receive()
{
    int s=Thread::couldBlock(d_name);
    d_mutex.lock();
    while(d_len == 0){
        d_recv_wait++;
        d_empty.wait(d_mutex);
        d_recv_wait--;
    }
    Item val=d_ring_buffer[d_head];
    d_head=ringNext(1);
    d_len--;
    if(d_send_wait)
        d_full.conditionSignal();
    d_mutex.unlock();
    if(d_max==0)
        d_rendezvous.up();
    Thread::couldBlockDone(s);
    return val;
}

template<class Item>
bool
SCICore::Thread::Mailbox<Item>::tryReceive(Item& item)
{
    d_mutex.lock();
    if(d_len == 0){
        d_mutex.unlock();
        return false;
    }
    item=d_ring_buffer[d_head];
    d_head=ringNext(1);
    d_len--;
    if(d_send_wait)
        d_full.conditionSignal();
    d_mutex.unlock();
    if(d_max==0)
        d_rendezvous.up();
    return true;
}

template<class Item>
int
SCICore::Thread::Mailbox<Item>::size() const
{
    return d_max;
}

template<class Item>
int
SCICore::Thread::Mailbox<Item>::numItems() const
{
    return d_len;
}

#endif

//
// $Log$
// Revision 1.5  1999/08/28 03:46:48  sparker
// Final updates before integration with PSE
//
// Revision 1.4  1999/08/25 19:00:48  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.3  1999/08/25 02:37:56  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

