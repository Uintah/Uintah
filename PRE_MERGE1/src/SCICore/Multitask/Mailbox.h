
/*
 *  Mailbox.h: Inter-task communication
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Multitask_Mailbox_h
#define SCI_Multitask_Mailbox_h 1

namespace SCICore {
namespace Multitask {

template<class Item> struct Mailbox_private;

template<class Item> class Mailbox {
    Mailbox_private<Item>* priv;
public:
    /*
     * Create a mailbox with a maximum queue size of <tt>size</tt> items.
     * If size is zero, then the mailbox will use <i>rendezvous semantics</i>,
     * where a sender will block until a receiver is waiting for the item.
     * The item will be handed off synchronously.
     */
    Mailbox(int size);

    /*
     * Destroy the mailbox.  All items in the queue are silently dropped.
     */
    ~Mailbox();

    /*
     * Puts <tt>item</tt> in the queue.  If the queue is full, the
     * thread will be blocked until there is room in the queue.  Multiple
     * threads may call <i>send</i> concurrently, and the items will be
     * placed in the queue in an undefined order.
     */
    void send(Item item);

    /*
     * Attempt to send <tt>item</tt> to the queue.  If the queue is full,
     * the thread will not be blocked, and <i>try_send</i> will return
     * true.  Otherwise, <i>try_send</i> returns true.
     */
    bool try_send(Item item);

    /*
     * Receive an item from the queue.  If the queue is empty, the
     * thread will block until another thread sends an item.  Multiple
     * threads may call <i>receive</i> concurrently, but no guarantee
     * is made as to which thread will receive the next token.  However,
     * implementors should give preference to the thread that has been
     * waiting the longest.
     */
    Item receive();

    /*
     * Attempt to receive <tt>item</tt> from the mailbox.  If the queue
     * is empty, the thread is not blocked and <i>try_receive</i> will
     * return false.  Otherwise, <i>try_receive</i> returns true.
     */
    int try_receive(Item& item);

    /*
     * Return the maximum size of the mailbox queue, as given in the
     * constructor.
     */
    int size() const;

    /*
     * Return the number of items currently in the queue.
     */
    int nitems() const;
};

} // End namespace Multitask
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included Mailbox.cc
//

#include <Multitask/ITC.h>
#include <Malloc/Allocator.h>

namespace SCICore {
namespace Multitask {

// Implement the Mailbox with a Mutex and Condition Variables
template<class Item> struct Mailbox_private {
    Mutex mutex;
    Item* ring_buffer;
    int head;
    int len;
    int max;
    ConditionVariable send_condition;
    ConditionVariable recv_condition;
    int send_wait;
    int recv_wait;
};


template<class Item>
Mailbox<Item>::Mailbox(int max)
{
    priv=scinew Mailbox_private<Item>;
    priv->ring_buffer=new Item[max];
    priv->head=0;
    priv->len=0;
    priv->send_wait=0;
    priv->recv_wait=0;
    priv->max=max;
}

template<class Item>
Mailbox<Item>::~Mailbox()
{
    delete[] priv->ring_buffer;
    delete priv;
}

#define NEXT(head, inc, max) ((head+inc)%max)

template<class Item>
void Mailbox<Item>::send(Item msg)
{
    priv->mutex.lock();
    // See if the message buffer is full...
    while(priv->len == priv->max){
	priv->send_wait++;
	priv->send_condition.wait(priv->mutex);
	priv->send_wait--;
    }
    priv->ring_buffer[NEXT(priv->head, priv->len, priv->max)]=msg;
    priv->len++;
    if(priv->recv_wait)
	priv->recv_condition.cond_signal();
    priv->mutex.unlock();
}

template<class Item>
Item Mailbox<Item>::receive()
{
    priv->mutex.lock();
    while(priv->len == 0){
	priv->recv_wait++;
	priv->recv_condition.wait(priv->mutex);
	priv->recv_wait--;
    }
    Item val=priv->ring_buffer[priv->head];
    priv->head=NEXT(priv->head, 1, priv->max);
    priv->len--;
    if(priv->send_wait)
	priv->send_condition.cond_signal();
    priv->mutex.unlock();
    return val;
}

template<class Item>
int Mailbox<Item>::try_receive(Item& item)
{
    priv->mutex.lock();
    if(priv->len == 0){
	priv->mutex.unlock();
	return 0;
    }
    item=priv->ring_buffer[priv->head];
    priv->head=NEXT(priv->head, 1, priv->max);
    priv->len--;
    if(priv->send_wait)
	priv->send_condition.cond_signal();
    priv->mutex.unlock();
    return 1;
}

template<class Item>
int Mailbox<Item>::size() const
{
    return priv->max;
}

template<class Item>
int Mailbox<Item>::nitems() const
{
    return priv->len;
}

} // End namespace Multitask
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:06  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:21  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:28  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif /* SCI_Multitask_Mailbox_h */
