
/*
 *  Mailbox.cc: Template implementation of mailbox code
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef __GNUG__
#pragma interface
#endif

#include <Multitask/ITC.h>
#include <Malloc/Allocator.h>

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
int Mailbox<Item>::size() const
{
    return priv->max;
}

template<class Item>
int Mailbox<Item>::nitems() const
{
    return priv->len;
}

