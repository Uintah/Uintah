
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

#ifdef __GNUG__
#pragma interface
#endif

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
    Mailbox(int max);
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

#endif /* SCI_Multitask_Mailbox_h */
