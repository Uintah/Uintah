
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
    Mailbox();
    Mailbox(int max);
    ~Mailbox();
    void send(Item);
    Item receive();
    int try_receive(Item&);
    int size() const;
    int nitems() const;
};

#endif /* SCI_Multitask_Mailbox_h */
