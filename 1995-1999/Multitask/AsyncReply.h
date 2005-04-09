
/*
 *  AsyncReply.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_Multitask_AsyncReply_h
#define SCI_Multitask_AsyncReply_h 1

#include <Multitask/ITC.h>

template<class Item> class AsyncReply {
    Semaphore sema;
    Item retval;
public:
    AsyncReply();
    ~AsyncReply();
    Item wait();
    void reply(Item);
};

#endif /* SCI_Multitask_AsyncReply_h */
