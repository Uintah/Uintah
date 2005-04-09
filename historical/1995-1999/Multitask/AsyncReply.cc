
/*
 *  AsyncReply.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Multitask/AsyncReply.h>

template<class Item>
AsyncReply<Item>::AsyncReply()
: sema(0)
{
}

template<class Item>
AsyncReply<Item>::~AsyncReply()
{
}

template<class Item>
Item AsyncReply<Item>::wait()
{
    sema.down();
    return retval;
}

template<class Item>
void AsyncReply<Item>::reply(Item i)
{
    retval=i;
    sema.up();
}
