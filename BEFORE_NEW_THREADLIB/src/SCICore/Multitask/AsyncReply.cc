//static char *id="@(#) $Id$";

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

namespace SCICore {
namespace Multitask {

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

} // End namespace Multitask
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:06  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

