
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

#include <SCICore/Multitask/ITC.h>

namespace SCICore {
namespace Multitask {

template<class Item> class AsyncReply {
    Semaphore sema;
    Item retval;
public:
    AsyncReply();
    ~AsyncReply();
    Item wait();
    void reply(Item);
};

} // End namespace Multitask
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included AsyncReply.cc
//

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
// Revision 1.2  1999/08/17 06:39:37  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:06  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:20  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:27  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//


#endif /* SCI_Multitask_AsyncReply_h */
