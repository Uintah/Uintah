
/*
 *  ProxyBase.h: Base class for all PIDL proxies
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Component/PIDL/ProxyBase.h>
#include <Component/Util/NotFinished.h>

ProxyBase::ProxyBase(const Reference& ref)
    : d_ref(ref)
{
}

ProxyBase::~ProxyBase()
{
}

bool ProxyBase::isa(const TypeSignature& ts) const
{
#if 0
    // Send a message...
    int size=sizeof(uint32)+ts.length();
    PCL::SendBuffer* buff=startpoint->requestBuffer(size);
    char* data=buff->getDataPtr();
    PCL::put(data, ts.length());
    PCL::put(data+sizeof(uint32), ts.c_str(), ts.length());
    startpoint->send(buff);

    // Wait for the result...
    ???;

    // Look for exceptions...

    bool flag;
    PCL::get(recvbuf+???, flag);
    return flag;
#else
    NOT_FINISHED("ProxyBase::isa");
    return true;
#endif
}

const Reference& ProxyBase::getReference() const
{
    return d_ref;
}

Startpoint* ProxyBase::getStartpoint() const
{
    return d_ref.d_startpoint;
}
//
// $Log$
// Revision 1.1  1999/08/30 17:39:47  sparker
// Updates to configure script:
//  rebuild configure if configure.in changes (Bug #35)
//  Fixed rule for rebuilding Makefile from Makefile.in (Bug #36)
//  Rerun configure if configure changes (Bug #37)
//  Don't build Makefiles for modules that aren't --enabled (Bug #49)
//  Updated Makfiles to build sidl and Component if --enable-parallel
// Updates to sidl code to compile on linux
// Imported PIDL code
// Created top-level Component directory
// Added ProcessManager class - a simpler interface to fork/exec (not finished)
//
//
