
/*
 *  Connection.cc: A Connection between two modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Connection.h>
#include <Port.h>

Connection::Connection(OPort* oport, IPort* iport)
: oport(oport), iport(iport)
{
    iport->attach(this);
    oport->attach(this);
    local=1;
}

Connection::~Connection()
{
    iport->attach(this);
    oport->attach(this);
}
