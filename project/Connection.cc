
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
#include <Module.h>
#include <Port.h>

Connection::Connection(Module* m1, int p1, Module* m2, int p2)
{
    oport=m1->oport(p1);
    iport=m2->iport(p2);
    local=1;
}

Connection::~Connection()
{
}

void Connection::attach(OPort* op)
{
    oport=op;
}

void Connection::attach(IPort* ip)
{
    iport=ip;
}
