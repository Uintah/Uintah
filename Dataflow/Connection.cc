
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <Dataflow/Port.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>

Connection::Connection(Module* m1, int p1, Module* m2, int p2)
{
    oport=m1->oport(p1);
    iport=m2->iport(p2);
    local=1;
    connected=1;
#if 0
    demand=0;
#endif
}

void Connection::connect()
{
    iport->attach(this);
    oport->attach(this);
    connected=1;
}

Connection::~Connection()
{
    if (connected) {
	iport->detach(this);
	oport->detach(this);
    }
}

Demand_Message::Demand_Message(Connection* conn)
  : MessageBase(MessageTypes::Demand), conn(conn)
{
}


Demand_Message::~Demand_Message()
{
}
