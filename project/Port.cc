
/*
 *  Port.cc: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Port.h>
#include <Data.h>
#include <Datatype.h>
#include <iostream.h>

Port::Port(Module* module, int which_port, const clString& dt,
	   const clString& name)
: module(module), which_port(which_port), name(name)
{
    datatype=Datatype::lookup(dt);
}

IPort::IPort(Module* module, int which_port, InData* data,
	     const clString& name)
: Port(module, which_port, data->typename(), name), data(data)
{
}

OPort::OPort(Module* module, int which_port, OutData* data,
	     const clString& name)
: Port(module, which_port, data->typename(), name), data(data)
{
}

void Port::attach(Connection* conn)
{
    connections.add(conn);
}

int Port::nconnections()
{
    return connections.size();
}

Connection* Port::connection(int i)
{
    return connections[i];
}
