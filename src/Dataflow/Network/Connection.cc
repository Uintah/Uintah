//static char *id="@(#) $Id$";

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

#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdio.h>
#ifdef _WIN32
#include <io.h>
#endif

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Dataflow/Port.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/MiscMath.h>

//#define DEBUG 1

namespace PSECore {
namespace Dataflow {

Connection::Connection(Module* m1, int p1, Module* m2, int p2)
{
    // mm- hack to get remote connections to work, only have one ptr
    if (m1 != 0)
    	oport = m1->oport(p1);
    else
	oport = 0;
    if (m2 != 0)
    	iport=m2->iport(p2);
    else 
	iport = 0;
    local=1;
    connected=1;
    socketPort = 0;
    remSocket = 0;
    remote = false;
    handle = 0;
#if 0
    demand=0;
#endif
}

// mm- can't attach to a port ptr that is null
void Connection::remoteConnect()
{
    if (iport)
    	iport->attach(this);
    if (oport)
    	oport->attach(this);
    connected=1;
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
   
    if (remSocket != 0) {
	close (remSocket);
#ifdef DEBUG
        cerr << "Connection::~Connection() just closed remote socket\n";
#endif
    }
}

Demand_Message::Demand_Message(Connection* conn)
  : MessageBase(Comm::MessageTypes::Demand), conn(conn)
{
}


Demand_Message::~Demand_Message()
{
}

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:21  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:57  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//
