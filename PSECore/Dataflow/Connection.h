
/*
 *  Connection.h: A Connection between two modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Connection_h
#define SCI_project_Connection_h 1

#include <SCICore/share/share.h>

#include <SCICore/Containers/String.h>
#include <PSECore/Comm/MessageBase.h>

namespace PSECore {
namespace Dataflow {

using SCICore::Containers::clString;
using PSECore::Comm::MessageBase;

class IPort;
class Module;
class OPort;

class SCICORESHARE Connection {
    int connected;
public:
    Connection(Module*, int, Module*, int);
    ~Connection();
    bool isRemote()   	{return remote;}   // mm-test of remote flag
    void setRemote()    {remote = true;}   // mm-set remote to true

    OPort* oport;
    IPort* iport;
    int local;
    clString id;
    bool remote;	// mm-flag for remote connection
    int handle;		// mm-connection handle for distrib. connections
    int socketPort;	// mm-port number for remote connections
    int remSocket;  	// mm-comm channel that has both sides connected

#if 0
    int demand;
#endif

    void wait_ready();

    void remoteConnect();    // mm-special method to connect remote endpoints
    void connect();
};


class SCICORESHARE Demand_Message : public MessageBase {
public:
    Connection* conn;
    Demand_Message(Connection* conn);
    virtual ~Demand_Message();
};

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:22  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:57  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 22:02:41  dav
// added back .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif /* SCI_project_Connection_h */
