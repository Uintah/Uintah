//static char *id="@(#) $Id$";

/*
 *  SlaveController.cc: Daemon - Central Slave interface to the Master's
 *   scheduler.
 *
 *  Keeps a socket to the scheduler on the Master for control information.
 *  Dispatches control directives (messages) sent from the Master to direct
 *  remote SCIRun networks (running on a Slave).
 *
 *  Written by:
 *   Michelle Miller
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Distributed/SlaveController.h>

#include <Containers/String.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <Dataflow/Network.h>
#include <Malloc/Allocator.h>
#include <TclInterface/Remote.h>

#include <stdio.h>
#ifdef _WIN32
#include <string.h>
#include <io.h>
#endif
#include <iostream>

namespace PSECore {
namespace Distributed {

//using PSECore::Dataflow::ModuleList;
//using PSECore::Dataflow::ModuleDB;
//using PSECore::Dataflow::makeModule;
using PSECore::Comm::MessageTypes;

using SCICore::Containers::clString;
using SCICore::TclInterface::Message;
using SCICore::TclInterface::requestConnect;

SlaveController::SlaveController (Network* net, char* hostname, char* portnum)
: Task("SlaveController"), net(net), master_socket(0), mailbox(100)
{
	strcpy (masterHost, hostname);
	masterPort = atoi (portnum);
	printf ("constructed SlaveController\tmasterHost= %s, masterPort= %d\n",
		masterHost, masterPort);
}

SlaveController::~SlaveController()
{ }

int
SlaveController::body(int)
{
    strcat (masterHost,".cs.utah.edu");
    master_socket = requestConnect (masterPort, masterHost);
    if (master_socket == -1) {
	perror ("outgoing connection setup failed");
	exit (-1);
    }
    if (main_loop() == -1)
	perror ("main_loop");
    return 0;
}

// since both sides have handle & id, a module can be accessed by either

int
SlaveController::createModule (char *name, char *id, int handle)
{
#ifdef DEBUG
    cerr << "Entering SlaveController::createModule (" << name << ", "
	 << id << ", " << handle << ")\n";
#endif

    //name += 1;			// mv pointer 1 to get rid of 'r'
    const clString name_str = name;
 
    // net->add_module (name_str, TRUE); add flag to call to denote slave

    // Dd: This doesn't appear to be used and "get_db()" is obsolete now...
    //    ModuleDB* db = ModuleList::get_db();

    makeModule maker=ModuleList::lookup(name_str);
    if(!maker){
        cerr << "Module: " << name_str << " not found!\n";
        return 0;
    }

    // calls the module constructor (name,id,SchedClass). id retains 'r'
    Module* mod = (*maker)(id);
    mod->handle = handle;
    mod->remote = true;		// keep just in case I need it...
    
    // add to array of Module ptrs in Network
    net->modules.add (mod);       

    // bind Network to Module instance, create thread, start event loop
    mod->set_context (0, net);

    // add Module id and ptr to Module to hash table of modules in network
    net->module_ids[mod->id] = mod;

    // add to hash table of handles and module ptrs
    if (mod->handle > 0)
	net->mod_handles.[mod->handle] = mod;

#ifdef DEBUG
    cerr << "Exiting SlaveController::createModule (" << name << ", "
	 << id << ", " << handle << ")\n";
#endif

    return 0;
}

int
SlaveController::createLocConnect (int outMod, int oport, int inMod,
				   int iport, char *connID, int connHandle)
{
#ifdef DEBUG
    cerr << "Entering SlaveController::createLocConnect(" << outMod << ", "
   	 << oport << ", " << inMod << ", " << iport << ", " << connID << ", "
	 << connHandle << ")\n";
#endif

    // unpack module handles into ptrs & instantiate connection
    Module *m1 = net->get_module_by_handle (outMod);
    Module *m2 = net->get_module_by_handle (inMod);

    // add connection to network : net->connect(m1, oport, m2, iport);
    Connection* conn = scinew Connection(m1, oport, m2, iport);
    conn->id = connID;
    conn->handle = connHandle;
    conn->connect();
    net->connections.add(conn);
    // Reschedule next time we can...
    net->reschedule = 1;

#ifdef DEBUG
    cerr << "Entering SlaveController::createLocConnect(" << outMod << ", "
   	 << oport << ", " << inMod << ", " << iport << ", " << connID << ", "
	 << connHandle << ")\n";
#endif
    return 0;
}

int
SlaveController::createRemConnect (bool fromRemote, int outMod, int oport,
				   int inMod, int iport, char *connID,
			   	   int connHandle, int socket)
{
    Module *m1;
    Module *m2;

#ifdef DEBUG
    cerr << "Entering SlaveController::createRemConnect (" << fromRemote
	 << ", " << outMod << ", " << oport << ", " << inMod << ", " << iport
	 << ", " << connID << ", " << connHandle << ", " << socket << ")\n"; 
#endif
    // unpack slave module handle into ptr & instantiate connection - only
    // need the ptrs on this side. Remote side is taken care of, so set to 0
    if (fromRemote) {
	m1 = net->get_module_by_handle (outMod);
	m2 = 0;
    } else {
	m1 = 0;
	m2 = net->get_module_by_handle (inMod);
    }

    // add connection to network : net->connect(m1, oport, m2, iport);
    Connection* conn = scinew Connection(m1, oport, m2, iport);
    conn->id = connID;
    conn->handle = connHandle;
    conn->setRemote();
    conn->socketPort = socket;
    conn->remoteConnect();
    net->connections.add(conn);
    // Reschedule next time we can...
    net->reschedule = 1;

    // request data connection
    conn->remSocket = requestConnect (conn->socketPort, masterHost);
    if (conn->remSocket == -1) {
	perror ("request connect failed");
	exit (-1);
    }

#ifdef DEBUG
    cerr << "Exiting SlaveController::createRemConnect (" << fromRemote
	 << ", " << outMod << ", " << oport << ", " << inMod << ", " << iport
	 << ", " << connID << ", " << connHandle << ", " << socket << ")\n"; 
#endif
    return 0;
}

int
SlaveController::executeModule (int handle)
{
#ifdef DEBUG
    cerr << "Entering SlaveController::executeModule (" << handle << ")\n";
#endif

    Module *mod = net->get_module_by_handle (handle);
    mod->mailbox.send(scinew R_Scheduler_Module_Message);

#ifdef DEBUG
    cerr << "Exiting SlaveController::executeModule (" << handle << ")\n";
#endif

    return 0;
}

int
SlaveController::triggerPort (int modHandle, int connHandle)
{
#ifdef DEBUG
    cerr << "Entering SlaveController::triggerPort (" << modHandle << ", "
	 << connHandle << ")\n";
#endif

    Module *mod = net->get_module_by_handle (modHandle);
    Connection *conn = net->get_connect_by_handle (connHandle);
    mod->mailbox.send(scinew R_Scheduler_Module_Message (conn));

#ifdef DEBUG
    cerr << "Exiting SlaveController::triggerPort (" << modHandle << ", "
	 << connHandle << ")\n";
#endif
    return 0;
}

int
SlaveController::deleteModule (int handle)
{
#ifdef DEBUG
    cerr << "Entering SlaveController::deleteModule (" << handle << ")\n";
#endif

    Module *mod = net->get_module_by_handle (handle);
    
    // traverse array of ptrs to Modules in Network to find this module
    int i;
    for (i = 0; i < net->modules.size(); i++)
        if (net->modules[i] == mod)
            break;
    if (i == net->modules.size())
        return -1;

    // remove array element corresponding to module, remove from hash tables
    net->modules.remove(i);
    net->module_ids.erase(mod->id);
    net->mod_handles.erase(handle);
    delete mod;  	// might cause problems with dangling refs in net

#ifdef DEBUG
    cerr << "Exiting SlaveController::deleteModule (" << handle << ")\n";
#endif
    return 0;
}

// don't really need 2 functions here, but I will leave separate til I know
// all differences that will be required.
int
SlaveController::deleteRemConnect (int connHandle)
{
#ifdef DEBUG
    cerr << "Entering SlaveController::deleteRemConnect (" << connHandle
	 << ")\n";
#endif

    Connection *conn = net->get_connect_by_handle (connHandle);
    
    // this construct sucks!  There MUST be a better way!
    // traverse array of ptrs to Connections in Network to find this conn
    int i;
    for (i = 0; i < net->connections.size(); i++)   
	if (net->connections[i] == conn)
		break;
    if (i == net->connections.size())
	return -1;

    // remove connection from array, hash table, and free memory
    net->connections.remove(i);
    net->conn_handles.erase(connHandle);
    delete conn;

#ifdef DEBUG
    cerr << "Exiting SlaveController::deleteRemConnect (" << connHandle
	 << ")\n";
#endif
    return 0;
}

int
SlaveController::deleteLocConnect (int connHandle)
{
#ifdef DEBUG
    cerr << "Entering SlaveController::deleteLocConnect (" << connHandle
	 << ")\n";
#endif

    Connection *conn = net->get_connect_by_handle (connHandle);
    
    // this construct sucks!  There MUST be a better way!
    // traverse array of ptrs to Connections in Network to find this conn
    int i;
    for (i = 0; i < net->connections.size(); i++)   
	if (net->connections[i] == conn)
		break;
    if (i == net->connections.size())
	return -1;

    // remove connection from array, hash table, and free memory
    net->connections.remove(i);
    net->conn_handles.erase(connHandle);
    delete conn;

#ifdef DEBUG
    cerr << "Exiting SlaveController::deleteLocConnect (" << connHandle
	 << ")\n";
#endif
    return 0;
}

int SlaveController::main_loop()
{
    char buf[BUFSIZE];
    int rval = 0;
    int result = 0;
    Message msg;

    for (;;) {
	// check socket for messages from master and dispatch
	bzero (buf, sizeof (buf));
	if ((rval = read (master_socket, buf, sizeof (buf))) < 0) {
	    perror ("reading master_socket message");
	}
	else if (rval > 0) {
	    bcopy (buf, (char *) &msg, sizeof (msg));
 	    switch (msg.type) {
	    	case CREATE_MOD:
		    result = createModule (msg.u.cm.name, msg.u.cm.id,
					   msg.u.cm.handle);
		    break;
	    	case CREATE_LOC_CONN:
		    result = createLocConnect (msg.u.clc.outModHandle,
					       msg.u.clc.oport,
					       msg.u.clc.inModHandle,
					       msg.u.clc.iport,
					       msg.u.clc.connID,
					       msg.u.clc.connHandle);
		    break;
	  	case CREATE_REM_CONN:
		    result = createRemConnect (msg.u.crc.fromRemote,
					       msg.u.crc.outModHandle,
				               msg.u.crc.oport,
					       msg.u.crc.inModHandle,
					       msg.u.crc.iport,
					       msg.u.crc.connID,
					       msg.u.crc.connHandle,
					       msg.u.crc.socketPort);
		    break;
	    	case EXECUTE_MOD:
		    result = executeModule (msg.u.e.modHandle);
		    break;
	  	case TRIGGER_PORT:
		    result = triggerPort (msg.u.tp.modHandle, 
					  msg.u.tp.connHandle);
		    break;
		case DELETE_MOD:
		    result = deleteModule (msg.u.dm.modHandle);
		    break;
		case DELETE_LOC_CONN:
		    result = deleteLocConnect (msg.u.dlc.connHandle);
		    break;
		case DELETE_REM_CONN:
		    result = deleteRemConnect (msg.u.drc.connHandle);
		    break;
		default:
		    printf ("Unrecognized message type received\n");
		    break;
	    } 
	// return ACK to master?  ReturnResult(result) RPC semantics
	} 
/*  cannot combine these 2 here.  both block - 2 different communications
 *  mechanisms.
 *    	else { 	 rval == 0 so there is nothing in msg queue
	    check for messages from modules on local machine and dispatch
ZZZ- does mailbox receive block?  YES.
	    MessageBase* localMsg = mailbox.receive();
            switch (localMsg->type) {
                case MessageTypes::MultiSend :
		     format multisend message;
		     send msg out master_socket;
		     if (write (master_socket, 
               	    break;
                case MessageTypes::ReSchedule :
		     format reschedule message;
		     send msg out master_socket;
            	    break;
                default:
                    cerr << "Unknown message type: " <<localMsg->type <<endl;
                    break;
            }
            delete localMsg;
	}
 */
    } /* for */
} /*function*/

R_Scheduler_Module_Message::R_Scheduler_Module_Message()
: MessageBase(MessageTypes::ExecuteModule)
{
}

R_Scheduler_Module_Message::R_Scheduler_Module_Message(Connection* conn)
: MessageBase(MessageTypes::TriggerPort), conn(conn)
{
}

R_Scheduler_Module_Message::~R_Scheduler_Module_Message()
{
}

/*

Module_Scheduler_Message::Module_Scheduler_Message()
: MessageBase(MessageTypes::ReSchedule)
{
}

Module_Scheduler_Message::Module_Scheduler_Message(OPort* p1, OPort* p2)
: MessageBase(MessageTypes::MultiSend), p1(p1), p2(p2)
{
}

Module_Scheduler_Message::~Module_Scheduler_Message()
{
}

 */

} // End namespace Distributed
} // End namespace PSECore

//
// $Log$
// Revision 1.4  2000/03/11 00:40:56  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.3  1999/10/07 02:07:22  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:38:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:01  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
