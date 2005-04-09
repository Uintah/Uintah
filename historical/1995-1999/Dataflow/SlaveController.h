/*
 *  SlaveController.h: Daemon - Central Slave interface to the Master's
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

#ifndef SCI_project_SlaveController_h
#define SCI_project_SlaveController_h

#include <Comm/MessageBase.h>
#include <Multitask/Task.h>
#include <Multitask/Mailbox.h>
#include <TCL/Remote.h>

class Connection;
class Datatype;
class MessageBase;
class Module;
class Network;
class OPort; 

class SlaveController : public Task {
    private:
	Network* 	net;
	char 		masterHost[HOSTNAME];	
	int 		masterPort;
	int 		master_socket;
    public:
    	Mailbox<MessageBase*> mailbox;
	SlaveController (Network*, char*, char*);
	~SlaveController();
    private:
	int createModule (char *name, char* id, int handle);
	int createLocConnect (int, int, int, int, char*, int);
	int createRemConnect (bool, int, int, int, int, char*, int, int);
	int executeModule (int handle);
	int triggerPort (int modHandle, int connHandle);
	int deleteModule (int modHandle);
 	int deleteRemConnect (int connHandle);
	int deleteLocConnect (int connHandle);
	int main_loop();
	virtual int body(int);
};

class R_Scheduler_Module_Message : public MessageBase {
public:
    Connection* conn;
    R_Scheduler_Module_Message();			// execute msg
    R_Scheduler_Module_Message(Connection* conn);	// trigger port
    virtual ~R_Scheduler_Module_Message();
};

class R_Module_Scheduler_Message : public MessageBase {
public:
    OPort* p1;
    OPort* p2;
    R_Module_Scheduler_Message();			// reschedule
    R_Module_Scheduler_Message(OPort*, OPort*); 	// multisend
    virtual ~R_Module_Scheduler_Message();
};

#endif
