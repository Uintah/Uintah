
/*
 *  NetworkEditor.h: Interface to Network Editor class from project
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_NetworkEditor_h
#define SCI_project_NetworkEditor_h 1

#include <Comm/MessageBase.h>
#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <TCL/TCL.h>

class Connection;
class Datatype;
class MessageBase;
class Module;
class Network;
class OPort;

class NetworkEditor : public Task, public TCL {
    Network* net;
    void multisend(OPort*);
    void do_scheduling(Module*);
    int first_schedule;
    int schedule;
public:
    Mailbox<MessageBase*> mailbox;

    NetworkEditor(Network*);
    ~NetworkEditor();

    void add_text(const clString&);
private:
    virtual int body(int);
    void main_loop();

    virtual void tcl_command(TCLArgs&, void*);
};

class Scheduler_Module_Message : public MessageBase {
public:
    Connection* conn;
    Scheduler_Module_Message();
    Scheduler_Module_Message(Connection* conn);
    virtual ~Scheduler_Module_Message();
};

class Module_Scheduler_Message : public MessageBase {
public:
    OPort* p1;
    OPort* p2;
    Module_Scheduler_Message();
    Module_Scheduler_Message(OPort*, OPort*);
    virtual ~Module_Scheduler_Message();
};

#endif
