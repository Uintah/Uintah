
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

#include <PSECore/share/share.h>
#include <PSECore/Comm/MessageBase.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/Thread/Mailbox.h>
#include <SCICore/Thread/Runnable.h>

namespace PSECore {
namespace Dataflow {

using SCICore::TclInterface::TCL;
using SCICore::TclInterface::TCLArgs;
using SCICore::Containers::clString;

using PSECore::Comm::MessageBase;

class Connection;
class Datatype;
class MessageBase;
class Module;
class Network;
class OPort;

class PSECORESHARE NetworkEditor : public SCICore::Thread::Runnable, public TCL {
    Network* net;
    void multisend(OPort*);
    void do_scheduling(Module*);
    int first_schedule;
    int schedule;
    void save_network(const clString&);
public:
    SCICore::Thread::Mailbox<MessageBase*> mailbox;

    NetworkEditor(Network*);
    ~NetworkEditor();

    void add_text(const clString&);
private:
    virtual void run();
    void main_loop();

    virtual void tcl_command(TCLArgs&, void*);
};

class PSECORESHARE Scheduler_Module_Message : public MessageBase {
public:
    Connection* conn;
    Scheduler_Module_Message();
    Scheduler_Module_Message(Connection* conn);
    virtual ~Scheduler_Module_Message();
};

class PSECORESHARE Module_Scheduler_Message : public MessageBase {
public:
    OPort* p1;
    OPort* p2;
    Module_Scheduler_Message();
    Module_Scheduler_Message(OPort*, OPort*);
    virtual ~Module_Scheduler_Message();
};

void postMessage(const clString& errmsg, bool err=true);
void postMessageNoCRLF(const clString& errmsg, bool err=true);

} // End namespace Dataflow
} // End namespace PSECore

#endif

