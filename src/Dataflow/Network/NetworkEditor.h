
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
#include <SCICore/Multitask/Task.h>
#include <SCICore/Multitask/ITC.h>
#include <SCICore/Multitask/Mailbox.h>
#include <SCICore/TclInterface/TCL.h>

namespace PSECore {
namespace Dataflow {

using SCICore::Multitask::Task;
using SCICore::Multitask::Mailbox;
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

class PSECORESHARE NetworkEditor : public Task, public TCL {
    Network* net;
    void multisend(OPort*);
    void do_scheduling(Module*);
    int first_schedule;
    int schedule;
    void save_network(const clString&);
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

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/26 23:59:07  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.2  1999/08/17 06:38:24  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:59  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 22:02:44  dav
// added back .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif

