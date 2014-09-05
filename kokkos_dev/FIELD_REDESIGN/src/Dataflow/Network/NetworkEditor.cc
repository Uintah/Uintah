//static char *id="@(#) $Id$";

/*
 *  NetworkEditor.cc: The network editor...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Distributed SCIRun changes:
 *   Michelle Miller
 *   Nov. 1997
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef _WIN32
#pragma warning(disable:4786)
#endif

#include <PSECore/Dataflow/NetworkEditor.h>
  
#include <SCICore/Containers/Queue.h>
#include <PSECore/Comm/MessageBase.h>
#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Dataflow/Network.h>
#include <PSECore/Dataflow/PackageDB.h>
#include <PSECore/Dataflow/Port.h>
#include <PSECore/Dataflow/ComponentNode.h>
#include <PSECore/Dataflow/GenFiles.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/Remote.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/Thread/Thread.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#endif

#include <fstream>
using std::ofstream;
#include <iostream>
using std::cerr;
using std::endl;
  
using SCICore::Thread::Thread;

//#define DEBUG 1
#include <tcl.h>
  
#ifdef _WIN32
extern "C" __declspec(dllimport) Tcl_Interp* the_interp;
#else
extern "C" Tcl_Interp* the_interp;
#endif
  
namespace PSECore {
namespace Dataflow {

using SCICore::TclInterface::Message;
using PSECore::Comm::MessageTypes;

// This function was added by Mohamed Dekhil for CSAFE
void init_notes ()
{
    // uid_t userID ;
    char d[40] ;
    char t[20] ;
    char n[80] ;
    time_t t1 ;
    struct tm *t2 ;
    // char *myvalue ;
    char *days[7] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
    char *months[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", 
			"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    
    
    // Construct date, time and user name strings here and pass then to TCL.

    //userID = getuid () ;
    //strcpy (n, getlogin()) ;
#ifndef _WIN32
    //strcpy (n, cuserid(NULL)) ;
    strcpy(n, getenv("LOGNAME"));
    printf ("User Name: %s\n", n) ;
#endif

    t1 = time(NULL) ;
    t2 = localtime (&t1) ;
    sprintf (d, " %s  %s %0d 19%0d", days[t2->tm_wday], months[t2->tm_mon], 
	     t2->tm_mday, t2->tm_year) ;
    sprintf (t, " %0d:%0d:%0d", t2->tm_hour, t2->tm_min, t2->tm_sec) ;

    /* myvalue = */
    Tcl_SetVar (the_interp, "userName", n, TCL_GLOBAL_ONLY) ;
    /* myvalue = */
    Tcl_SetVar (the_interp, "runDate", d, TCL_GLOBAL_ONLY) ;
    /* myvalue = */
    Tcl_SetVar (the_interp, "runTime", t, TCL_GLOBAL_ONLY) ;
}

NetworkEditor::NetworkEditor(Network* net)
: mailbox("NetworkEditor request FIFO", 100), net(net),
    first_schedule(1), schedule(1)
{
    // Create User interface...
    TCL::add_command("netedit", this, 0);
    TCL::source_once("$PSECoreTCL/NetworkEditor.tcl");
    TCL::execute("makeNetworkEditor");

    // Initialize the network
    net->initialize(this);

    // This part was added by Mohamed Dekhil for CSAFE
    init_notes () ;
}

NetworkEditor::~NetworkEditor()
{
}

void
NetworkEditor::run()
{
    // Go into Main loop...
    do_scheduling(0);
    main_loop();
}

void NetworkEditor::main_loop()
{
    // Dispatch events...
    int done=0;
    while(!done){
	MessageBase* msg=mailbox.receive();
	// Dispatch message....
	switch(msg->type){
	case MessageTypes::MultiSend:
	    {
		//cerr << "Got multisend\n";
		Module_Scheduler_Message* mmsg=(Module_Scheduler_Message*)msg;
		multisend(mmsg->p1);
		if(mmsg->p2)
		    multisend(mmsg->p2);
		// Do not re-execute sender

		// do_scheduling on the module instance bound to
		// the output port p1 (the first arg in Multisend() call)
		do_scheduling(mmsg->p1->get_module()); 
	    }
	    break;
	case MessageTypes::ReSchedule:
	    do_scheduling(0);
	    break;
	default:
	    cerr << "Unknown message type: " << msg->type << endl;
	    break;
	};
	delete msg;
    };
}

void NetworkEditor::multisend(OPort* oport)
{
    int nc=oport->nconnections();
    for(int c=0;c<nc;c++){
	Connection* conn=oport->connection(c);
	IPort* iport=conn->iport;
	Module* m=iport->get_module();
	if(!m->need_execute){
	    m->need_execute=1;
	}
    }
}

void NetworkEditor::do_scheduling(Module* exclude)
{
  using SCICore::Containers::Queue;

    Message msg;

    if(!schedule)
	return;
    int nmodules=net->nmodules();
    Queue<Module*> needexecute;		

    // build queue of module ptrs to execute
    int i;			    
    for(i=0;i<nmodules;i++){
	Module* module=net->module(i);
	if(module->need_execute)
	    needexecute.append(module);
    }
    if(needexecute.is_empty()){
	return;
    }

    // For all of the modules that need executing, execute the
    // downstream modules and arrange for the data to be sent to them
    // mm - this doesn't really execute them. It just adds modules to
    // the queue of those to execute based on dataflow dependencies.

    Array1<Connection*> to_trigger;
    while(!needexecute.is_empty()){
	Module* module=needexecute.pop();
	// Add oports
	int no=module->noports();
	int i;
	for(i=0;i<no;i++){
	    OPort* oport=module->oport(i);
	    int nc=oport->nconnections();
	    for(int c=0;c<nc;c++){
		Connection* conn=oport->connection(c);
		IPort* iport=conn->iport;
		Module* m=iport->get_module();
		if(m != exclude && !m->need_execute){
		    m->need_execute=1;
		    needexecute.append(m);
		}
	    }
	}

	// Now, look upstream...
	int ni=module->niports();
	for(i=0;i<ni;i++){
	    IPort* iport=module->iport(i);
	    if(iport->nconnections()){
		Connection* conn=iport->connection(0);
		OPort* oport=conn->oport;
		Module* m=oport->get_module();
		if(!m->need_execute){
		    if(m != exclude){
			if(module->sched_class != Module::SalmonSpecial){
			    // If this oport already has the data, add it
			    // to the to_trigger list...
			    if(oport->have_data()){
				to_trigger.add(conn);
			    } else {
				m->need_execute=1;
				needexecute.append(m);
			    }
			}
		    }
		}
	    }

	}
    }

    // Trigger the ports in the trigger list...
    for(i=0;i<to_trigger.size();i++){
	Connection* conn=to_trigger[i];
	OPort* oport=conn->oport;
	Module* module=oport->get_module();
	//cerr << "Triggering " << module->name << endl;
	if(module->need_execute){
	    // Executing this module, don't actually trigger....
	}
    	else if (module->isSkeleton()) {

	    // format TriggerPortMsg
            msg.type        = TRIGGER_PORT;
            msg.u.tp.modHandle = module->handle;
	    msg.u.tp.connHandle = conn->handle;

	    // send msg to slave
	    char buf[BUFSIZE];
       	    bzero (buf, sizeof (buf));
            bcopy ((char *) &msg, buf, sizeof (msg));
            write (net->slave_socket, buf, sizeof(buf));
	}
	else {
	    module->mailbox.send(scinew Scheduler_Module_Message(conn));
	}
    }

    // Trigger any modules that need executing...
    for(i=0;i<nmodules;i++){
	Module* module=net->module(i);
	if(module->need_execute){

	    // emulate local fire and forget mailbox semantics? YES!
	    if (module->isSkeleton()) {
		
		// format ExecuteMsg
	        msg.type        = EXECUTE_MOD;
        	msg.u.e.modHandle = module->handle;

		// send msg to slave
        	char buf[BUFSIZE];
        	bzero (buf, sizeof (buf));
        	bcopy ((char *) &msg, buf, sizeof (msg));
        	write (net->slave_socket, buf, sizeof(buf));
	    }
	    else {
	    	module->mailbox.send(scinew Scheduler_Module_Message);
	    }
	    module->need_execute=0;
#ifdef DEBUG
	    cerr << "Firing " << module->name << endl;
#endif
	}
    }

#ifdef STUPID_SCHEDULER
    for(int i=0;i<nmodules;i++){
	Module* module=net->module(i);

	// Tell it to trigger...
	module->mailbox.send(scinew Scheduler_Module_Message);

	// Reset the state...
	module->sched_state=Module::SchedDormant;
    }
#endif
}

Scheduler_Module_Message::Scheduler_Module_Message()
: MessageBase(MessageTypes::ExecuteModule)
{
}

Scheduler_Module_Message::Scheduler_Module_Message(Connection* conn)
: MessageBase(MessageTypes::TriggerPort), conn(conn)
{
}

Scheduler_Module_Message::~Scheduler_Module_Message()
{
}

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

void NetworkEditor::add_text(const clString& str)
{
    TCL::execute("global netedit_errortext");
    TCL::execute("$netedit_errortext configure -state normal");
    TCL::execute("$netedit_errortext insert end \""+str+"\n\"");
    TCL::execute("$netedit_errortext configure -state disabled");
}

void NetworkEditor::save_network(const clString& filename)
{
    using SCICore::TclInterface::TCLTask;

    char *myvalue ;

    ofstream out(filename());
    if(!out)
      return;
    out << "# SCI Network 1.0\n";
    out << "\n";
    out << "######################\n";
    out << "# These commands generated automatically, DO NOT REMOVE!\n";
    out << "loadfile "<<filename<<"\n";
    out << "return\n";
    out << "######################\n";
    out << "::netedit dontschedule\n";
    net->read_lock();

    // Added by Mohamed Dekhil for saving extra information

    TCLTask::lock();

    myvalue = Tcl_GetVar (the_interp, "userName", TCL_GLOBAL_ONLY) ;
    if (myvalue != NULL) {
      out << "global userName\nset userName \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }
    myvalue = Tcl_GetVar (the_interp, "runDate", TCL_GLOBAL_ONLY) ;
    if (myvalue != NULL) {
      out << "global runDate\nset runDate \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }
    myvalue = Tcl_GetVar (the_interp, "runTime", TCL_GLOBAL_ONLY) ;
    if (myvalue != NULL) {
      out << "global runTime\nset runTime \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }
    myvalue = Tcl_GetVar (the_interp, "notes", TCL_GLOBAL_ONLY) ;
    if (myvalue != NULL) {
      out << "global notes\nset notes \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }

    TCLTask::unlock();


    // --------------------------------------------------------------------

    int i;
    for(i=0;i<net->nmodules();i++){
        Module* module=net->module(i);
	int x, y;
	module->get_position(x,y);
        out << "set m" << i << " [addModuleAtPosition \""
            << module->packageName << "\" \""<< module->categoryName
            <<"\" \""<< module->moduleName<<"\" "
            << x << " " << y << "]\n";

    }
    out << "\n";
    for(i=0;i<net->nconnections();i++){
        Connection* conn=net->connection(i);
	out << "addConnection $m";
	// Find the "from" module...
	int j;
	for(j=0;j<net->nmodules();j++){
	    Module* m=net->module(j);
	    if(conn->oport->get_module() == m){
	        out << j << " " << conn->oport->get_which_port();
		break;
	    }
	}
	out << " $m";
	for(j=0;j<net->nmodules();j++){
	    Module* m=net->module(j);
	    if(conn->iport->get_module() == m){
	        out << j << " " << conn->iport->get_which_port();
		break;
	    }
	}
	out << "\n";
    }
    out << "\n";
    // Emit variables...
    for(i=0;i<net->nmodules();i++){
        Module* module=net->module(i);
cerr << "Emit vars for " << module->name << endl;
	module->emit_vars(out);
    }

    for(i=0;i<net->nmodules();i++){
        Module* module=net->module(i);
        clString result;
	TCL::eval("winfo exists .ui"+module->id, result);
	int res;
	if(result.get_int(res) && res){
	    out << "$m" << i << " ui\n";
	}
    }
    // Let it rip...
    out << "\n";
//    out << "proc ok {} {\n";
    out << "::netedit scheduleok\n";
//    out << "}\n";
//    out << "\n";
    net->read_unlock();
}


void NetworkEditor::tcl_command(TCLArgs& args, void*)
{
    using SCICore::Containers::to_string;
    using namespace PSECore::Dataflow;

    if(args.count() < 2){
	args.error("netedit needs a minor command");
	return;
    }
    if(args[1] == "quit"){
	Thread::exitAll(0);
    } else if(args[1] == "addmodule"){
	if(args.count() < 5){
	    args.error("netedit addmodule needs a package name,"
                       "category name and module name");
	    return;
	}
	Module* mod=net->add_module(args[2],args[3],args[4]);
	if(!mod){
	    args.error("Module not found");
	    return;
	}
	// Add a TCL command for this module...
	TCL::add_command(mod->id+"-c", mod, 0);
	args.result(mod->id);
    } else if(args[1] == "deletemodule"){
	if(args.count() < 3){
	    args.error("netedit deletemodule needs a module name");
	    return;
	}
	Module* mod=net->get_module_by_id(args[2]);
	TCL::delete_command( mod->id+"-c" );
	if(!net->delete_module(args[2])){
	    args.error("Cannot delete module "+args[2]);
	}
    } else if(args[1] == "addconnection"){
	if(args.count() < 6){
	    args.error("netedit addconnection needs 4 args");
	    return;
	}
	Module* omod=net->get_module_by_id(args[2]);
	if(!omod){
	    args.error("netedit addconnection can't find output module");
	    return;
	}
	int owhich;
	if(!args[3].get_int(owhich)){
	    args.error("netedit addconnection can't parse owhich");
	    return;
	}
	Module* imod=net->get_module_by_id(args[4]);
	if(!imod){
	    args.error("netedit addconnection can't find input module");
	    return;
	}
	int iwhich;
	if(!args[5].get_int(iwhich)){
	    args.error("netedit addconnection can't parse iwhich");
	    return;
	}
	args.result(net->connect(omod, owhich, imod, iwhich));
    } else if(args[1] == "deleteconnection"){
	if (args.count() < 3){
	    args.error("netedit deleteconnection needs 1 arg");
	    return;
	}
	if (!net->disconnect(args[2])) {
	    args.error("Cannot find connection "+args[2]+" for deletion");
	}
    } else if(args[1] == "getconnected"){
	if(args.count() < 3){
	    args.error("netedit getconnections needs a module name");
	    return;
	}
	Module* mod=net->get_module_by_id(args[2]);
	if(!mod){
	    args.error("netedit addconnection can't find output module");
	    return;
	}
	Array1<clString> res;
	int i;
	for(i=0;i<mod->niports();i++){
	    Port* p=mod->iport(i);
	    for(int c=0;c<p->nconnections();c++){
		Connection* conn=p->connection(c);
		Array1<clString> cinfo(5);
		cinfo[0]=conn->id;
		cinfo[1]=conn->oport->get_module()->id;
		cinfo[2]=to_string(conn->oport->get_which_port());
		cinfo[3]=conn->iport->get_module()->id;
		cinfo[4]=to_string(conn->iport->get_which_port());
		res.add(args.make_list(cinfo));
	    }
	}
	for(i=0;i<mod->noports();i++){
	    Port* p=mod->oport(i);
	    for(int c=0;c<p->nconnections();c++){
		Connection* conn=p->connection(c);
		Array1<clString> cinfo(5);
		cinfo[0]=conn->id;
		cinfo[1]=conn->oport->get_module()->id;
		cinfo[2]=to_string(conn->oport->get_which_port());
		cinfo[3]=conn->iport->get_module()->id;
		cinfo[4]=to_string(conn->iport->get_which_port());
		res.add(args.make_list(cinfo));
	    }
	}
	args.result(args.make_list(res));
    } else if(args[1] == "packageNames") {
      args.result(args.make_list(packageDB.packageNames()));
    } else if(args[1] == "categoryNames") {
      if(args.count() != 3) {
        args.error("Usage: netedit categoryNames <packageName>");
        return;
      }
      args.result(args.make_list(packageDB.categoryNames(args[2])));
    } else if(args[1] == "moduleNames") {
      if(args.count() != 4) {
        args.error("Usage: netedit moduleNames <packageName> <categoryName>");
        return;
      }
      args.result(args.make_list(packageDB.moduleNames(args[2],args[3])));
    } else if(args[1] == "findiports"){
	// Find all of the iports in the network that have the same type
	// As the specified one...
	if(args.count() < 4){
	    args.error("netedit findiports needs a module name and port number");
	    return;
	}
	Module* mod=net->get_module_by_id(args[2]);
	if(!mod){
	    args.error("cannot find module "+args[2]);
	    return;
	}
	int which;
	if(!args[3].get_int(which) || which<0 || which>=mod->noports()){
	    args.error("bad port number");
	    return;
	}
	OPort* oport=mod->oport(which);
	Array1<clString> iports;
	for(int i=0;i<net->nmodules();i++){
	    Module* m=net->module(i);
	    for(int j=0;j<m->niports();j++){
		IPort* iport=m->iport(j);
		if(iport->nconnections() == 0 && 
		   oport->get_typename() == iport->get_typename()){
		    iports.add(args.make_list(m->id, to_string(j)));
		}
	    }
	}
	args.result(args.make_list(iports));
    } else if(args[1] == "findoports"){
	// Find all of the oports in the network that have the same type
	// As the specified one...
	if(args.count() < 4){
	    args.error("netedit findoports needs a module name and port number");
	    return;
	}
	Module* mod=net->get_module_by_id(args[2]);
	if(!mod){
	    args.error("cannot find module "+args[2]);
	    return;
	}
	int which;
	if(!args[3].get_int(which) || which<0 || which>=mod->niports()){
	    args.error("bad port number");
	    return;
	}
	IPort* iport=mod->iport(which);
	if(iport->nconnections() > 0){
	    // Already connected - none
	    args.result("");
	    return;
	}
	Array1<clString> oports;
	for(int i=0;i<net->nmodules();i++){
	    Module* m=net->module(i);
	    for(int j=0;j<m->noports();j++){
		OPort* oport=m->oport(j);
		if(oport->get_typename() == iport->get_typename()){
		    oports.add(args.make_list(m->id, to_string(j)));
		}
	    }
	}
	args.result(args.make_list(oports));
    } else if(args[1] == "dontschedule"){
	schedule=0;
    } else if(args[1] == "scheduleok"){
	schedule=1;
	mailbox.send(new Module_Scheduler_Message());
    } else if(args[1] == "reset_scheduler"){
        for(int i=0;i<net->nmodules();i++){
	    Module* m=net->module(i);
	    m->need_execute=0;
	}
    } else if(args[1] == "savenetwork"){
        if(args.count() < 3){
	    args.error("savenetwork needs a filename");
	    return;
	}
	save_network(args[2]);
    } else if (args[1] == "load_component_spec"){
      if (args.count()!=3) {
	args.error("load_component_spec needs 1 argument");
	return;
      }
      component_node* n = CreateComponentNode(1);
      ReadComponentNodeFromFile(n,args[2]());
      char string[100]="\0";
      sprintf(string,"%d",(long)n);
      TCL::execute(clString("GetPathAndPackage ")+string+" "+
		   n->name+" "+n->category);
    } else if (args[1] == "create_pac_cat_mod"){
        int check = 1;
        if (args.count()!=7) {
          args.error("create_pac_cat_mod needs 5 arguments");
          return;
        }
	GenPackage((char*)args[3](),(char*)args[2]());
	GenCategory((char*)args[4](),(char*)args[3](),(char*)args[2]());
	GenComponent((component_node*)atol(args[6]()),
		     (char*)args[3](),(char*)args[2]());
        if (!check) {
          args.error("create_pac_cat_mod failed.");
          return;
        }
    } else if (args[1] == "create_cat_mod"){
        int check = 1;
        if (args.count()!=7) {
          args.error("create_cat_mod needs 3 arguments");
          return;
        }
	GenCategory((char*)args[4](),(char*)args[3](),(char*)args[2]());
	GenComponent((component_node*)atol(args[6]()),
		     (char*)args[3](),(char*)args[2]());
	if (!check) {
	  args.error("create_cat_mod failed.");
	  return;
	}
    } else if (args[1] == "create_mod"){
        int check = 1;
        if (args.count()!=7) {
          args.error("create_mod needs 3 arguments");
          return;
        }
	GenComponent((component_node*)atol(args[6]()),
		     (char*)args[3](),(char*)args[2]());
	if (!check) {
	  args.error("create_mod failed.");
	  return;
	}
    } else if (args[1] == "set_group") {
	if (args.count()!=3) {
	    args.error("create_mod needs 1 argument");	
	    return;
	}
	
	cerr << "group name: args[2]";
	// group=args[2];
	    
    } else {
	args.error("Unknown minor command for netedit");
    }
}

void postMessage(const clString& errmsg, bool err)
{
  clString tag;
  if(err)
    tag += " errtag";
  TCL::execute(clString(".top.errorFrame.text insert end \"")+
	       errmsg+"\\n\""+tag);
  TCL::execute(".top.errorFrame.text see end");
}

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.12.2.3  2000/10/26 14:16:49  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.18  2000/10/24 05:57:41  moulding
// new module maker Phase 2: new module maker goes online
//
// These changes clean out the last remnants of the old module maker and
// bring the new module maker online.
//
// Revision 1.17  2000/10/23 09:19:47  moulding
// some changes to the new module maker.
//
// Revision 1.16  2000/10/21 18:35:10  moulding
// more work for new module maker.
//
// Revision 1.15  2000/08/31 15:25:47  nbenson
// modified save_network()
//
// Revision 1.14  2000/08/25 17:40:51  nbenson
// Modified save_network() member function as part of module-loading bug
// fix.  See /PSECore/GUI/NetworkEditor.tcl for more info.
//
// Revision 1.13  2000/06/07 00:02:32  moulding
// removed the package maker, ane added the module maker
//
// Revision 1.12  2000/03/20 21:50:13  yarden
// Linux port: replace the now defunct cuserid with getenv(LOGNAME)
//
// Revision 1.11  2000/03/18 00:17:38  moulding
// Experimental automatic package maker added
//
// Revision 1.10  1999/11/12 01:38:30  ikits
// Added ANL AVTC site visit modifications to make the demos work.
// Fixed bugs in PSECore/Datatypes/SoundPort.[h,cc] and PSECore/Dataflow/NetworkEditor.cc
// Put in temporary scale_changed fix into PSECore/Widgets/BaseWidget.cc
//
// Revision 1.9  1999/10/07 02:07:19  sparker
// use standard iostreams and complex type
//
// Revision 1.8  1999/09/08 02:26:41  sparker
// Various #include cleanups
//
// Revision 1.7  1999/08/30 18:47:52  kuzimmer
// Modified so that dataflow scripts can be read and written properly
//
// Revision 1.6  1999/08/29 00:46:50  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/28 17:54:30  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/23 06:30:33  sparker
// Linux port
// Added X11 configuration options
// Removed many warnings
//
// Revision 1.3  1999/08/17 06:38:24  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:59  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:20  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/06/21 23:52:49  dav
// updated makefiles.main
//
// Revision 1.2  1999/06/10 17:19:01  kuzimmer
// modified the savefile function to insert ::netedit dontschedule at the beginning of the saved script and ::netedit scheduleok at the end.
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//
