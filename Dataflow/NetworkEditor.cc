
/*
 *  NetworkEditor.cc: The network editor...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/NetworkEditor.h>

#include <Classlib/NotFinished.h>
#include <Comm/MessageBase.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Dataflow/Network.h>
#include <Dataflow/Port.h>
#include <Math/MiscMath.h>
#include <TCL/TCL.h>

NetworkEditor::NetworkEditor(Network* net)
: Task("Network Editor", 1), net(net),
  first_schedule(1), mailbox(100)
{
    // Create User interface...
    TCL::add_command("netedit", this, 0);
    TCL::source_once("Dataflow/NetworkEditor.tcl");
    TCL::execute("makeNetworkEditor");

    // Initialize the network
    net->initialize(this);

}

NetworkEditor::~NetworkEditor()
{
}

int NetworkEditor::body(int)
{
    // Go into Main loop...
    do_scheduling();
    main_loop();
    return 0;
}

void NetworkEditor::main_loop()
{
    // Dispatch events...
    int done=0;
    while(!done){
	MessageBase* msg=mailbox.receive();
	// Dispatch message....
	switch(msg->type){
	case MessageTypes::ReSchedule:
	    do_scheduling();
	    break;
	default:
	    cerr << "Unknown message type: " << msg->type << endl;
	    break;
	};
	delete msg;
    };
}

void NetworkEditor::do_scheduling()
{
    // Each Stream (Connection) has one of three properties
    // Dormant - not needed this time
    // New - Contains new data
    // Repeat - Another copy of old data
    //
    // New's are propogated downstream - never blocked
    // Repeats are propogated upstream - blocked by "sources"
    //
    // A module is considered 'New' if any of it's upstream modules are
    // receiving new data
    // A module is considered 'Repeat' if any of it's downstream modules
    // are 'Repeat'
    //
    int nmodules=net->nmodules();
#ifdef BROKEN
    int changed=1;
    int any_changed=1;
    while(changed){
	changed=0;
	int nmodules=net->nmodules();
	for(int i=0;i<nmodules;i++){
	    Module* mod=net->module(i);
	    changed |= mod->should_execute();
	}
	any_changed|=changed;
    }
    if(first_schedule || any_changed){
	// Do the scheduling...
	for(int i=0;i<nmodules;i++){
	    Module* module=net->module(i);

	    // Tell it to trigger...
	    if(module->sched_state != Module::SchedDormant){
		module->mailbox.send(new Scheduler_Module_Message);

		// Reset the state...
		module->sched_state=Module::SchedDormant;
	    }
	}
	first_schedule=0;
    }
#endif
    for(int i=0;i<nmodules;i++){
	Module* module=net->module(i);

	// Tell it to trigger...
	module->mailbox.send(new Scheduler_Module_Message);

	// Reset the state...
	module->sched_state=Module::SchedDormant;
    }
}

Scheduler_Module_Message::Scheduler_Module_Message()
: MessageBase(MessageTypes::ExecuteModule)
{
}

Scheduler_Module_Message::~Scheduler_Module_Message()
{
}

Module_Scheduler_Message::Module_Scheduler_Message()
: MessageBase(MessageTypes::ReSchedule)
{
}

Module_Scheduler_Message::~Module_Scheduler_Message()
{
}

void NetworkEditor::add_text(const clString& str)
{
    TCL::execute("$netedit_errortext insert end \""+str+"\n\"");
}

void NetworkEditor::tcl_command(TCLArgs& args, void*)
{
    if(args.count() < 2){
	args.error("netedit needs a minor command");
	return;
    }
    if(args[1] == "quit"){
	Task::exit_all(-1);
    } else if(args[1] == "addmodule"){
	if(args.count() < 3){
	    args.error("netedit addmodule needs a module name");
	    return;
	}
	Module* mod=net->add_module(args[2]);
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
	for(int i=0;i<mod->niports();i++){
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
    } else if(args[1] == "completelist"){
	ModuleCategory* allcat=ModuleList::get_all();
	ModuleCategoryIter iter(allcat);
	Array1<clString> mods(allcat->size());
	int i=0;
	for(iter.first();iter.ok();++iter){
	    mods[i++]=iter.get_key();
	}
	args.result(args.make_list(mods));
    } else if(args[1] == "catlist"){
	ModuleDB* db=ModuleList::get_db();
	Array1<clString> cats(db->size());
	ModuleDBIter dbiter(db);
	int idb=0;
	for(dbiter.first();dbiter.ok();++dbiter){
	    clString catname(dbiter.get_key());
	    ModuleCategory* cat=dbiter.get_data();
	    Array1<clString> mods(cat->size());
	    ModuleCategoryIter catiter(cat);
	    int ic=0;
	    for(catiter.first();catiter.ok();++catiter){
		clString modname(catiter.get_key());
		mods[ic++]=modname;
	    }
	    clString modlist(args.make_list(mods));
	    cats[idb++]=args.make_list(catname, modlist);
	}
	args.result(args.make_list(cats));
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
    } else {
	args.error("Unknown minor command for netedit");
    }
}
