/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  NetworkEditor.cc: The network editor...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Distributed Dataflow changes:
 *   Michelle Miller
 *   Nov. 1997
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef _WIN32
#pragma warning(disable:4786)
#endif

#include <Dataflow/Network/NetworkEditor.h>

#include <Dataflow/Comm/MessageBase.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Port.h>
#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/Network/GenFiles.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/GuiInterface/TCLstrbuff.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/sci_system.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
using namespace std;
  
namespace SCIRun {


NetworkEditor::NetworkEditor(Network* net, GuiInterface* gui)
  : net(net), gui(gui)
{
  // Create User interface...
  gui->add_command("netedit", this, 0);
  gui->source_once("$DataflowTCL/NetworkEditor.tcl");
  gui->execute("makeNetworkEditor");
}

NetworkEditor::~NetworkEditor()
{
}


static void
emit_tclstyle_copyright(ostream &out)
{
  out <<
    "#\n"
    "# The contents of this file are subject to the University of Utah Public\n"
    "# License (the \"License\"); you may not use this file except in compliance\n"
    "# with the License.\n"
    "# \n"
    "# Software distributed under the License is distributed on an \"AS IS\"\n"
    "# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the\n"
    "# License for the specific language governing rights and limitations under\n"
    "# the License.\n"
    "# \n"
    "# The Original Source Code is SCIRun, released March 12, 2001.\n"
    "# \n"
    "# The Original Source Code was developed by the University of Utah.\n"
    "# Portions created by UNIVERSITY are Copyright (C) 2001, 1994 \n"
    "# University of Utah. All Rights Reserved.\n"
    "\n"
    "set results [sourceSettingsFile]\n"
    "\n"
"if { $results == \"failed\" } {\n"
    "\n"
    "    ::netedit scheduleok\n"
    "    return \n"
    "\n"
"} else {\n"
    "\n"
    "    set DATADIR [lindex $results 0]\n"
    "    set DATASET [lindex $results 1]\n"
    "}\n"
    "\n"
    "source $DATADIR/$DATASET/$DATASET.settings\n";
}

void NetworkEditor::save_network(const string& filename, 
				 const string &subnet_num)
{
    ofstream out(filename.c_str());

    if(!out)
      return;
    out << "# SCI Network 1.20\n";
    if (getenv("SCI_INSERT_NET_COPYRIGHT")) { emit_tclstyle_copyright(out); }
    out << "\n";
    out << "::netedit dontschedule\n\n";
    net->read_lock();

    // Added by Mohamed Dekhil for saving extra information
    gui->lock();

    string myvalue;
    if (!gui->get("userName", myvalue)){
      out << "global userName\nset userName \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }
    if (!gui->get("runDate", myvalue)){
      out << "global runDate\nset runDate \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }
    if (!gui->get("runTime", myvalue)){
      out << "global runTime\nset runTime \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }
    if (!gui->get("notes", myvalue)){
      out << "global notes\nset notes \"" << myvalue << "\"\n" ;
      out << "\n" ;
    }
    gui->unlock();
   

    out.close();
    net->read_unlock();
    gui->execute("writeSubnetModulesAndConnections {"+filename+"} "+subnet_num);
    net->read_lock();
    out.open(filename.c_str(), ofstream::out | ofstream::app);

    int i;
    // Emit variables...
    string midx;
    for(i=0;i<net->nmodules();i++){
        Module* module=net->module(i);
	gui->eval("modVarName "+module->id, midx);
	if (midx.size()) {
	  module->emit_vars(out, midx);
	}
    }

    for(i=0;i<net->nmodules();i++){
        Module* module=net->module(i);
	gui->eval("modVarName "+module->id, midx);
	if (midx.size()) {
	  string result;
	  gui->eval("winfo exists .ui" + module->id, result);
	  int res;
	  if(string_to_int(result, res) && (res == 1)) {
	    out << midx << " initialize_ui\n";
	  }
	}
    }
    out << "\n";
    out << "::netedit scheduleok\n";
    net->read_unlock();
}


void NetworkEditor::tcl_command(GuiArgs& args, void*)
{
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
	if(mod){
	  // Add a TCL command for this module...
	  gui->add_command(mod->id+"-c", mod, 0);
	  args.result(mod->id);
	}
    } else if(args[1] == "deletemodule"){
	if(args.count() < 3){
	    args.error("netedit deletemodule needs a module name");
	    return;
	}
	Module* mod=net->get_module_by_id(args[2]);
	gui->delete_command( mod->id+"-c" );
	if(!net->delete_module(args[2])){
	    args.error("Cannot delete module "+args[2]);
	}
    } else if(args[1] == "deletemodule_warn"){
	if(args.count() < 3){
	    args.error("netedit deletemodule_warn needs a module name");
	    return;
	}
	Module* mod=net->get_module_by_id(args[2]);
	mod->delete_warn();
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
	if(!string_to_int(args[3], owhich)) {
	    args.error("netedit addconnection can't parse owhich");
	    return;
	}
	Module* imod=net->get_module_by_id(args[4]);
	if(!imod){
	    args.error("netedit addconnection can't find input module");
	    return;
	}
	int iwhich;
	if(!string_to_int(args[5], iwhich)) {
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
    } else if(args[1] == "blockconnection"){
	if (args.count() < 3){
	    args.error("netedit blockconnection needs 1 arg");
	    return;
	}
	net->block_connection(args[2]);
    } else if(args[1] == "unblockconnection"){
	if (args.count() < 3){
	    args.error("netedit unblockconnection needs 1 arg");
	    return;
	}
	net->unblock_connection(args[2]);
    } else if(args[1] == "getconnected"){
	if(args.count() < 3){
	    args.error("netedit getconnected needs a module name");
	    return;
	}
	Module* mod=net->get_module_by_id(args[2]);
	if(!mod){
	    args.error("netedit getconnected can't find output module");
	    return;
	}
	vector<string> res;
	int i;
	for(i=0;i<mod->numIPorts();i++){
	    Port* p=mod->getIPort(i);
	    for(int c=0;c<p->nconnections();c++){
		Connection* conn=p->connection(c);
		vector<string> cinfo(4);
		cinfo[0]=conn->oport->get_module()->id;
		cinfo[1]=to_string(conn->oport->get_which_port());
		cinfo[2]=conn->iport->get_module()->id;
		cinfo[3]=to_string(conn->iport->get_which_port());
		res.push_back(args.make_list(cinfo));
	    }
	}
	for(i=0;i<mod->numOPorts();i++){
	    Port* p=mod->getOPort(i);
	    for(int c=0;c<p->nconnections();c++){
		Connection* conn=p->connection(c);
		vector<string> cinfo(4);
		cinfo[0]=conn->oport->get_module()->id;
		cinfo[1]=to_string(conn->oport->get_which_port());
		cinfo[2]=conn->iport->get_module()->id;
		cinfo[3]=to_string(conn->iport->get_which_port());
		res.push_back(args.make_list(cinfo));
	    }
	}
	args.result(args.make_list(res));
    } else if(args[1] == "packageNames") {
      args.result(args.make_list(packageDB->packageNames()));
    } else if(args[1] == "categoryNames") {
      if(args.count() != 3) {
        args.error("Usage: netedit categoryNames <packageName>");
        return;
      }
      args.result(args.make_list(packageDB->categoryNames(args[2])));
    } else if(args[1] == "moduleNames") {
      if(args.count() != 4) {
        args.error("Usage: netedit moduleNames <packageName> <categoryName>");
        return;
      }
      args.result(args.make_list(packageDB->moduleNames(args[2],args[3])));
    } else if(args[1] == "getCategoryName") {
      if(args.count() != 5) {
	args.error("Usage: netedit getCategoryName <packageName> <categoryName> <moduleName>");
	return;
      }
      args.result(packageDB->getCategoryName(args[2], args[3], args[4]));
    } else if(args[1] == "dontschedule"){
    } else if(args[1] == "scheduleok"){
	net->schedule();
    } else if(args[1] == "scheduleall"){
        for(int i=0;i<net->nmodules();i++){
	    Module* m=net->module(i);
	    m->need_execute=1;
	}
	net->schedule();
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
	string filename = args[2];
	for (int i = 3; i < args.count() - 1; i++)
	  filename = filename +" "+args[i];
	save_network(filename,args[args.count()-1]);
    } else if(args[1] == "packageName"){
        if(args.count() != 3){
	    args.error("packageName needs a module id");
	}
	Module* mod=net->get_module_by_id(args[2]);
	if(!mod){
	  args.error("cannot find module "+args[2]);
	  return;
	}
	args.result(mod->packageName);
	return;
    } else if(args[1] == "categoryName"){
        if(args.count() != 3){
	    args.error("categoryName needs a module id");
	}
	Module* mod=net->get_module_by_id(args[2]);
	if(!mod){
	  args.error("cannot find module "+args[2]);
	  return;
	}
	args.result(mod->categoryName);
	return;
    }  else if(args[1] == "moduleName"){
        if(args.count() != 3){
	    args.error("moduleName needs a module id");
	}
	Module* mod=net->get_module_by_id(args[2]);
	if(!mod){
	  args.error("cannot find module "+args[2]);
	  return;
	}
	args.result(mod->moduleName);
	return;
    } else if (args[1] == "create_pac_cat_mod"){
      if (args.count()!=7) {
          args.error("create_pac_cat_mod needs 5 arguments");
          return;
      }
      component_node* n = CreateComponentNode(1);
      int check = ReadComponentNodeFromFile(n,args[6].c_str(), gui);
      if (check!=1) {
	args.error("NetworkEditor: XML file did not pass validation: " + 
		   args[2] + ".  Please see the messages window for details.");
	return;
      }
      if (n->name==NOT_SET||n->category==NOT_SET) {
	args.error("NetworkEditor: XML file does not define"
		   " a component name and/or does not define a"
		   "  category: " + args[2]);
	return;
      }
      if (!(GenPackage((char*)args[3].c_str(),(char*)args[2].c_str()) &&
	    GenCategory((char*)args[4].c_str(),(char*)args[3].c_str(),
			(char*)args[2].c_str()) &&
	    GenComponent(n, (char*)args[3].c_str(),(char*)args[2].c_str()))) {
        args.error("Unable to create new package, category or module."
		   "  Check your paths and names and try again.");
	return;
      }
    } else if (args[1] == "create_cat_mod"){
      if (args.count()!=7) {
	args.error("create_cat_mod needs 3 arguments");
	return;
      }
      component_node* n = CreateComponentNode(1);
      int check = ReadComponentNodeFromFile(n,args[6].c_str(), gui);
      if (check!=1) {
	args.error("NetworkEditor: XML file did not pass validation: " + 
		   args[2] + ".  Please see the messages window for details.");
	return;
      }
      if (n->name==NOT_SET||n->category==NOT_SET) {
	args.error("NetworkEditor: XML file does not define"
		   " a component name and/or does not define a"
		   "  category: " + args[2]);
	return;
      }
      
      if (!(GenCategory((char*)args[4].c_str(),(char*)args[3].c_str(),
			(char*)args[2].c_str()) &&
	    GenComponent(n, (char*)args[3].c_str(),(char*)args[2].c_str()))) {
        args.error("Unable to create new category or module."
		   "  Check your paths and names and try again.");
	return;
      }
    } else if (args[1] == "create_mod"){
      if (args.count()!=7) {
          args.error("create_mod needs 3 arguments");
        return;
      }
      component_node* n = CreateComponentNode(1);
      int check = ReadComponentNodeFromFile(n,args[6].c_str(), gui);
      if (check!=1) {
	args.error("NetworkEditor: XML file did not pass validation: " + 
		   args[2] + ".  Please see the messages window for details.");
	return;
      }
      if (n->name==NOT_SET||n->category==NOT_SET) {
	args.error("NetworkEditor: XML file does not define"
		   " a component name and/or does not define a"
		   "  category: " + args[2]);
	return;
      }
      if (!(GenComponent(n, (char*)args[3].c_str(),(char*)args[2].c_str()))) {
          args.error("Unable to create new module."
		     "  Check your paths and names and try again.");
	return;
      }
    } else if (args[1] == "sci_system" && args.count() > 2){
      string command = args[2];
      for (int i = 3; i < args.count(); i++) {
	command = command + " " + args[i];
      }
      sci_system(command.c_str());
      return;
    } else {
	args.error("Unknown minor command for netedit");
    }
}
} // End namespace SCIRun
