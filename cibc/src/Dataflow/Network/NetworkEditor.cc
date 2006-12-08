/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#  pragma warning(disable:4786)
#  define EXPERIMENTAL_TCL_THREAD
#endif

#include <Dataflow/Network/NetworkEditor.h>

#include <Dataflow/Comm/MessageBase.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkIO.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Ports/Port.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/Network/GenFiles.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Dataflow/GuiInterface/GuiCallback.h>
#include <Dataflow/GuiInterface/GuiInterface.h>
#include <Dataflow/GuiInterface/TCLTask.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/sci_system.h>
#include <Core/Util/Environment.h>
#include <Core/Exceptions/GuiException.h>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
using namespace std;
  
namespace SCIRun {

static bool
scheduling_starting_callback(void *data)
{
  const string proc("set_network_executing 1");
  GuiInterface* gui = (GuiInterface*)data;
  gui->execute(proc);
  return true;
}

static bool
scheduling_done_callback(void *data)
{
  const string proc("set_network_executing 0");
  GuiInterface* gui = (GuiInterface*)data;
  gui->execute(proc);
  return true;
}

// init the static pointer. 
Network* NetworkEditor::net_ = 0;

NetworkEditor::NetworkEditor(Network* net, GuiInterface* gui) :
  gui_(gui)
{
  net_ = net;
  net->get_scheduler()->add_start_callback(scheduling_starting_callback, gui);
  net->get_scheduler()->add_callback(scheduling_done_callback, gui);

  // Create User interface...
  gui_->add_command("netedit", this, 0);
  ASSERT(sci_getenv("SCIRUN_SRCDIR"));
  gui_->source_once(sci_getenv("SCIRUN_SRCDIR")+
                   string("/Dataflow/GUI/NetworkEditor.tcl"));
  gui_->execute("makeNetworkEditor");
}

NetworkEditor::~NetworkEditor()
{
}


void
NetworkEditor::tcl_command(GuiArgs& args, void*)
{
  static NetworkIO *netio = 0;
  if (args.count() < 2) {
    throw "netedit needs a minor command";
  }
  if(args[1] == "quit") {
#ifndef EXPERIMENTAL_TCL_THREAD
    TCLTask::unlock();
#endif
    Thread::exitAll(0);
  } else if (args[1] == "addmodule") {
    if(args.count() < 5)
      throw "netedit addmodule needs a package name,"
	" category name and module name";

    Module* mod = net_->add_module(args[2],args[3],args[4]);
    if(!mod)
      throw "netedit addmodule cannot add module "+args[2]+args[3]+args[4];
    gui_->add_command(mod->id_+"-c", mod, 0);
    args.result(mod->id_);
  } else if (args[1] == "deletemodule") {
    if(args.count() < 3)
      throw "netedit deletemodule needs a module name";

    if(!net_->delete_module(args[2])) 
      throw GuiException("Cannot delete module "+args[2]);
  } else if (args[1] == "deletemodule_warn") {
    if(args.count() < 3)
      throw "netedit deletemodule_warn needs a module name";

    Module* mod=net_->get_module_by_id(args[2]);
    // I don't think the following should happen, but due to what
    // I think is a race condition, it has happened.  This check 
    // avoids a core dump (bad memory access).
    if (!mod)
      throw "get_module_by_name failed for "+args[2];
    mod->delete_warn();
  } else if(args[1] == "addconnection") {
    if(args.count() < 6)
      throw "netedit addconnection needs 4 args";
    Module* omod = net_->get_module_by_id(args[2]);
    if(!omod)
      throw "netedit addconnection can't find output module";
    int owhich = args.get_int(3);
    Module* imod = net_->get_module_by_id(args[4]);
    if(!imod)
      throw "netedit addconnection can't find input module";
    int iwhich = args.get_int(5);
    args.result(net_->connect(omod, owhich, imod, iwhich));
  } else if(args[1] == "deleteconnection") {
    if (args.count() < 3)
      throw "netedit deleteconnection needs 1 arg";
    if (args.count() == 4 && args[3] == "1")
      net_->disable_connection(args[2]);
    if (!net_->disconnect(args[2]))
      throw "Cannot find connection "+args[2]+" for deletion";
  } else if(args[1] == "supportsPortCaching") {
    if(args.count() < 4)
      throw "netedit supportsPortCaching needs 2 args";
    Module* omod = net_->get_module_by_id(args[2]);
    if(!omod)
    {
      args.result("0");
    }
    else
    {
      const int owhich = args.get_int(3);
      if (owhich >= omod->num_output_ports())
        throw "netedit supportsPortCaching can't find output port";
      args.result(omod->oport_supports_cache_flag(owhich)?"1":"0");
    }
  } else if(args[1] == "isPortCaching") {
    if(args.count() < 4)
      throw "netedit isPortCaching needs 4 args";
    Module* omod = net_->get_module_by_id(args[2]);
    if(!omod)
      throw "netedit isPortCaching can't find output module";
    const int owhich = args.get_int(3);
    if (owhich >= omod->num_output_ports())
      throw "netedit isPortCaching can't find output port";
    args.result(omod->get_oport_cache_flag(owhich)?"1":"0");
  } else if(args[1] == "setPortCaching") {
    if(args.count() < 5)
      throw "netedit setPortCaching needs 5 args";
    Module* omod = net_->get_module_by_id(args[2]);
    if(!omod)
      throw "netedit setPortCaching can't find output module";
    const int owhich = args.get_int(3);
    if (owhich >= omod->num_output_ports())
      throw "netedit setPortCaching can't find output port";
    const int cache =  args.get_int(4);
    omod->set_oport_cache_flag(owhich, cache);
  } else if(args[1] == "packageNames") {
    args.result(args.make_list(packageDB->packageNames()));
  } else if(args[1] == "categoryNames") {
    if(args.count() != 3)
      throw "Usage: netedit categoryNames <packageName>";
    args.result(args.make_list(packageDB->categoryNames(args[2])));
  } else if(args[1] == "moduleNames") {
    if(args.count() != 4)
      throw "Usage: netedit moduleNames <packageName> <categoryName>";
    args.result(args.make_list(packageDB->moduleNames(args[2],args[3])));
  } else if(args[1] == "getCategoryName") {
    if(args.count() != 5)
      throw "Usage: netedit getCategoryName"
	"<packageName> <categoryName> <moduleName>";
    args.result(packageDB->getCategoryName(args[2], args[3], args[4]));
  } else if(args[1] == "dontschedule"){
  } else if(args[1] == "scheduleok"){
    net_->schedule();
  } else if(args[1] == "scheduleall"){
    net_->schedule_all();
  } else if(args[1] == "reset_scheduler"){
    for(int i=0;i<net_->nmodules();i++){
      Module* m=net_->module(i);
      m->need_execute_=0;
    }
  } else if(args[1] == "packageName"){
    if(args.count() != 3)
      throw "packageName needs a module id";
    Module* mod=net_->get_module_by_id(args[2]);
    if(!mod)
      throw "cannot find module "+args[2];
    args.result(mod->package_name_);
  } else if(args[1] == "categoryName"){
    if(args.count() != 3)
      throw "categoryName needs a module id";
    Module* mod=net_->get_module_by_id(args[2]);
    if(!mod)
      throw "cannot find module "+args[2];
    args.result(mod->category_name_);
  } else if(args[1] == "moduleName"){
    if(args.count() != 3)
      throw "moduleName needs a module id";
    Module* mod=net_->get_module_by_id(args[2]);
    if(!mod)
      throw "cannot find module "+args[2];
    args.result(mod->module_name_);
  } else if (args[1] == "create_pac_cat_mod") {
    if (args.count()!=7)
      throw "create_pac_cat_mod needs 5 arguments";
    ModuleInfo mi;
    bool success = read_component_file(mi, args[6].c_str());
    if (! success)
      throw "NetworkEditor: 0) XML file did not pass validation: " + 
	args[2] + ".  Please see the messages window for details.";

    if (!(GenPackage((char*)args[3].c_str(),(char*)args[2].c_str()) &&
          GenCategory((char*)args[4].c_str(),(char*)args[3].c_str(),
                      (char*)args[2].c_str()) &&
          GenComponent(mi, (char*)args[3].c_str(),(char*)args[2].c_str())))
      throw "Unable to create new package, category or module."
	"  Check your paths and names and try again.";
  } else if (args[1] == "create_cat_mod") {
    if (args.count()!=7)
      throw "create_cat_mod needs 3 arguments";
    ModuleInfo mi;
    bool success = read_component_file(mi, args[6].c_str());
    if (!success)
      throw "NetworkEditor: 1) XML file did not pass validation: " + 
	args[2] + ".  Please see the messages window for details.";

    if (!(GenCategory((char*)args[4].c_str(),(char*)args[3].c_str(),
                      (char*)args[2].c_str()) &&
          GenComponent(mi, (char*)args[3].c_str(),(char*)args[2].c_str())))
      throw "Unable to create new category or module."
	"  Check your paths and names and try again.";
  } else if (args[1] == "create_mod"){
    if (args.count()!=7) 
      throw "create_mod needs 3 arguments";
    ModuleInfo mi;
    bool success = read_component_file(mi, args[6].c_str());
    if (! success)
      throw "NetworkEditor: 2) XML file did not pass validation: " + 
	args[2] + ".  Please see the messages window for details.";
    if (!(GenComponent(mi, (char*)args[3].c_str(),(char*)args[2].c_str())))
      throw "Unable to create new module."
	"  Check your paths and names and try again.";
  } else if (args[1] == "sci_system" && args.count() > 2) {
    string command = args[2];
    for (int i = 3; i < args.count(); i++) {
      command = command + " " + args[i];
    }
    args.result(to_string(sci_system(command.c_str())));
  } else if (args[1] == "getenv" && args.count() == 3){
    const char *result = sci_getenv( args[2] );
    if (result) {
      args.result(string(result));
    }
  } else if (args[1] == "setenv" && args.count() == 4){
    sci_putenv(args[2], args[3]);
  } else if (args[1] == "net_read_lock" && args.count() == 2){
    net_->read_lock();
  } else if (args[1] == "net_read_unlock" && args.count() == 2){
    net_->read_unlock();
  } else if (args[1] == "module_oport_datatypes") {
    if (args.count() != 5)
      throw "netedit module_oport_datatypes expects a "
	"package, category, and module";
    const ModuleInfo* info = packageDB->GetModuleInfo(args[4],args[3],args[2]);
    if (!info)
      throw "netedit module_oports cant find "+
	args[2]+"->"+args[3]+"->"+args[4];
    string result("");

    vector<OPortInfo*>::const_iterator i2 = info->oports_.begin();
    while (i2 < info->oports_.end())
    {
      OPortInfo* op = *i2++;
      result += op->datatype + " ";
    }
    args.result(result);
  } else if (args[1] == "module_iport_datatypes") {
    if (args.count() != 5)
      throw "netedit module_iport_datatypes expects a "
	"package, category, and module";
    const ModuleInfo* info = packageDB->GetModuleInfo(args[4],args[3],args[2]);
    if (!info)
      throw "netedit module_oports cant find "+
	args[2]+"->"+args[3]+"->"+args[4];
    string result("");
    vector<IPortInfo*>::const_iterator i1 = info->iports_.begin();
    while (i1 < info->iports_.end())
    {
      IPortInfo* ip = *i1++;
      result += ip->datatype + " ";
    }
    args.result(result);
  } else if (args[1] == "presave") {
    for(int i=0;i<net_->nmodules();i++)
      net_->module(i)->presave();
  } else if (args[1] == "start-net-doc") {
    if (netio) {
      delete netio;
      netio = 0;
    }
    netio = new NetworkIO();
    netio->start_net_doc(args[2], args[3]);
  } else if (args[1] == "write-net-doc") {
    netio->write_net_doc();
  } else if (args[1] == "network-variable") {
    netio->add_net_var(args[2], args[3]);
  } else if (args[1] == "net-add-env-var") {
    netio->add_environment_sub(args[2], args[3]);
  } else if (args[1] == "network-note") {
    netio->add_net_note(args[2]);
  } else if (args[1] == "add-module") {
    netio->add_module_node(args[2], args[3], args[4], args[5]);
  } else if (args[1] == "module-position") {
    netio->add_module_position(args[2], args[3], args[4]);
  } else if (args[1] == "mod-note") {
    netio->add_module_note(args[2], args[3]);
  } else if (args[1] == "mod-note-pos") {
    netio->add_module_note_position(args[2], args[3]);
  } else if (args[1] == "mod-note-col") {
    netio->add_module_note_color(args[2], args[3]);
  } else if (args[1] == "mod-connection") {
    netio->add_connection_node(args[2], args[3], args[4], args[5], args[6]);
  } else if (args[1] == "conn-disabled") {
    netio->set_disabled_connection(args[2]);
  } else if (args[1] == "conn-route") {
    netio->add_connection_route(args[2], args[3]);
  } else if (args[1] == "conn-note") {
    netio->add_connection_note(args[2], args[3]);
  } else if (args[1] == "conn-note-pos") {
    netio->add_connection_note_position(args[2], args[3]);
  } else if (args[1] == "conn-note-col") {
    netio->add_connection_note_color(args[2], args[3]);
  } else if (args[1] == "set-port-caching") {
    netio->set_port_caching(args[2], args[3], args[4]);
  } else if (args[1] == "set-modgui-visible") {
    netio->set_module_gui_visible(args[2]);
  } else if (args[1] == "add-mod-var") {
    netio->add_module_variable(args[2], args[3], args[4],false,false);
  } else if (args[1] == "add-mod-substvar") {
    netio->add_module_variable(args[2], args[3], args[4],false,true);
  } else if (args[1] == "add-mod-filevar") {
		if (args[5] == "1" || args[5] == "true" || args[5] == "yes" || args[5] == "on") netio->add_module_variable(args[2], args[3], args[4],true,true,true);
		else netio->add_module_variable(args[2], args[3], args[4],true,true,false);
  } else if (args[1] == "add-modgui-callback") {
    netio->add_module_gui_callback(args[2], args[3]);
  } else if (args[1] == "subnet-start") {
    netio->push_subnet_scope(args[2], args[3]);
  } else if (args[1] == "subnet-end") {
    netio->pop_subnet_scope();
  } else if (args[1] == "load_srn") {
    NetworkIO::load_net(args[2]);
    NetworkIO ln;
    ln.load_network();
  } else  {
    throw "Unknown minor command for netedit";
  }
} // end tcl_command()
  
} // End namespace SCIRun
