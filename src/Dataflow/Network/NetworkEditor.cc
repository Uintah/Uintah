/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/GuiInterface/TCLTask.h>
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


NetworkEditor::NetworkEditor(Network* net, GuiInterface* gui) :
  net(net), gui(gui)
{
  // Create User interface...
  gui->add_command("netedit", this, 0);
  ASSERT(sci_getenv("SCIRUN_SRCDIR"));
  gui->source_once(sci_getenv("SCIRUN_SRCDIR")+
                   string("/Dataflow/GUI/NetworkEditor.tcl"));
  gui->execute("makeNetworkEditor");
}

NetworkEditor::~NetworkEditor()
{
}


void
NetworkEditor::tcl_command(GuiArgs& args, void*)
{
  if (args.count() < 2) {
    throw "netedit needs a minor command";
  }
  if(args[1] == "quit") {
    TCLTask::unlock();
    Thread::exitAll(0);
  } else if (args[1] == "addmodule") {
    if(args.count() < 5)
      throw "netedit addmodule needs a package name,"
	" category name and module name";

    Module* mod = net->add_module(args[2],args[3],args[4]);
    if(!mod)
      throw "netedit addmodule cannot add module "+args[2]+args[3]+args[4];
    args.result(mod->id);
  } else if (args[1] == "deletemodule") {
    if(args.count() < 3)
      throw "netedit deletemodule needs a module name";

    if(!net->delete_module(args[2])) 
      throw GuiException("Cannot delete module "+args[2]);
  } else if (args[1] == "deletemodule_warn") {
    if(args.count() < 3)
      throw "netedit deletemodule_warn needs a module name";

    Module* mod=net->get_module_by_id(args[2]);
    // I don't think the following should happen, but due to what
    // I think is a race condition, it has happened.  This check 
    // avoids a core dump (bad memory access).
    if (!mod)
      throw "get_module_by_name failed for "+args[2];
    mod->delete_warn();
  } else if(args[1] == "addconnection") {
    if(args.count() < 6)
      throw "netedit addconnection needs 4 args";
    Module* omod=net->get_module_by_id(args[2]);
    int owhich = args.get_int(3);
    Module* imod=net->get_module_by_id(args[4]);
    int iwhich = args.get_int(5);

    if(!omod)
      throw "netedit addconnection can't find output module";
    if(!imod)
      throw "netedit addconnection can't find input module";

    if (imod->lastportdynamic && iwhich >= imod->iports.size())
      iwhich = imod->iports.size()-1;

    args.result(net->connect(omod, owhich, imod, iwhich));
  } else if(args[1] == "deleteconnection") {
    if (args.count() < 3)
      throw "netedit deleteconnection needs 1 arg";
    if (args.count() == 4 && args[3] == "1")
      net->disable_connection(args[2]);
    if (!net->disconnect(args[2]))
      throw "Cannot find connection "+args[2]+" for deletion";
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
  } else if(args[1] == "packageName"){
    if(args.count() != 3)
      throw "packageName needs a module id";
    Module* mod=net->get_module_by_id(args[2]);
    if(!mod)
      throw "cannot find module "+args[2];
    args.result(mod->packageName);
  } else if(args[1] == "categoryName"){
    if(args.count() != 3)
      throw "categoryName needs a module id";
    Module* mod=net->get_module_by_id(args[2]);
    if(!mod)
      throw "cannot find module "+args[2];
    args.result(mod->categoryName);
  } else if(args[1] == "moduleName"){
    if(args.count() != 3)
      throw "moduleName needs a module id";
    Module* mod=net->get_module_by_id(args[2]);
    if(!mod)
      throw "cannot find module "+args[2];
    args.result(mod->moduleName);
  } else if (args[1] == "create_pac_cat_mod") {
    if (args.count()!=7)
      throw "create_pac_cat_mod needs 5 arguments";
    component_node* n = CreateComponentNode(1);
    int check = ReadComponentNodeFromFile(n,args[6].c_str(), gui);
    if (check!=1)
      throw "NetworkEditor: 0) XML file did not pass validation: " + 
	args[2] + ".  Please see the messages window for details.";

    if (n->name==NOT_SET||n->category==NOT_SET)
      throw "NetworkEditor: XML file does not define"
	" a component name and/or does not define a"
	"  category: " + args[2];

    if (!(GenPackage((char*)args[3].c_str(),(char*)args[2].c_str()) &&
          GenCategory((char*)args[4].c_str(),(char*)args[3].c_str(),
                      (char*)args[2].c_str()) &&
          GenComponent(n, (char*)args[3].c_str(),(char*)args[2].c_str())))
      throw "Unable to create new package, category or module."
	"  Check your paths and names and try again.";
  } else if (args[1] == "create_cat_mod") {
    if (args.count()!=7)
      throw "create_cat_mod needs 3 arguments";
    component_node* n = CreateComponentNode(1);
    int check = ReadComponentNodeFromFile(n,args[6].c_str(), gui);
    if (check!=1)
      throw "NetworkEditor: 1) XML file did not pass validation: " + 
	args[2] + ".  Please see the messages window for details.";
    if (n->name==NOT_SET||n->category==NOT_SET)
      throw "NetworkEditor: XML file does not define"
	" a component name and/or does not define a"
	"  category: " + args[2];
    if (!(GenCategory((char*)args[4].c_str(),(char*)args[3].c_str(),
                      (char*)args[2].c_str()) &&
          GenComponent(n, (char*)args[3].c_str(),(char*)args[2].c_str())))
      throw "Unable to create new category or module."
	"  Check your paths and names and try again.";
  } else if (args[1] == "create_mod"){
    if (args.count()!=7) 
      throw "create_mod needs 3 arguments";
    component_node* n = CreateComponentNode(1);
    int check = ReadComponentNodeFromFile(n,args[6].c_str(), gui);
    if (check!=1)
      throw "NetworkEditor: 2) XML file did not pass validation: " + 
	args[2] + ".  Please see the messages window for details.";
    if (n->name==NOT_SET||n->category==NOT_SET)
      throw "NetworkEditor: XML file does not define"
	" a component name and/or does not define a"
	"  category: " + args[2];
    if (!(GenComponent(n, (char*)args[3].c_str(),(char*)args[2].c_str())))
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
    net->read_lock();
  } else if (args[1] == "net_read_unlock" && args.count() == 2){
    net->read_unlock();
  } else if (args[1] == "module_oport_datatypes") {
    if (args.count() != 5)
      throw "netedit module_oport_datatypes expects a "
	"package, category, and module";
    const ModuleInfo* info = packageDB->GetModuleInfo(args[4],args[3],args[2]);
    if (!info)
      throw "netedit module_oports cant find "+
	args[2]+"->"+args[3]+"->"+args[4];
    string result("");
    for (std::map<int,OPortInfo*>::iterator op = info->oports->begin(); 
         op != info->oports->end(); ++op) {
      result += (*op).second->datatype + " ";
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
    for (std::map<int,IPortInfo*>::iterator ip = info->iports->begin(); 
         ip != info->iports->end(); ++ip) {
      result += (*ip).second->datatype + " ";
    }
    args.result(result);
  } else if (args[1] == "presave") {
    for(int i=0;i<net->nmodules();i++)
      net->module(i)->presave();
  } else {
    throw "Unknown minor command for netedit";
  }
} // end tcl_command()
  
} // End namespace SCIRun
