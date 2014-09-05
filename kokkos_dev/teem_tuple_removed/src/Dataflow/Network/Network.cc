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
 *  Network.cc: The core of dataflow...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Distributed modifications by:
 *   Michelle Miller
 *   Dec. 1997
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>

#ifdef _WIN32
#include <io.h>
#endif
#include <stdio.h>
#include <iostream>
#include <sstream>
using namespace SCIRun;
using namespace std;

Network::Network()
  : the_lock("Network lock"), sched(0)
{
}
 
Network::~Network()
{
}

// For now, we just use a simple mutex for both reading and writing
void Network::read_lock()
{
    the_lock.lock();
}

void Network::read_unlock()
{
    the_lock.unlock();
}

void Network::write_lock()
{
    the_lock.lock();
}

void Network::write_unlock()
{
    the_lock.unlock();
}

int Network::nmodules()
{
    return modules.size();
}

Module* Network::module(int i)
{
    return modules[i];
}

int Network::nconnections()
{
    return connections.size();
}

Connection* Network::connection(int i)
{
    return connections[i];
}

string Network::connect(Module* m1, int p1, Module* m2, int p2)
{
    if (p1 >= m1->numOPorts() || p2 >= m2->numIPorts())
    {
      return "";
    }

    ostringstream ids;
    ids << m1->id << "_p" << p1 << "_to_" << m2->id << "_p" << p2;
    Connection* conn=scinew Connection(m1, p1, m2, p2, ids.str());
    connections.push_back(conn);

    // Reschedule next time we can.
    reschedule=1;

    return conn->id;
}

int
Network::disconnect(const string& connId)
{
  unsigned int i;
  for (i = 0; i < connections.size(); i++)
    if (connections[i]->id == connId)
      break;
  if (i == connections.size()) {
    return 0;
  }
 
  delete connections[i];
  connections.erase(connections.begin() + i);
  return 1;
}

static string
remove_spaces(const string& str)
{
  string result;
  for (string::const_iterator i = str.begin(); i != str.end(); i++)
  {
    if (*i != ' ') { result += *i; }
  }
  return result;
}


Module* Network::add_module2(const string& packageName,
			     const string& categoryName,
			     const string& moduleName)
{
  Module* module = add_module(packageName, categoryName, moduleName);

  GuiInterface* gui = module->gui;
  // Add a TCL command for this module...
  gui->add_command(module->id+"-c", module, 0);
  ostringstream command;
  command << "addModule2 " << packageName << " " << categoryName << " "
	 << moduleName << " " << module->id << '\n';
  gui->execute(command.str());
  return module;
}

Module* Network::add_module(const string& packageName,
                            const string& categoryName,
                            const string& moduleName)
{ 

  // Find a unique id in the Network for the new instance of this module and
  // form an instance name from it

  string instanceName;
  {
    const string name = remove_spaces(packageName + "_" +
				      categoryName + "_" +
				      moduleName + "_");
    for (int i=0; get_module_by_id(instanceName = name + to_string(i)); i++);
  }

  // Instantiate the module

  Module* mod = packageDB->instantiateModule(packageName, categoryName,
					     moduleName, instanceName);
  if(!mod) {
    cerr << "Error: can't create instance " << instanceName << "\n";
    return 0;
  }
  modules.push_back(mod);

  // Binds NetworkEditor and Network instances to module instance.  
  // Instantiates ModuleHelper and starts event loop.
  mod->set_context(sched, this);

  // add Module id and ptr to Module to hash table of modules in network
  module_ids[mod->id] = mod;
  
  return mod;
}

void Network::add_instantiated_module(Module* mod)
{
  if(!mod) {
    cerr << "Error: can't add instanated module\n";
    return;
  }
  modules.push_back(mod);

  // Binds NetworkEditor and Network instances to module instance.
  // Instantiates ModuleHelper and starts event loop.
  mod->set_context(sched, this);
  
  // add Module id and ptr to Module to hash table of modules in network
  module_ids[mod->id] = mod;

  GuiInterface* gui = mod->gui;
  // Add a TCL command for this module...
  gui->add_command(mod->id+"-c", mod, 0);
  ostringstream command;

  string packageName = "unknown";
  string categoryName = "unknown";
  string moduleName = "unknonw";
  command << "addModule2 " << packageName << " " << categoryName << " "
	 << moduleName << " " << mod->id << '\n';
  gui->execute(command.str());
}

Module* Network::get_module_by_id(const string& id)
{
    MapStringModule::iterator mod;
    mod = module_ids.find(id);
    if (mod != module_ids.end()) {
	return (*mod).second;
    } else {
	return 0;
    }
}

namespace SCIRun {
class DeleteModuleThread : public Runnable 
{
public:

  DeleteModuleThread(Network* n, const string& i) :
    net_(n),
    id_(i),
    status_(1)
  {}

  void run() {
    Module* mod = net_->get_module_by_id(id_);
    if (!mod) {
      status_ = 0;
      return;
    }
    // traverse array of ptrs to Modules in Network to find this module
    unsigned int i;
    for (i = 0; i < net_->modules.size(); i++)
      if (net_->modules[i] == mod)
	break;
    if (i == net_->modules.size()) {
      status_ = 0;
      return;
    }

    // remove array element corresponding to module, remove from hash table
    net_->modules.erase(net_->modules.begin() + i);
    mod->kill_helper();
    delete mod;
  }
  
  int status() { return status_; }
private:
  Network          *net_;
  string            id_;
  int               status_;
};
}

int Network::delete_module(const string& id)
{
  DeleteModuleThread * dm = scinew DeleteModuleThread((Network*)this, id);
  string tname("Delete module: " + id);
  Thread *mod_deleter = scinew Thread(dm, tname.c_str());
  int rval = dm->status();
  mod_deleter->detach();
  return rval;
}

void Network::schedule()
{
  sched->do_scheduling();
}

void Network::attach(Scheduler* _sched)
{
  sched=_sched;
}

void
Network::disable_connection(const string& connId)
{
  for (unsigned int i = 0; i < connections.size(); i++)
    if (connections[i]->id == connId)
    {
      connections[i]->disabled_ = true;
      return;
    }
}


Scheduler* Network::get_scheduler()
{
  return sched;
}
