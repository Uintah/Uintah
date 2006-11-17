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
#include <Core/Util/Environment.h>


#ifdef _WIN32
#include <io.h>
#endif
#include <stdio.h>
#include <iostream>
#include <sstream>
using namespace SCIRun;
using namespace std;


Network::Network()
  : the_lock("Network lock"), 
	  wait_delete_("Signal module deletion"),
		sched(0)
{
}

 
Network::~Network()
{
}


// For now, we just use a simple mutex for both reading and writing
void
Network::read_lock()
{
  the_lock.lock();
}


void
Network::read_unlock()
{
  the_lock.unlock();
}


void
Network::write_lock()
{
  the_lock.lock();
}


void
Network::write_unlock()
{
  the_lock.unlock();
}


int
Network::nmodules()
{
	the_lock.lock();
	int sz = static_cast<int>(modules.size());
	the_lock.unlock();
  return (sz);
}


Module*
Network::module(int i)
{
	the_lock.lock();
	Module* ptr = modules[i];
	the_lock.unlock();
  return ptr;
}


Connection*
Network::connection(int i)
{
	the_lock.lock();
	Connection* ptr = connections[i];
	the_lock.unlock();
  return ptr;
}


string
Network::connect(Module* m1, int p1, Module* m2, int p2)
{
  
  // dynamic port safeguard.
  if (m2->lastportdynamic_ && p2 >= m2->iports_.size()) {
    p2 = m2->iports_.size() - 1;
  }
  
  if (p1 >= m1->num_output_ports() || p2 >= m2->num_input_ports())
  {
    return "";
  }

  ostringstream ids;
  ids << m1->id_ << "_p" << p1 << "_to_" << m2->id_ << "_p" << p2;
  Connection* conn = scinew Connection(m1, p1, m2, p2, ids.str());
  
	the_lock.lock();
	connections.push_back(conn);

  // Reschedule next time we can.
  reschedule=1;
	the_lock.unlock();

  return conn->id;
}


int
Network::disconnect(const string& connId)
{
	the_lock.lock();
  unsigned int i;
  for (i = 0; i < connections.size(); i++)
  {
    if (connections[i]->id == connId)
    {
      break;
    }
  }
  if (i == connections.size())
  {
    return 0;
  }
 
  delete connections[i];
  connections.erase(connections.begin() + i);
	the_lock.unlock();
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


// SCIRunComponentModel uses Network::add_module2 to instantiate a
// SCIRun module in the SCIRun2 framework.
Module*
Network::add_module2(const string& packageName,
                     const string& categoryName,
                     const string& moduleName)
{
  Module* module = add_module(packageName, categoryName, moduleName);
  module->get_gui()->eval("addModule2 "+packageName+" "+categoryName+
			 " "+moduleName+" "+module->id_);
  return module;
}


Module*
Network::add_module(const string& packageName,
		    const string& categoryName,
		    const string& moduleName)
{ 
  const string name = 
    remove_spaces(packageName + "_" + categoryName + "_" + moduleName + "_");
  // Find a unique id in the Network for the new instance of this module and
  // form an instance name from it
  string instanceName;
  for (int i=0; get_module_by_id(instanceName = name + to_string(i)); i++);

  // Instantiate the module
  Module* mod = packageDB->instantiateModule(packageName, categoryName,
					     moduleName, instanceName);
  if (!mod) {
    cerr << "Error: can't create instance " << instanceName << "\n";
    return 0;
  }
	
	the_lock.lock();
  modules.push_back(mod);
	the_lock.unlock();

  // Binds NetworkEditor and Network instances to module instance.  
  // Instantiates ModuleHelper and starts event loop.
  mod->set_context(this);

  // add Module id and ptr to Module to hash table of modules in network
	the_lock.lock();
  module_ids[mod->id_] = mod;
  the_lock.unlock();
	
	mod->gui_->add_command(mod->id_+"-c", mod, 0);
  return mod;
}


void
Network::add_instantiated_module(Module* mod)
{
  if(!mod) {
    cerr << "Error: can't add instanated module\n";
    return;
  }
	
	the_lock.lock();
  modules.push_back(mod);
	the_lock.unlock();
	
  // Binds NetworkEditor and Network instances to module instance.
  // Instantiates ModuleHelper and starts event loop.
  mod->set_context(this);
  
  // add Module id and ptr to Module to hash table of modules in network
	the_lock.lock();
  module_ids[mod->id_] = mod;
  the_lock.unlock();
	
  GuiInterface* gui = mod->gui_;
  // Add a TCL command for this module...
  gui->add_command(mod->id_+"-c", mod, 0);

  gui->eval("addModule2 unknown unknown unknown "+mod->id_);
}


Module*
Network::get_module_by_id(const string& id)
{
	Module* m = 0;

	the_lock.lock();
  MapStringModule::iterator mod;
  mod = module_ids.find(id);
  if (mod != module_ids.end())
  {
    m = (*mod).second;
  }

	the_lock.unlock();
	
	return (m);
}



namespace SCIRun {

class DeleteModuleThread : public Runnable 
{
private:

  Network *net_;
  Module *module_;

public:

  DeleteModuleThread(Network *net, Module *module) :
    net_(net),
    module_(module)
  {
    ASSERT(net);
    ASSERT(module);
  }

  void run()
  {
		// Get module ID
		string mod_id = module_->get_id();

		// Remove module from list
		net_->write_lock();
    unsigned int vpos = 0;
    while (vpos<net_->modules.size() && net_->modules[vpos]!=module_) ++vpos;
    ASSERT(vpos<net_->modules.size());
    net_->modules.erase(net_->modules.begin()+vpos);		
		net_->write_unlock();

    delete module_;

		net_->write_lock();
    Network::MapStringModule::iterator const mpos = 
      net_->module_ids.find(mod_id);
    ASSERT(mpos != net_->module_ids.end());
    net_->module_ids.erase(mpos);


		//Signal that we are done with deleting
		net_->wait_delete_.conditionBroadcast();
		net_->write_unlock();
  }
};
}


// Delete_module will start a seperate thead that waits until a module
// is done executing before deleting it.
int
Network::delete_module(const string& id)
{
  Module* module_ptr = get_module_by_id(id);
  if (!module_ptr) return 0;

  DeleteModuleThread * dm = scinew DeleteModuleThread(this,module_ptr);
  const string tname("Delete module: " + id);
  Thread *mod_deleter = scinew Thread(dm, tname.c_str());
  mod_deleter->detach();
	
	// Wait until module is deleted
	//the_lock.lock();
	//while(module_ids.find(id) != module_ids.end())
	//{
	//	wait_delete_.wait(the_lock);
	//	the_lock.lock();
	//}
	//the_lock.unlock();
	
  return 1;
}



void
Network::schedule()
{
  sched->do_scheduling();
}


void
Network::schedule_all()
{
  if (sci_getenv_p("SCI_REGRESSION_TESTING"))
  {
    static int excounter = 0;
    excounter++;
    cout << "STARTING EXECUTION " + to_string(excounter) + "\n";
  }

  for(int i=0; i<nmodules(); i++)
  {
    Module* m = module(i);
    m->need_execute_ = true;
  }
  schedule();
}


void
Network::attach(Scheduler* _sched)
{
  sched = _sched;
}


void
Network::disable_connection(const string& connId)
{
	the_lock.lock();
	for (unsigned int i = 0; i < connections.size(); i++)
    if (connections[i]->id == connId)
    {
      connections[i]->disabled_ = true;
			the_lock.unlock();
      return;
    }
	the_lock.unlock();
}


Scheduler*
Network::get_scheduler()
{
  return sched;
}
