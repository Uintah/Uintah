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
 *  Network.h: Interface to Network class from project
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

#ifndef SCI_project_Network_h
#define SCI_project_Network_h 1

#include <Dataflow/share/share.h>
#include <Core/Thread/Mutex.h>
#include <string>
#include <vector>
#include <map>

namespace SCIRun {

using std::string;
using std::vector;
using std::map;

class Connection;
class Module;
class GuiInterface;
class Scheduler;
class DeleteModuleThread;

// CollabVis code begin
class ViewServer;
// CollabVis code end
  
class PSECORESHARE Network {
  friend class DeleteModuleThread;
public:
    
  typedef map<string, Connection*>	MapStringConnection;
  typedef map<string, Module*>	MapStringModule;
    
private:
  Mutex the_lock;

  MapStringConnection conn_ids;
    
  vector<Connection*> connections;
  vector<Module*> modules;
    
  MapStringModule module_ids;
    
  int reschedule;
  Scheduler* sched;

public:
  Network();
  ~Network();

  void read_lock();
  void read_unlock();
  void write_lock();
  void write_unlock();

  int nmodules();
  Module* module(int);

  int nconnections();
  Connection* connection(int);
  string connect(Module*, int, Module*, int);
  int disconnect(const string&);
  void block_connection(const string&);
  void unblock_connection(const string&);
    
  Module* add_module(const string& packageName,
		     const string& categoryName,
		     const string& moduleName);
  // For SCIRun2
  Module* add_module2(const string& packageName,
		      const string& categoryName,
		      const string& moduleName);
  int delete_module(const string& name);

  Module* get_module_by_id(const string& id);

  void schedule();
  void attach(Scheduler*);

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  ViewServer *server;
#endif
  // CollabVis code end
};

} // End namespace SCIRun


#endif
