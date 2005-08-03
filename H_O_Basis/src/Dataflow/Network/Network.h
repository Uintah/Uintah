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
  
class Network {
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

  Connection* connection(int);
  string connect(Module*, int, Module*, int);
  int disconnect(const string&);
  void disable_connection(const string& connId);
  Module* add_module(const string& packageName,
		     const string& categoryName,
		     const string& moduleName);
  // For SCIRun2
  Module* add_module2(const string& packageName,
		      const string& categoryName,
		      const string& moduleName);
  void add_instantiated_module(Module* mod);
  int delete_module(const string& name);


  Module* get_module_by_id(const string& id);

  void schedule();
  Scheduler* get_scheduler();
  void attach(Scheduler*);

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  ViewServer *server;
#endif
  // CollabVis code end
};

} // End namespace SCIRun


#endif
