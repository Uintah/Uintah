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
class NetworkEditor;

class PSECORESHARE Network {
public:
    
    typedef map<string, Connection*>	MapStringConnection;
    typedef map<string, Module*>	MapStringModule;
    typedef map<int, Connection*>	MapIntConnection;
    typedef map<int, Module*>		MapIntModule;
    
private:
    Mutex the_lock;
    int read_file(const string&);

    MapStringConnection conn_ids;
    
    NetworkEditor* netedit;
    int first;
    int nextHandle;
public:				// mm-hack to get direct access
    vector<Connection*> connections;
    vector<Module*> modules;
    
    MapStringModule module_ids;
    MapIntModule mod_handles;
    MapIntConnection conn_handles;
    
    int slave_socket;
    int reschedule;
public:
    Network(int first);
    ~Network();

    void initialize(NetworkEditor*);

    void read_lock();
    void read_unlock();
    void write_lock();
    void write_unlock();

    int nmodules();
    Module* module(int);

    int getNextHandle()  { return ++nextHandle; }  // mm

    int nconnections();
    Connection* connection(int);
    string connect(Module*, int, Module*, int);
    int disconnect(const string&);
    Connection* get_connect_by_handle (int handle); 	// mm
    
    Module* add_module(const string& packageName,
                       const string& categoryName,
                       const string& moduleName);
    int delete_module(const string& name);

    Module* get_module_by_id(const string& id); 	
    Module* get_module_by_handle (int handle);		// mm

};

} // End namespace SCIRun


#endif
