
/*
 *  Network.h: Interface to Network class from project
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

#ifndef SCI_project_Network_h
#define SCI_project_Network_h 1

#include <PSECore/share/share.h>

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Thread/Mutex.h>

#include <map.h>

namespace PSECore {
namespace Dataflow {

using SCICore::Containers::clString;
using SCICore::Containers::Array1;

class Connection;

class Module;
class NetworkEditor;

class PSECORESHARE Network {
public:
    
    typedef map<clString, Connection*>	MapClStringConnection;
    typedef map<clString, Module*>	MapClStringModule;
    typedef map<int, Connection*>	MapIntConnection;
    typedef map<int, Module*>		MapIntModule;
    
private:
    SCICore::Thread::Mutex the_lock;
    int read_file(const clString&);

    MapClStringConnection conn_ids;
    
    NetworkEditor* netedit;
    int first;
    int nextHandle;
public:				// mm-hack to get direct access
    Array1<Connection*> connections;
    Array1<Module*> modules;
    
    MapClStringModule module_ids;
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
    clString connect(Module*, int, Module*, int);
    int disconnect(const clString&);
    Connection* get_connect_by_handle (int handle); 	// mm
    
    Module* add_module(const clString& packageName,
                       const clString& categoryName,
                       const clString& moduleName);
    int delete_module(const clString& name);

    Module* get_module_by_id(const clString& id); 	
    Module* get_module_by_handle (int handle);		// mm

};

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.6  2000/03/17 18:47:28  dahart
// Included STL map header files where I forgot them, and removed less<>
// parameter from map declarations
//
// Revision 1.5  2000/03/11 00:40:55  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.4  1999/08/28 17:54:29  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/26 23:59:07  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.2  1999/08/17 06:38:23  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:58  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 22:02:44  dav
// added back .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
