
/*
 *  Network.h: Interface to Network class from project
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Network_h
#define SCI_project_Network_h 1

#include <Multitask/ITC.h>
#include <Classlib/Array1.h>
#include <Classlib/HashTable.h>
#include <Classlib/String.h>
class Connection;
class Module;
class NetworkEditor;

class Network {
    Mutex the_lock;
    int read_file(const clString&);
    Array1<Module*> modules;
    Array1<Connection*> connections;

    HashTable<clString, Module*> module_ids;
    HashTable<clString, Connection*> conn_ids;
    NetworkEditor* netedit;
    int first;
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

    int nconnections();
    Connection* connection(int);
    clString connect(Module*, int, Module*, int);
    
    Module* add_module(const clString& name);
    int delete_module(const clString& name);

    Module* get_module_by_id(const clString& id);
};

#endif
