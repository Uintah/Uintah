
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
class clString;
class Module;

class Network {
    Mutex the_lock;
    int read_file(const clString&);
    Array1<Module*> modules;
public:
    Network(int first);
    ~Network();

    void read_lock();
    void read_unlock();
    void write_lock();
    void write_unlock();

    int nmodules();
    Module* module(int);
};

#endif
