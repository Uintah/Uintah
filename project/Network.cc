
/*
 *  Network.cc: The core of dataflow...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Network.h>
#include <Connection.h>
#include <Module.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <Classlib/Args.h>
#include <iostream.h>
#include <stdlib.h>

static Arg_stringval initial_net("net", "", "specify initial network to load");

Network::Network(int first)
: first(first), netedit(0)
{
}

Network::~Network()
{
}

int Network::read_file(const clString&)
{
    NOT_FINISHED("Network::read_file");
    return 1;
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

void Network::connect(Module* m1, int p1, Module* m2, int p2)
{
    Connection* conn=new Connection(m1, p1, m2, p2);
    connections.add(conn);
    // Reschedule next time we can...
    reschedule=1;
}

void Network::initialize(NetworkEditor* _netedit)
{
    netedit=_netedit;
    if(first && initial_net.is_set()){
	if(!read_file(initial_net.value())){
	    cerr << "Can't read initial map\n";
	    exit(-1);
	}
    }
#if 0
    modules.add((*ModuleList::lookup("SoundReader"))());
    modules[0]->set_context(netedit, this);
    modules.add((*ModuleList::lookup("SoundFilter"))());
    modules[1]->ypos=110;
    modules[1]->set_context(netedit, this);
    modules.add((*ModuleList::lookup("SoundOutput"))());
    modules[2]->ypos=210;
    modules[2]->set_context(netedit, this);
    connect(modules[0], 0, modules[1], 0);
    connect(modules[1], 0, modules[2], 0);
#endif
    modules.add((*ModuleList::lookup("IsoSurface"))());
    modules[0]->set_context(netedit, this);
    modules.add((*ModuleList::lookup("Salmon"))());
    modules[1]->ypos=110;
    modules[1]->set_context(netedit, this);
    connect(modules[0], 0, modules[1], 0);
}
