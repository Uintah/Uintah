
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

#define NETEDIT_CANVAS_SIZE 2000

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
    conn->set_context(netedit);
    conn->connect();
    connections.add(conn);
    // Reschedule next time we can...
    reschedule=1;
}

void Network::connect(Connection* conn)
{
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
    add_module("ScalarFieldReader");
    add_module("IsoSurface");
    add_module("Salmon");
    connect(modules[0], 0, modules[1], 0);
    connect(modules[1], 0, modules[2], 0);
#if 0
    add_module("MeshView");
    add_module("Salmon");
    connect(modules[0], 0, modules[1], 0);
#endif
}

void Network::add_module(const clString& name, int xpos, int ypos)
{
    makeModule maker=ModuleList::lookup(name);
    if(!maker){
	cerr << "Module: " << name << " not found!\n";
	return;
    }
    Module* mod=(*maker)();
    if(xpos==-1 || ypos==-1){
	xpos=ypos=10;
	int found=1;
	while(found){
	    found=0;
	    for(int i=0;i<modules.size();i++){
		Module* mod=modules[i];
		if(mod->xpos==xpos
		   && mod->ypos==ypos){
		    ypos+=75;
		    if(ypos > NETEDIT_CANVAS_SIZE-50){
			xpos+=150;
			ypos=10;
		    }
		    found=1;
		    break;
		}
	    }
	}
    }
    modules.add(mod);
    mod->xpos=xpos;
    mod->ypos=ypos;
    mod->set_context(netedit, this);
}
