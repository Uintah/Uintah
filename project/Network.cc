
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
#include <Module.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <Classlib/Args.h>
#include <iostream.h>
#include <stdlib.h>

static Arg_stringval initial_net("net", "", "specify initial network to load");

Network::Network(int first)
{
    if(first && initial_net.is_set()){
	if(!read_file(initial_net.value())){
	    cerr << "Can't read initial map\n";
	    exit(-1);
	}
    }
    modules.add((*ModuleList::lookup("WidgetReal"))());
    modules.add((*ModuleList::lookup("WidgetReal"))());
    modules[0]->activate(0);
    modules[1]->activate(0);
}

Network::~Network()
{
}

int Network::read_file(const clString& filename)
{
    NOT_FINISHED("Network::read_file");
    return 1;
}

void Network::lock()
{
    the_lock.lock();
}

void Network::unlock()
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

