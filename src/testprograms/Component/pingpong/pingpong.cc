
/*
 *  pingpong.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include "PingPong_impl.h"
#include "PingPong_sidl.h"
#include <Core/Thread/Time.h>

using std::cerr;
using std::cout;

using namespace SCIRun;

void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  -server  - server process\n";
    cerr << "  -client URL  - client process\n";
    cerr << "\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    using std::string;
    using Component::PIDL::Object;
    using Component::PIDL::PIDLException;
    using Component::PIDL::PIDL;
    using PingPong::PingPong_impl;
    using PingPong::PingPong;

    try {
	PIDL::initialize(argc, argv);

	bool client=false;
	bool server=false;
	string client_url;
	int reps=1000;

	for(int i=1;i<argc;i++){
	    string arg(argv[i]);
	    if(arg == "-server"){
		if(client)
		    usage(argv[0]);
		server=true;
	    } else if(arg == "-client"){
		if(server)
		    usage(argv[0]);
		if(++i>=argc)
		    usage(argv[0]);
		client_url=argv[i];
		client=true;
	    } else if(arg == "-reps"){
		if(++i>=argc)
		    usage(argv[0]);
		reps=atoi(argv[i]);
	    } else {
		usage(argv[0]);
	    }
	}
	if(!client && !server)
	    usage(argv[0]);

	if(server) {
	    cerr << "Creating PingPong object\n";
	    PingPong_impl* pp=new PingPong_impl;
	    cerr << "Waiting for pingpong connections...\n";
	    cerr << pp->getURL().getString() << '\n';
	} else {
	    Object obj=PIDL::objectFrom(client_url);
	    PingPong pp=pidl_cast<PingPong>(obj);
	    if(!pp){
		cerr << "Wrong object type!\n";
		abort();
	    }
	    double stime=Time::currentSeconds();
	    for(int i=0;i<reps;i++){
		int j=pp->pingpong(i);
		if(i != j)
		    cerr << "BAD data: " << i << " vs. " << j << '\n';
	    }
	    double dt=Time::currentSeconds()-stime;
	    cerr << reps << " reps in " << dt << " seconds\n";
	    double us=dt/reps*1000*1000;
	    cerr << us << " us/rep\n";
	}
	PIDL::serveObjects();
    } catch(const Exception& e) {
	cerr << "Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
	abort();
    }
    return 0;
}

