
/*
 *  pingpong.cc
 *  $Id$
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
#include <Component/PIDL/PIDL.h>
#include "PingPong_impl.h"
#include "PingPong_sidl.h"
#include <SCICore/Thread/Time.h>

using std::cerr;
using std::cout;

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
    using SCICore::Thread::Time;

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
    } catch(const SCICore::Exceptions::Exception& e) {
	cerr << "Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
	abort();
    }
    return 0;
}

//
// $Log$
// Revision 1.4  1999/09/26 06:13:00  sparker
// Added (distributed) reference counting to PIDL objects.
// Began campaign against memory leaks.  There seem to be no more
//   per-message memory leaks.
// Added a test program to flush out memory leaks
// Fixed other Component testprograms so that they work with ref counting
// Added a getPointer method to PIDL handles
//
// Revision 1.3  1999/09/21 06:08:14  sparker
// Removed extraneous using statement
//
// Revision 1.2  1999/09/17 05:08:45  sparker
// Updates to work with automatic sidl code generation
//
// Revision 1.1  1999/09/07 07:35:38  sparker
// Now builds some of the test programs, including a pingpong test
// program for the new component model
//
//
//
