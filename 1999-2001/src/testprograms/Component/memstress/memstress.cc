
/*
 *  memstress.cc
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <Component/PIDL/PIDL.h>
#include "memstress_sidl.h"
#include <SCICore/Thread/Time.h>
#include <vector>
using std::cerr;
using std::cout;
using std::vector;
using memstress::Server;

class Server_impl : public memstress::Server_interface {
public:
    Server_impl();
    virtual ~Server_impl();
    virtual void ping();
    virtual Server newObject();
    virtual Server returnReference();
};

Server_impl::Server_impl()
{
}

Server_impl::~Server_impl()
{
}

void Server_impl::ping()
{
}

Server Server_impl::newObject()
{
    return new Server_impl();
}

Server Server_impl::returnReference()
{
    return this;
}

void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  -server  - server process\n";
    cerr << "  -client URL  - client process\n";
    cerr << "  -test NAME - do test NAME\n";
    cerr << "  -reps N - do test N times\n";
    cerr << "\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    using std::string;
    using Component::PIDL::Object;
    using Component::PIDL::PIDLException;
    using Component::PIDL::PIDL;
    using SCICore::Thread::Time;

    try {
	PIDL::initialize(argc, argv);

	bool client=false;
	bool server=false;
	string client_url;
	string test="all";
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
	    } else if(arg == "-test"){
		if(++i >= argc)
		    usage(argv[0]);
		test=argv[i];
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

	Server pp;
	if(server) {
	    cerr << "Creating memstress object\n";
	    pp=new Server_impl;
	    cerr << "Waiting for memstress connections...\n";
	    cerr << pp->getURL().getString() << '\n';
	} else {
	    Object obj=PIDL::objectFrom(client_url);
	    Server s=pidl_cast<Server>(obj);

	    if(test == "ping" || test == "all"){
		double stime=Time::currentSeconds();
		for(int i=0;i<reps;i++)
		    s->ping();
		double dt=Time::currentSeconds()-stime;
		cerr << "ping: " << reps << " reps in " << dt << " seconds\n";
		double us=dt/reps*1000*1000;
		cerr << "ping: " << us << " us/rep\n";
	    }
	    if(test == "upcast" || test == "all"){
		double stime=Time::currentSeconds();
		for(int i=0;i<reps;i++){
		    Server s2=pidl_cast<Server>(obj);
		    s2->ping();
		}
		double dt=Time::currentSeconds()-stime;
		cerr << "upcast: " << reps << " reps in " << dt << " seconds\n";
		double us=dt/reps*1000*1000;
		cerr << "upcast: " << us << " us/rep\n";
	    }
	    /* This is not done for "all" since there is a leak in
	     *  nexus somewhere... -sparker
	     */
	    if(test == "urlcast"){
		double stime=Time::currentSeconds();
		for(int i=0;i<reps;i++){
		    Object obj2=PIDL::objectFrom(client_url);
		    Server s2=pidl_cast<Server>(obj2);
		    s2->ping();
		}
		double dt=Time::currentSeconds()-stime;
		cerr << "urlcast: " << reps << " reps in " << dt << " seconds\n";
		double us=dt/reps*1000*1000;
		cerr << "urlcast: " << us << " us/rep\n";
	    }
	    if(test == "refreturn" || test == "all"){
		double stime=Time::currentSeconds();
		for(int i=0;i<reps;i++){
		    Server s2=s->returnReference();
		    s2->ping();
		}
		double dt=Time::currentSeconds()-stime;
		cerr << "refreturn: " << reps << " reps in " << dt << " seconds\n";
		double us=dt/reps*1000*1000;
		cerr << "refreturn: " << us << " us/rep\n";
	    }
	    if(test == "newobject" || test == "all"){
		double stime=Time::currentSeconds();
		for(int i=0;i<reps;i++){
		    Server s2=s->newObject();
		    s2->ping();
		}
		double dt=Time::currentSeconds()-stime;
		cerr << "newobject: " << reps << " reps in " << dt << " seconds\n";
		double us=dt/reps*1000*1000;
		cerr << "newobject: " << us << " us/rep\n";
	    }
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
// Revision 1.1  1999/09/26 06:12:59  sparker
// Added (distributed) reference counting to PIDL objects.
// Began campaign against memory leaks.  There seem to be no more
//   per-message memory leaks.
// Added a test program to flush out memory leaks
// Fixed other Component testprograms so that they work with ref counting
// Added a getPointer method to PIDL handles
//
//
