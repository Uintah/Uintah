
/*
 *  objects.cc
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
#include "objects_sidl.h"
#include <SCICore/Thread/Time.h>
#include <vector>
using std::cerr;
using std::cout;
using std::vector;
using objects_test::Client;
using objects_test::RingMaster;

class RingMaster_impl : public objects_test::RingMaster_interface {
    vector<Client> clients;
public:
    RingMaster_impl();
    virtual ~RingMaster_impl();
    virtual int registerClient(const Client& c);
};

RingMaster_impl::RingMaster_impl()
{
}

RingMaster_impl::~RingMaster_impl()
{
}

int RingMaster_impl::registerClient(const Client& c)
{
    clients.push_back(c);
    for(vector<Client>::iterator iter=clients.begin();
	iter != clients.end(); iter++){
	(*iter)->notify(c);
    }
    cerr << "Done with notify client\n";
    return clients.size();
}

class Client_impl : public objects_test::Client_interface {
    vector<Client> clients;
public:
    Client_impl();
    virtual ~Client_impl();
    void notify(const Client& newclient);
    int ping(int);
};

Client_impl::Client_impl()
{
}

Client_impl::~Client_impl()
{
}

void Client_impl::notify(const Client& a)
{
    clients.push_back(a);
    int c=1;
    for(vector<Client>::iterator iter=clients.begin();
	iter != clients.end(); iter++){
	if((*iter)->ping(c) != c){
	    cerr << "Wrong result!\n";
	    exit(1);
	}
    }
    cerr << "Pinged " << clients.size() << " clients\n";
}

int Client_impl::ping(int p)
{
    return p;
}

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
    using SCICore::Thread::Time;

    try {
	PIDL::initialize(argc, argv);

	bool client=false;
	bool server=false;
	string client_url;

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
	    } else {
		usage(argv[0]);
	    }
	}
	if(!client && !server)
	    usage(argv[0]);

	if(server) {
	    cerr << "Creating objects object\n";
	    RingMaster_impl* pp=new RingMaster_impl;
	    cerr << "Waiting for objects connections...\n";
	    cerr << pp->getURL().getString() << '\n';
	} else {
	    Object obj=PIDL::objectFrom(client_url);
	    RingMaster rm=pidl_cast<RingMaster>(obj);

	    Client_impl* me=new Client_impl;
	    int myid=rm->registerClient(me);
	    cerr << "nclients now " << myid << '\n';
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
// Revision 1.2  1999/09/26 06:13:00  sparker
// Added (distributed) reference counting to PIDL objects.
// Began campaign against memory leaks.  There seem to be no more
//   per-message memory leaks.
// Added a test program to flush out memory leaks
// Fixed other Component testprograms so that they work with ref counting
// Added a getPointer method to PIDL handles
//
// Revision 1.1  1999/09/24 06:26:28  sparker
// Further implementation of new Component model and IDL parser, including:
//  - fixed bugs in multiple inheritance
//  - added test for multiple inheritance
//  - fixed bugs in object reference send/receive
//  - added test for sending objects
//  - beginnings of support for separate compilation of sidl files
//  - beginnings of CIA spec implementation
//  - beginnings of cocoon docs in PIDL
//  - cleaned up initalization sequence of server objects
//  - use globus_nexus_startpoint_eventually_destroy (contained in
// 	the globus-1.1-utah.patch)
//
//
