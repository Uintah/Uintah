/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  objects.cc
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
#include <Core/CCA/PIDL/PIDL.h>
#include <testprograms/Component/objects/objects_sidl.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <vector>

using std::cerr;
using std::cout;
using std::vector;

using objects_test::Client;
using objects_test::RingMaster;

using namespace SCIRun;

class RingMaster_impl : public objects_test::RingMaster {
    vector<Client::pointer> clients;
public:
    RingMaster_impl();
    virtual ~RingMaster_impl();
    virtual int registerClient(const Client::pointer& c);
};

RingMaster_impl::RingMaster_impl()
{
}

RingMaster_impl::~RingMaster_impl()
{
}

int RingMaster_impl::registerClient(const Client::pointer& c)
{
   clients.push_back(c);
   for(vector<Client::pointer>::iterator iter=clients.begin();
	iter != clients.end(); iter++){
	(*iter)->notify(c);
   }
   cerr << "Done with notify client\n";
   return clients.size();
}

class Client_impl : public objects_test::Client {
    vector<Client::pointer> clients;
public:
    Client_impl();
    virtual ~Client_impl();
    void notify(const Client::pointer& newclient);
    int ping(int);
};

Client_impl::Client_impl()
{
}

Client_impl::~Client_impl()
{
}

void Client_impl::notify(const Client::pointer& a)
{
    clients.push_back(a);
    int c=1;
    for(vector<Client::pointer>::iterator iter=clients.begin();
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
    Thread::exitAll(1);
}

int main(int argc, char* argv[])
{
    using std::string;

    try {
      PIDL::initialize();

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

	RingMaster::pointer pp;
	if(server) {
	  cerr << "Creating objects object\n";
	  pp=RingMaster::pointer(new RingMaster_impl);
	  cerr << "Waiting for objects connections...\n";
	  cerr << pp->getURL().getString() << '\n';
	} else {
	  Object::pointer obj=PIDL::objectFrom(client_url);
	  RingMaster::pointer rm=pidl_cast<RingMaster::pointer>(obj);
	  Client_impl* me=new Client_impl;
	  int myid=rm->registerClient(Client::pointer(me));
	  cerr << "Test Successful: myid=" << myid << '\n';
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

