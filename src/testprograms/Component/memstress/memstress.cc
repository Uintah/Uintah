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
 *  memstress.cc
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
#include <Core/CCA/PIDL/PIDL.h>
#include <testprograms/Component/memstress/memstress_sidl.h>
#include <Core/Thread/Time.h>
#include <vector>
using std::cerr;
using std::cout;
using std::vector;
using memstress::Server;

using namespace SCIRun;

class Server_impl : public memstress::Server {
public:
    Server_impl();
    virtual ~Server_impl();
    virtual void ping();
    virtual Server::pointer newObject();
    virtual Server::pointer returnReference();
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

Server::pointer Server_impl::newObject()
{
  return Server::pointer(new Server_impl());
}

Server::pointer Server_impl::returnReference()
{
  return Server::pointer(this);
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

    try {
      PIDL::initialize(argc,argv);

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

	Server::pointer pp;
	if(server) {
	  pp=Server::pointer(new Server_impl);
	  cerr << "Waiting for memstress connections...\n";
	  cerr << pp->getURL().getString() << '\n';
	} else {
	    Object::pointer obj=PIDL::objectFrom(client_url);
	    Server::pointer s(pidl_cast<Server::pointer>(obj));

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
		  Server::pointer s2(pidl_cast<Server::pointer>(obj));
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
		  Object::pointer obj2=PIDL::objectFrom(client_url);
		  Server::pointer s2=pidl_cast<Server::pointer>(obj2);
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
		    Server::pointer s2=s->returnReference();
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
		    Server::pointer s2=s->newObject();
		    s2->ping();
		}
		double dt=Time::currentSeconds()-stime;
		cerr << "newobject: " << reps << " reps in " << dt << " seconds\n";
		double us=dt/reps*1000*1000;
		cerr << "newobject: " << us << " us/rep\n";
	    }
	}
	PIDL::serveObjects();
        PIDL::finalize();
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

