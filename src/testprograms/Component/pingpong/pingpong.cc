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

#include <Core/CCA/Component/PIDL/MalformedURL.h>

#include <testprograms/Component/pingpong/PingPong_impl.h>
#include <testprograms/Component/pingpong/PingPong_sidl.h>
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

    using PingPong_ns::PingPong_impl;
    using PingPong_ns::PingPong;

    try {
      cout << "initialize:\n";
	PIDL::PIDL::initialize(argc, argv);
      cout << "done initialize:\n";

	bool client=false;
	bool server=false;
	string client_url;
	int reps=100;

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
	  cout << "objectFrom: " << client_url << "\n";
	    PIDL::Object obj=PIDL::PIDL::objectFrom(client_url);
	    cout << "got it\n";
	    PingPong pp=pidl_cast<PingPong>(obj);
	    cout << "done with pidl_cast\n";
	    if(!pp){
		cerr << "Wrong object type!\n";
		abort();
	    }
	    double stime=Time::currentSeconds();
	    cout << "time is: " << stime << "\n";
	    for(int i=0;i<reps;i++){
	      cerr << i << ": ping!\n";
		int j=pp->pingpong(i);
		if(i != j)
		    cerr << "BAD data: " << i << " vs. " << j << '\n';
	    }
	    double dt=Time::currentSeconds()-stime;
	    cerr << reps << " reps in " << dt << " seconds\n";
	    double us=dt/reps*1000*1000;
	    cerr << us << " us/rep\n";
	}
    } catch(const PIDL::MalformedURL& e) {
	cerr << "pingpong.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const Exception& e) {
	cerr << "pingpong.cc: Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
	abort();
    }
	cerr << "Serve Objects!\n";
	//int k; cin >> k;
	PIDL::PIDL::serveObjects();
	cerr << "Done Serve Objects!\n";
    return 0;
}

