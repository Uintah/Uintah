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
#include <Core/CCA/PIDL/PIDL.h>

#include <Core/CCA/PIDL/MalformedURL.h>

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
      PIDL::initialize(argc,argv);

	bool client=false;
	bool server=false;
	string client_url;
	int reps=10;

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
	    PingPong_impl::pointer pp(new PingPong_impl);
	    pp->addReference();
	    cerr << "Waiting for pingpong connections...\n";
	    cerr << pp->getURL().getString() << '\n';
	} else {
	  Object::pointer obj=PIDL::objectFrom(client_url);
	  cerr << "Object_from completed\n";

	  PingPong::pointer pp=pidl_cast<PingPong::pointer>(obj);
	  cerr << "pidl_case completed\n";
	  if(pp.isNull()){
	    cerr << "pp_isnull\n";
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
    } catch(const MalformedURL& e) {
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
    PIDL::serveObjects();
    PIDL::finalize();
    return 0;
}

