//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : spectest.cc
//    Author : Martin Cole
//    Date   : Tue Aug 21 11:17:14 2001

#include <Core/CCA/PIDL/PIDL.h>
#include <testprograms/Component/spectest/spectest_sidl.h>
#include <testprograms/Component/spectest/spectest_impl.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>

#include <iostream>
#include <vector>
#include <sstream>
#include <unistd.h>

using namespace std;
using namespace SCIRun;
using namespace spectest;
// FIX_ME MC
/* This example framework should create a couple example components and 
   make them available using the port interfaces.  It should also listen 
   and accept components that wish to register with this framework.
*/


//! Server should start up and listen for components that want to provide
//! a service, or clients that want to use a service. Server is simply the 
//! the middle man that hooks up services, perhaps it should get a finders fee.
#if 0
void server_mode() 
{
  cerr << "Server starting..." << endl;;
  
  ConnectionEventService 
}

static void fail(char* why)
{
  cerr << "Failure: " << why << endl;
  Thread::exitAll(1);
}
#endif

static void usage(char* progname)
{
  cerr << "usage: " << progname << " [options]" << endl;
  cerr << "valid options are:" << endl;
  cerr << "  -server  - server process" << endl;
  cerr << "  -client URL  - client process" << endl;
  cerr << "  -reps N - do test N times" << endl;
  cerr << "" << endl;
  Thread::exitAll(1);
}

int main(int argc, char* argv[])
{
  using std::string;

  //  usleep( 900000 );

  bool client=false;
  bool server=false;
  bool consumer = false;
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
    } else if(arg == "-consumer"){
      consumer = true;
    } else {
      usage(argv[0]);
    }
  }
  if(!client && !server)
    usage(argv[0]);
  Framework::pointer fw;
  try {
    PIDL::initialize(argc,argv);
    sleep( 1 ); // Give threads enough time to come up.

    if(server) {
      cerr << "Creating spectest server object\n";
      fw = new Framework_impl;
      cerr << "Waiting for spectest connections...\n";
      cerr << fw->getURL().getString() << '\n';
    } else {
      Object::pointer obj = PIDL::objectFrom(client_url);
      cerr << "in the middle\n";
      Framework::pointer fw = pidl_cast<Framework::pointer>(obj);
      CCA::Services::pointer s = fw->get_services();

      if (consumer) {
	ConsumerInt cint;
	cint.setServices(s);
	cint.go();
      } else {
	RandomInt rint;
	rint.setServices(s);
	rint.go();
      }
    }
    //    CurseComponent cc;
    //Services my_services;
    //cc.setServices(my_services);
    //cc.go();

    cerr << "Spectest successful" << endl;
  } catch(const Exception& e) {
    cerr << "spectest caught exception:" << endl;
    cerr << e.message() << endl;
    cerr << "main caught thread = " << Thread::self() << endl;
    //Thread::exitAll(1);
  } catch(...) {
    cerr << "Caught unexpected exception!" << endl;
    //Thread::exitAll(1);
  }
  PIDL::serveObjects();
  cerr << "done" << endl;
  return 0;
}

