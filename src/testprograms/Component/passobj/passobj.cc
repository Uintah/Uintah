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
 *  passobj.cc
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May, 2003
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <vector>
#include <unistd.h>
#include <string>
#include <Core/CCA/PIDL/PIDL.h>

#include <Core/CCA/Comm/SocketEpChannel.h>
#include <Core/CCA/Comm/SocketSpChannel.h>
#include <Core/CCA/Comm/Message.h>

#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/passobj/passobj_impl.h>

#include <Core/Thread/Time.h>

using namespace std;

using namespace SCIRun;
using namespace passobj_ns;



void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  pass  - server process for pass object\n";
    cerr << "  port  - server process for port object\n";
    cerr << "  client  - client process\n";
    cerr << "\n";

    exit(1);
}

int main(int argc, char* argv[])
{
    using std::string;
    try{
      PIDL::initialize(argc,argv);
      bool client=false;
      bool server_pass=false;
      bool server_port=false;
      string url;
      int reps=1;
      
      for(int i=1;i<argc;i++){
	string arg(argv[i]);
	if(arg == "pass") server_pass=true;
	else if(arg == "port") server_port=true;
        else if(arg == "client") client=true;
      }
      if(!client && !server_pass && !server_port)
	usage(argv[0]);
      
      if(server_pass) {
	Pass_impl::pointer pass(new Pass_impl);
	pass->addReference();
	cerr << "Pass is Waiting for connections...\n";
	ofstream f("pass.url");
	std::string s;
	f<<pass->getURL().getString()<<endl;
	f.close();
      } 
      else if(server_port) {
	Port_impl::pointer port(new Port_impl);
	port->addReference();
	cerr << "Port is Waiting for connections...\n";
	ofstream f("port.url");
	std::string s;
	f<<port->getURL().getString()<<endl;
	f.close();
      } 
      else if(client){
	std::string passurl, porturl;

	ifstream f1("pass.url");
	f1>>passurl;
	cerr<<"passurl="<<passurl<<endl;
	f1.close();

	ifstream f2("port.url");
	f2>>porturl;
	cerr<<"porturl="<<porturl<<endl;
	f2.close();

	Object::pointer passObj=PIDL::objectFrom(passurl);
	Object::pointer portObj=PIDL::objectFrom(porturl);

	Pass::pointer pass=pidl_cast<Pass::pointer>(passObj);
	Port::pointer port=pidl_cast<Port::pointer>(portObj);

	if(passObj.isNull() || portObj.isNull() ){
	  cerr << "pass or port is null\n";
	  abort();
	}
	
	cerr << "Calling pass(port)...\n"<<endl;
	cerr<<pass->pass(port)<<endl;
	cerr << "Calling pass(port)...Done\n"<<endl;
      }

      PIDL::serveObjects();
      cerr << "exits\n";
      PIDL::finalize();

    } catch(const MalformedURL& e) {
	cerr << "passobj.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const InternalError &e) {
	cerr << "Caught unexpected exception!\n";
	cerr << e.message() << '\n';
	abort();
    } catch(const Exception& e) {
	cerr << "passobj.cc: Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
	abort();
    }
    return 0;
}


