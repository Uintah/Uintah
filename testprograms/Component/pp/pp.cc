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
 *  pp.cc
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
#include <string>
#include <Core/CCA/Component/PIDL/PIDL.h>

#include <Core/CCA/Component/Comm/SocketEpChannel.h>

#include <Core/CCA/Component/PIDL/MalformedURL.h>

#include <testprograms/Component/pp/PingPong_impl.h>

#include <Core/Thread/Time.h>

using namespace std;

using namespace SCIRun;
using namespace PingPong_ns;



void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  -server  - server process\n";
    cerr << "  -client  - client process\n";
    cerr << "  -stop  - stop the server\n";
    cerr << "  -test  - test the ploader\n";
    cerr << "\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    using std::string;

    SocketEpChannel *sep=new SocketEpChannel();
    cerr<<"Server URL="<<sep->getUrl()<<endl;
    try{
      PIDL::initialize(argc,argv);
      bool client=false;
      bool server=false;
      bool stop=false;
      bool test=false;
      string url;
      int reps=100;
      
      for(int i=1;i<argc;i++){
	string arg(argv[i]);
	if(arg == "server") server=true;
        else if(arg == "client") client=true;
        else if(arg == "stop")   stop=true;
        else if(arg == "test")   test=true;
        else url=arg;
      }
      if(!client && !server && !stop && !test)
	usage(argv[0]);
      
      if(server) {
	PingPong_impl::pointer pp=PingPong_impl::pointer(new PingPong_impl);
	pp->addReference();
	cerr << "Waiting for pp connections...\n";
	ofstream f("pp.url");
	std::string s;
	f<<pp->getURL().getString();
	f.close();
      } 

      else if(client){
	ifstream f("pp.url");
	std::string s;
	f>>s;
	f.close();
	Object::pointer obj=
	  PIDL::objectFrom(s);
	cerr << "Object_from completed\n";
	PingPong::pointer pp=pidl_cast<PingPong::pointer>(obj);
	cerr << "pidl_cast completed\n";
	if(pp.isNull()){
	  cerr << "pp_isnull\n";
	  abort();
	}

	int i=1999;
	cerr << "Calling pingpong....\n";
	int j=pp->pingpong(i);
	if(i!=j){
	  cerr << "BAD data: " << i << " vs. " << j << '\n';
	}
	else{
	  cerr << "Successful\n";
	}
	
      }
      else if(stop){
	ifstream f("pp.url");
	std::string s;
	f>>s;
	f.close();
	Object::pointer obj=PIDL::objectFrom(s);
	cerr << "Object_from completed\n";
	PingPong::pointer pp=pidl_cast<PingPong::pointer>(obj);
	cerr << "pidl_cast completed\n";
	if(pp.isNull()){
	  cerr << "pp_isnull\n";
	  abort();
	}
	cerr << "Calling pp.stop()....\n";
	pp->stop();
      }
      else if(test){
	cerr<<"url="<<url;
	Object::pointer obj=PIDL::objectFrom(url);
	cerr << "Object_from completed\n";
	//PingPong::pointer pp=pidl_cast<PingPong::pointer>(obj);
	//cerr << "pidl_cast completed\n";
	/*if(pp.isNull()){
	  cerr << "pp_isnull\n";
	  abort();
	  }*/
      }

      PIDL::serveObjects();
      cerr << "exits\n";
      PIDL::finalize();

    } catch(const MalformedURL& e) {
	cerr << "pp.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const InternalError &e) {
	cerr << "Caught unexpected exception!\n";
	cerr << e.message() << '\n';
	abort();
    } catch(const Exception& e) {
	cerr << "pp.cc: Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
	abort();
    }
    return 0;
}


