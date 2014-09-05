/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <unistd.h>
#include <string>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/SocketEpChannel.h>
#include <Core/CCA/PIDL/SocketSpChannel.h>
#include <Core/CCA/PIDL/Message.h>

#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/pp/PingPong_impl.h>

#include <Core/Thread/Time.h>

using namespace SCIRun;
using namespace PingPong_ns;



void usage(char* progname)
{
    std::cerr << "usage: " << progname << " [options]\n";
    std::cerr << "valid options are:\n";
    std::cerr << "  -server  - server process\n";
    std::cerr << "  -client  - client process\n";
    std::cerr << "  -stop  - stop the server\n";
    std::cerr << "  -test  - test the ploader\n";
    std::cerr << "\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    using std::string;
    try{
      PIDL::initialize();
      bool client=false;
      bool server=false;
      bool stop=false;
      bool test=false;
      string url;
      
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
	PingPong_impl::pointer pp(new PingPong_impl);
	Port_impl::pointer port(new Port_impl);
	port->addReference();
	pp->addReference();
	std::cerr << "Port is Waiting connections...\n";
	std::ofstream f("pp.url");
	std::string s;
	f<<pp->getURL().getString()<< std::endl;
	f<<port->getURL().getString()<< std::endl;
	f.close();
      } 

      else if(client){
	std::ifstream f("pp.url");
	std::string ppurl, porturl;
	f>>ppurl;
	f>>porturl;
	std::cerr<<"ppurl="<<ppurl<< std::endl;
	std::cerr<<"porturl="<<porturl<< std::endl;
	f.close();
	Object::pointer ppobj=
	  PIDL::objectFrom(ppurl);
	Object::pointer portobj=
	  PIDL::objectFrom(porturl);
	std::cerr << "Object_from completed\n";

	PingPong::pointer pp=pidl_cast<PingPong::pointer>(ppobj);
	Port::pointer port=pidl_cast<Port::pointer>(portobj);
	std::cerr << "pidl_cast completed\n";

	if(port.isNull() || pp.isNull() ){
	  std::cerr << "port or pp is null\n";
	  abort();
	}

	std::cerr << "Calling pingpong....\n";
	std::cerr<<pp->pingpong(port)<< std::endl;
      }
      else if(stop){
	std::ifstream f("pp.url");
	std::string ppurl, porturl;
	f>>ppurl;
	f>>porturl;
	std::cerr<<"ppurl="<<ppurl<< std::endl;
	std::cerr<<"porturl="<<porturl<< std::endl;
	f.close();
	Object::pointer ppobj=
	  PIDL::objectFrom(ppurl);
	Object::pointer portobj=
	 PIDL::objectFrom(porturl);
	std::cerr << "Object_from completed\n";
	
	PingPong::pointer pp=pidl_cast<PingPong::pointer>(ppobj);
	Port::pointer port=pidl_cast<Port::pointer>(portobj);
	std::cerr << "pidl_cast completed\n";
	
	if(port.isNull() || pp.isNull() ){
	  std::cerr << "port or pp is null\n";
	  abort();
	}
	
	std::cerr << "Calling stop() for both server objects....\n";
	pp->stop();
	port->stop();
	
      }
      else if(test){
	std::cerr<<"url="<<url;
	Object::pointer obj=PIDL::objectFrom(url);
	std::cerr << "Object_from completed\n";
      }

      PIDL::serveObjects();
      PIDL::finalize();
      std::cerr << "exits\n";
    } catch(const MalformedURL& e) {
	std::cerr << "pp.cc: Caught MalformedURL exception:\n";
	std::cerr << e.message() << '\n';
    } catch(const InternalError &e) {
	std::cerr << "Caught unexpected exception!\n";
	std::cerr << e.message() << '\n';
	abort();
    } catch(const Exception& e) {
	std::cerr << "pp.cc: Caught exception:\n";
	std::cerr << e.message() << '\n';
	abort();
    } catch(...) {
	std::cerr << "Caught unexpected exception!\n";
	abort();
    }
    return 0;
}


