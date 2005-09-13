/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
      PIDL::initialize();
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


