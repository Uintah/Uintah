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
 *  Hello.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#include <CCA/Core/Hello/Hello.h>
#include <Core/Thread/Time.h>
#include <SCIRun/Core/TypeMapImpl.h>

#include <iostream>


#include <unistd.h>

using namespace SCIRun;
using namespace sci::cca;

extern "C" Component::pointer make_SCIRun_Hello()
{
    return Component::pointer(new Hello());
}


Hello::Hello()
  : text("GO hasn't been called yet!")
{
}

Hello::~Hello()
{
  services->unregisterUsesPort("message");
}

void Hello::setServices(const Services::pointer& svc)
{
  services = svc;
  std::cerr << "svc->createTypeMap...";
  TypeMap::pointer properties = svc->createTypeMap();
  std::cerr << "Done\n";
  
  HelloGoPort::pointer goPort( new HelloGoPort(this) );
  
  std::cerr << "svc->addProvidesPort(gop)...";  
  svc->addProvidesPort(goPort, "go", "sci.cca.ports.GoPort", TypeMap::pointer(0));
  std::cerr << "Done\n";
  
  properties->putString("cca.portName", "message");
  properties->putString("cca.portType", "sci.cca.ports.StringPort");

  svc->registerUsesPort("message","sci.cca.ports.MessagePort", properties);
  
}

int Hello::go()
{
  if (services.isNull()) {
    std::cerr << "Null services!\n";
    return 1;
  }
  std::cerr << "Hello.go.getPort...";
  double st = SCIRun::Time::currentSeconds();
  
  Port::pointer port = services->getPort("message");	
  ports::MessagePort::pointer message =  pidl_cast<ports::MessagePort::pointer>(port);
  std::string name = message->getString();
  
  double t = Time::currentSeconds() - st;
  std::cerr << "Done in " << t << "secs\n";
  std::cerr << t*1000*1000 << " us/rep\n";
  
  if (! name.empty()) text = name;
  
  services->releasePort("message");
  
  return 0;
}
