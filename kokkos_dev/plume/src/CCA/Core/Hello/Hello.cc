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
using namespace sci::cca::ports;

extern "C" Component::pointer make_cca_core_Hello()
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
  
  services->addProvidesPort( new HelloGoPort(this), "go", "sci.cca.ports.GoPort", TypeMap::pointer(0));

  TypeMap::pointer properties = services->createTypeMap();

  properties->putString("cca.portName", "hello");
  properties->putString("cca.portType", "sci.cca.ports.StringPort");

  services->registerUsesPort("hello","sci.cca.ports.StringPort", properties);
  
}

int Hello::go()
{
  if (services.isNull()) {
    std::cerr << "services no set. go request ignored\n";
    return 1;
  }

  std::cerr << "Hello ";

  double start = SCIRun::Time::currentSeconds();

  Port::pointer port = services->getPort("hello");	
  StringPort::pointer message =  pidl_cast<StringPort::pointer>(port);
  std::string answer = message->getString();

  double sec = Time::currentSeconds() - start;

  std::cerr << answer <<"\n\n";
  std::cerr << "Done in " << sec << "secs\n";
  
  services->releasePort("hello");
  
  return 0;
}
