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
 *  HelloWorldBridge.cc:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#include <CCA/Components/HelloWorldBridge/HelloWorldBridge.h>
#include <iostream>

using namespace std;
using namespace SCIRun;

extern "C" BridgeComponent* make_Bridge_HelloWorldBridge()
{
  return static_cast<BridgeComponent*>(new HelloWorldBridge());
}


HelloWorldBridge::HelloWorldBridge(){
}

HelloWorldBridge::~HelloWorldBridge(){
}

void HelloWorldBridge::setServices(const BridgeServices* svc){
  ::std::cerr << "Begin HWB::setSVcs\n";
  services=const_cast<BridgeServices*>(svc);
  StringPort::pointer* strport = new StringPort::pointer;
  (*strport) = StringPort::pointer(new StringPort(services));
  ::std::cerr << "MIdway HWB::setSVcs\n";
  services->addProvidesPort((void*)strport,"stringport","sci.cca.ports.StringPort",CCA);
  services->registerUsesPort("idport","gov.cca.ports.IDPort",Babel);
  ::std::cerr << "End HWB::setSVcs\n";
}

std::string StringPort::getString() {
  gov::cca::ports::IDPort s = mysvcs->getBabelPort("idport");
  if(s._is_nil()) return "NIL\n"; 
  return s.getID();
}

