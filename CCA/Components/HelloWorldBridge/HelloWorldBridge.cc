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

#include "HelloWorldBridge.h"

using namespace somethingspc;
using namespace SCIRun;
using namespace NewPort;
using namespace SCIRun;
using namespace std;

extern "C" BridgeComponent* make_Bridge_HelloWorldBridge()
{
  return static_cast<BridgeComponent*>(new Bridge221());
}

static BridgeServices* st_svcs;

char* skel_NewPort_StringPort_getString( struct NewPort_StringPort__object* self )
{
  std::string _result;
  somethingspc::StringPort* _this = reinterpret_cast< somethingspc::StringPort*>(self->d_data);
  _result = _this->getString();
  return sidl_String_strdup(_result.c_str());
}

void skel_NewPort_StringPort__ctor(struct NewPort_StringPort__object* self ) {
  self->d_data = reinterpret_cast< void*>(new somethingspc::StringPort());
}

void skel_NewPort_StringPort__dtor(struct NewPort_StringPort__object* self ) {
  delete ( reinterpret_cast< somethingspc::StringPort*>(self->d_data) );
}

Bridge221::Bridge221(){
}

Bridge221::~Bridge221(){
}

void Bridge221::setServices(const BridgeServices* svc) {
  services=const_cast<BridgeServices*>(svc);
st_svcs = const_cast<BridgeServices*>(svc);
  NewPort::StringPort dp = NewPort::StringPort::_create();
  dp._get_ior()->d_epv->f_getString = skel_NewPort_StringPort_getString;
  dp._get_ior()->d_epv->f__ctor = (skel_NewPort_StringPort__ctor);
  dp._get_ior()->d_epv->f__dtor = (skel_NewPort_StringPort__dtor);
  services->addProvidesPort((void*)&dp,"pport","gov.cca.ports.StringPort",Babel);
  services->registerUsesPort("uport","sci.cca.ports.StringPort",CCA);

}

somethingspc::StringPort::StringPort(){
}

somethingspc::StringPort::~StringPort(){
}

string somethingspc::StringPort::getString() {
  sci::cca::Port::pointer pp = st_svcs->getCCAPort("uport");
  sci::cca::ports::StringPort::pointer sp=pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
  return sp->getString();

}

