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

#include <CCA/Components/Hello/Hello.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>
#include <Core/Thread/Time.h>

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Hello()
{
  return sci::cca::Component::pointer(new Hello());
}


Hello::Hello(){
}

Hello::~Hello(){
}

void Hello::setServices(const sci::cca::Services::pointer& svc){
  services=svc;
  cerr<<"svc->createTypeMap...";
  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  cerr<<"Done\n";

  myUIPort::pointer uip=myUIPort::pointer(new myUIPort);
  myGoPort::pointer gop=myGoPort::pointer(new myGoPort(svc));

  cerr<<"svc->addProvidesPort(uip)...";  
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort",  sci::cca::TypeMap::pointer(0));
  cerr<<"Done\n";

  cerr<<"svc->addProvidesPort(gop)...";  
  svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort",  sci::cca::TypeMap::pointer(0));
  cerr<<"Done\n";

  svc->registerUsesPort("stringport","sci.cca.ports.StringPort", sci::cca::TypeMap::pointer(0));
}

int myUIPort::ui(){
  cout<<"UI button is clicked!\n";
  return 0;
}

myGoPort::myGoPort(const sci::cca::Services::pointer& svc){
  this->services=svc;
}

int myGoPort::go(){
  if(services.isNull()){
    cerr<<"Null services!\n";
    return 1;
  }
  cerr<<"Hello.go.getPort...";
  double st = SCIRun::Time::currentSeconds();

  sci::cca::Port::pointer pp=services->getPort("stringport");	
  //cerr<<"Done\n";
  //if(pp.isNull()){
  //  cerr<<"stringport is not available!\n";
  //  return 1;
  //}  
  //else{
  //  cerr<<"stringport is not null\n";
  //}
  //cerr<<"Hello.go.pidl_cast...";
  sci::cca::ports::StringPort::pointer sp=pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
  //cerr<<"Done\n";
  std::string name=sp->getString();

  double t=Time::currentSeconds()-st;
  ::std::cerr << "Done in " << t << "secs\n";
  ::std::cerr << t*1000*1000 << " us/rep\n";


  services->releasePort("stringport");

  cout<<"Hello "<<name<<endl;
  return 0;
}
 
