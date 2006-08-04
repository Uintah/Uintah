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
 *  Services.cc: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */


#include <string>
#include <iostream>
#include <SCIRun/Corba/Services.h>
#include <SCIRun/Corba/Component.h>

using namespace SCIRun;
using namespace corba;

Services::Services(Component* com)
{
  this->com=com;
  wait=true;
  ready2go=false;
  //TODO
}

Services::~Services()
{
  //TODO
}


int
Services::registerUsesPort(const ::std::string& portName, const ::std::string& portType){
  //TODO:

  UsesPort *port=new UsesPort();
  port->setName(portName);
  port->setType(portType);
  com->addPort(port);
  return 0;
}


int
Services:: unregisterUsesPort(const ::std::string& portName){
  //TODO
  return 0;
}

int 
Services::addProvidesPort(const ::std::string& portName, const ::std::string& portType, const ::std::string& ior){
  //TODO
  ProvidesPort *port=new ProvidesPort();
  port->setName(portName);
  port->setType(portType);
  port->setIOR(ior);
  com->addPort(port);
  return 0;
}
  
int
Services::removeProvidesPort(const ::std::string& portName){
  //TODO
  return 0;
}

::std::string 
Services::getIOR(const ::std::string& portName){
  UsesPort* up=com->getUsesPort(portName);
  ProvidesPort *pp=up->getConnectedProvidesPort();
  return pp->getIOR();
}

int
Services::check(){
  //TODO: unsafe
  while(wait);
  wait=true;
  return 0;
}

int 
Services::done(){
  //TODO: unsafe 
  wait=false;
  return 0;
}

bool 
Services::go(){
  if(ready2go){
    ready2go=false;
    return true;
  }else{
    return false;
  }
}


void
Services::letGo(){
  ready2go=true;
}

sci::cca::CorbaServices::pointer
Services::getCorbaServices(const std::string & url){
  Object::pointer obj=PIDL::objectFrom(url);
  sci::cca::CorbaServices::pointer ptr=  pidl_cast<sci::cca::CorbaServices::pointer>(obj);
  return ptr;
}


