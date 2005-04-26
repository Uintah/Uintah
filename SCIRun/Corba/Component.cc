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
 *  Component.cc: 
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
#include <SCIRun/Corba/UsesPort.h>
#include <SCIRun/Corba/ProvidesPort.h>
#include <SCIRun/Corba/Component.h>
#include <SCIRun/Corba/Services.h>

using namespace SCIRun;
using namespace corba;

Component::Component()
{
  have_ui=true;
  //  have_ui=false;
}

Component::~Component()
{
  //TODO
  std::cerr<<" *** ~Component *** called" << std::endl;
}

void
Component::enableUI(bool new_thread)
{
  have_ui=true;
  new_ui_thread=new_thread;
}

void 
Component::disableUI()
{
  have_ui=false;
}

bool 
Component::haveUI()
{
  return have_ui;
}

Port* 
Component::getPort(const std::string &name)
{
  for(unsigned int i=0; i<uports.size(); i++)
    {
    if(name==uports[i]->getName()) return uports[i];
    }
  for(unsigned int i=0; i<pports.size(); i++)
    {
    if(name==pports[i]->getName()) return pports[i];
    }
  return 0;
}

void
Component::addPort(Port *port)
{
  port->setComponent(this);
  if(port->isUses())
    {
    uports.push_back((UsesPort*)port);
    }
  else
    {
    pports.push_back((ProvidesPort*)port);
    }
}

void
Component::removePort(Port *port)
{
  if(port->isUses())
    {
    for(unsigned int i=0; i<uports.size(); i++)
      {
      if(uports[i]==port)
        {
        uports.erase(uports.begin()+i);
        return;
        }
      }
    }
  else
    {
    for(unsigned int i=0; i<pports.size(); i++)
      {
      if(pports[i]==port)
        {
        pports.erase(pports.begin()+i);
        return;
        }
      }
    }
}

void
Component::removePort(const std::string &name)
{
  removePort(getPort(name));
}

int 
Component::numUsesPorts()
{
  return uports.size();
}

int 
Component::numProvidesPorts()
{
  return pports.size();
}

bool
Component::isThreadedUI()
{
  return new_ui_thread;
}

UsesPort* 
Component::getUsesPort(unsigned int index)
{
  if(index>uports.size()) return 0;
  return uports[index];
}


UsesPort* 
Component::getUsesPort(const std::string &name)
{
  for(unsigned i=0; i<uports.size();i++){
    if(name==uports[i]->getName()) return uports[i];
  }
  return 0;
}


ProvidesPort* 
Component::getProvidesPort(unsigned int index)
{
  if(index>pports.size()) return 0;
  return pports[index];
}

ProvidesPort* 
Component::getProvidesPort(const std::string &name)
{
 for(unsigned i=0; i<pports.size();i++){
    if(name==pports[i]->getName()) return pports[i];
  }
  return 0;
}

int 
Component::popupUI()
{
  services->letGo();
  return 0;
}

void
Component::setServices(Services *services)
{
  this->services=services;
}
