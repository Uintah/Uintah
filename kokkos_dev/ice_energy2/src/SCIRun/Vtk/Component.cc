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
#include <SCIRun/Vtk/InPort.h>
#include <SCIRun/Vtk/OutPort.h>
#include <SCIRun/Vtk/Component.h>

using namespace SCIRun;
using namespace vtk;

Component::Component()
{
  have_ui=false;
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
  for(unsigned int i=0; i<iports.size(); i++)
    {
    if(name==iports[i]->getName()) return iports[i];
    }
  for(unsigned int i=0; i<oports.size(); i++)
    {
    if(name==oports[i]->getName()) return oports[i];
    }
  return 0;
}

void
Component::addPort(Port *port)
{
  port->setComponent(this);
  if(port->isInput())
    {
    iports.push_back((InPort*)port);
    }
  else
    {
    oports.push_back((OutPort*)port);
    }
}

void
Component::removePort(Port *port)
{
  if(port->isInput())
    {
    for(unsigned int i=0; i<iports.size(); i++)
      {
      if(iports[i]==port)
        {
        iports.erase(iports.begin()+i);
        return;
        }
      }
    }
  else
    {
    for(unsigned int i=0; i<oports.size(); i++)
      {
      if(oports[i]==port)
        {
        oports.erase(oports.begin()+i);
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
Component::numIPorts()
{
  return iports.size();
}

int 
Component::numOPorts()
{
  return oports.size();
}

bool
Component::isThreadedUI()
{
  return new_ui_thread;
}

InPort* 
Component::getIPort(unsigned int index)
{
  if(index>iports.size()) return 0;
  return iports[index];
}

OutPort* 
Component::getOPort(unsigned int index)
{
  if(index>oports.size()) return 0;
  return oports[index];
}

int 
Component::popupUI()
{
  return 0;
}

void 
Component::updateAllOutPorts(int flag){
  for(unsigned i=0; i<oports.size(); i++){
    oports[i]->update(flag);
  }
}
