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
 *  BabelPortInstance.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#include <SCIRun/Babel/BabelPortInstance.h>
#include <iostream>

namespace SCIRun {

BabelPortInstance::BabelPortInstance(const std::string& name,
                                     const std::string& type,
                                     const gov::cca::TypeMap& properties,
                                     PortType porttype)
  : porttype(porttype), name(name), type(type), properties(properties),
    useCount(0)
{
}

BabelPortInstance::BabelPortInstance(const std::string& name,
                                     const std::string& type,
                                     const gov::cca::TypeMap& properties,
                                     const gov::cca::Port& port,
                                     PortType porttype)
  : porttype(porttype), name(name), type(type), properties(properties),
    port(port), useCount(0)
{
}

BabelPortInstance::~BabelPortInstance()
{
}

bool BabelPortInstance::connect(PortInstance* to)
{
  if(!canConnectTo(to))
    {
    return false;
    }
  //BabelPortInstance* p2 = dynamic_cast<BabelPortInstance*>(to);
  PortInstance* p2 = to;
  if(!p2)
    {
    return false;
    }
  if(portType() == From && p2->portType() == To)
    {    connections.push_back(p2);    }
  else
    {    p2->connect(this);    }
  return true;
}

PortInstance::PortType BabelPortInstance::portType()
{
  if(porttype == Uses)
    {    return From;    }
  else
    {    return To;      }
}

std::string BabelPortInstance::getType()
{
  return type;
}

std::string BabelPortInstance::getModel()
{
  return "babel"; 
}

std::string BabelPortInstance::getUniqueName()
{
  // Babel names are already guaranteed to be unique
  return name;
}

bool BabelPortInstance::disconnect(PortInstance* to)
{
  BabelPortInstance* p2 = dynamic_cast<BabelPortInstance*>(to);
  if(!p2) {    return false; }

  if(porttype !=Uses)
    {
    std::cerr<<"disconnect can be called only by user"<<std::endl; 
    return false;
    } 
  std::vector<PortInstance*>::iterator iter;
  for(iter=connections.begin(); iter<connections.end();iter++)
    {
    if(p2==(*iter))
      {
      connections.erase(iter);
      return true;
      }
    }
  return false;
}

bool BabelPortInstance::canConnectTo(PortInstance* to)
{
  //BabelPortInstance* p2 = dynamic_cast<BabelPortInstance*>(to);
  PortInstance* p2 = to;
  //  std::cerr<<"try to connect:"<<std::endl;
  //std::cerr<<"type(p1)="<<type<<std::endl;
  //std::cerr<<"type(p2)="<<p2->type<<std::endl;
  //std::cerr<<"port type(p1)="<<porttype<<std::endl;
  //std::cerr<<"port type(p2)="<<p2->porttype<<std::endl;
  
  if( p2 && getType()==p2->getType() && portType()!=p2->portType() )
    {
    if(available() && p2->available()) return true;
    }
  return false;
}

bool BabelPortInstance::available()
{
  return portType()==To || connections.size()==0;
}

PortInstance* BabelPortInstance::getPeer()
{
  return connections[0];
}

std::string BabelPortInstance::getName()
{
  return name;
}

void BabelPortInstance::incrementUseCount()
{
  useCount++;
}

bool BabelPortInstance::decrementUseCount()
{
  if(useCount<=0)
    {    return false; }
  
  useCount--;
  return true;
}

} // end namespace SCIRun
