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
 *  HelloWorldBridge.h
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#ifndef SCIRun_CCA_Components_World_h
#define SCIRun_CCA_Components_World_h

#include <Core/CCA/spec/cca_sidl.h>
//#include <CCA/Components/Hello/Hello_sidl.h>

#include <CCA/Components/BabelTest/who/gov_cca_ports_IDPort.hh>

#include <SCIRun/Bridge/BridgeComponent.h>
#include <SCIRun/Bridge/BridgeServices.h>


namespace SCIRun {

  class HelloWorldBridge : public BridgeComponent{
  public:
    HelloWorldBridge();
    virtual ~HelloWorldBridge();
    virtual void setServices(const BridgeServices* svc);
  private:
    HelloWorldBridge(const HelloWorldBridge&);
    HelloWorldBridge& operator=(const HelloWorldBridge&);
    BridgeServices* services;
  };

  class StringPort: public sci::cca::ports::StringPort{
  public:
    StringPort(BridgeServices* svc) : mysvcs(svc) { }
    std::string getString(); 
  private:
    BridgeServices* mysvcs;
  };

} //namepace SCIRun


#endif
