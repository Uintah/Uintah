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
 *  PortInfoBase.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_CCA_PortInfoBase_h
#define SCIRun_CCA_PortInfoBase_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>

#include <map>
#include <string>
#include <vector>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::core;

  /**
   * \class PortInfoBase
   *
   */
  template<class Interface> 
  class PortInfoBase : public Interface
  {
  public:
    PortInfoBase(const std::string& portname, 
		 const std::string& classname,
		 const TypeMap::pointer& properties,
		 PortType porttype);
    
    PortInfoBase(const std::string& portname, 
		 const std::string& classname,
		 const TypeMap::pointer& properties,
		 const Port::pointer& port,
		 PortType porttype);

    ~PortInfoBase();
    
    /* 
     *  Baseementation of PortInfo interface
     */
    
    /** */
    virtual bool connect(const PortInfo::pointer &);;
    /** */
    virtual bool disconnect(const PortInfo::pointer &);
    /** */
    virtual bool available();
    /** */
    virtual bool canConnectTo(const PortInfo::pointer &);
    /** */
    virtual int numOfConnections();
    /** */
    virtual bool isConnected();

    /** */
    virtual Port::pointer getPort();
    /** */
    virtual PortInfo::pointer getPeer();
    /** */
    virtual PortType getPortType();

    /** */
    virtual TypeMap::pointer getProperties();

    /** */
    virtual std::string getName();
    /** */
    virtual std::string getClass();

    /** */
    virtual void incrementUseCount();
    /** */
    virtual bool decrementUseCount();

    virtual bool inUse();

  protected:
    void setPort( const Port::pointer &port);
    
  private:
    std::string name;
    std::string type;
    TypeMap::pointer properties;
    std::vector<PortInfo::pointer> connections;
    
    core::PortType portType;
    Port::pointer port;

    int useCount;
    SCIRun::Mutex lock;

    PortInfoBase(const PortInfoBase&);
    PortInfoBase& operator=(const PortInfoBase&);
  };

} // end namespace SCIRun

//#include <SCIRun/Core/PortInfoBase.code>

#endif
