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
 *  PortInfoImpl.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_CCA_PortInfoImpl_h
#define SCIRun_CCA_PortInfoImpl_h

#include <Core/CCA/spec/sci_sidl.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <map>
#include <string>
#include <vector>

namespace SCIRun {
  
  namespace Distributed = sci::cca::distributed;
  /**
   * \class PortInfoImpl
   *
   */
  template<class Base> 
  class PortInfoImpl : public Base
  {
  public:
    PortInfoImpl(const std::string& portname, 
		 const std::string& classname,
		 const sci::cca::TypeMap::pointer& properties,
		 Distributed::PortType porttype);
    
    PortInfoImpl(const std::string& portname, 
		 const std::string& classname,
		 const sci::cca::TypeMap::pointer& properties,
		 const sci::cca::Port::pointer& port,
		 Distributed::PortType porttype);

    ~PortInfoImpl();
    
    /* 
     *  Implementation of Distributed::PortInfo interface
     */
    
    /** */
    virtual bool connect(const Distributed::PortInfo::pointer &);;
    /** */
    virtual bool disconnect(const Distributed::PortInfo::pointer &);
    /** */
    virtual bool available();
    /** */
    virtual bool canConnectTo(const Distributed::PortInfo::pointer &);
    /** */
    virtual int numOfConnections();

    /** */
    virtual sci::cca::Port::pointer getPort();
    /** */
    virtual Distributed::PortInfo::pointer getPeer();
    /** */
    virtual Distributed::PortType getPortType();
    
    /** */
    virtual std::string getName();
    /** */
    virtual std::string getType();

    /** */
    virtual void incrementUseCount();
    /** */
    virtual bool decrementUseCount();

  protected:
    void setPort( const sci::cca::Port::pointer &port);
    
  private:
    std::string name;
    std::string type;
    sci::cca::TypeMap::pointer properties;
    std::vector<Distributed::PortInfo::pointer> connections;
    
    sci::cca::distributed::PortType portType;
    sci::cca::Port::pointer port;

    int useCount;
    SCIRun::Mutex lock;

    PortInfoImpl(const PortInfoImpl&);
    PortInfoImpl& operator=(const PortInfoImpl&);
  };

} // end namespace SCIRun

//#include <SCIRun/Distributed/PortInfoImpl.code>

#endif
