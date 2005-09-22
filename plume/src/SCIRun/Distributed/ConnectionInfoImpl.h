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
 *  ConnectionInfoImpl.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Distributed_ConnectionInfoImpl_h
#define SCIRun_Distributed_ConnectionInfoImpl_h

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Distributed/ConnectionIDImpl.h>
#include <map>
#include <string>

namespace SCIRun {
  
  class DistributedFramework;
  namespace Distributed = sci::cca::distributed;

  /**
   * \class ConnectionInfo
   *
   */
  
  template<class Base>
  class ConnectionInfoImpl : public ConnectionIDImpl<Base>
  {
  public:
    typedef Distributed::ConnectionInfo::pointer pointer;
    typedef sci::cca::ComponentID::pointer cid_pointer;

    
    ConnectionInfoImpl(const cid_pointer &provider, 
		       const std::string &providerPortName,
		       const cid_pointer &user,
		       const std::string &userPortName);
    virtual ~ConnectionInfoImpl();
    
    /*
     * sci.cca.distributed.ConnectionInfo interface
     */
    
    virtual sci::cca::TypeMap::pointer getProperties() { return properties; }
    virtual void setProperties(const sci::cca::TypeMap::pointer &prpperties) { this->properties = properties; }
    
  private:
    sci::cca::TypeMap::pointer properties;
        
    // prevent using these directly
    ConnectionInfoImpl(const ConnectionInfoImpl&);
    ConnectionInfoImpl& operator=(const ConnectionInfoImpl&);
  };
  
} // end namespace SCIRun

#endif
