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
 *  CCAComponentInfoImpl.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Distributed_CCAComponentInfoImpl_h
#define SCIRun_Distributed_CCAComponentInfoImpl_h

#include <SCIRun/Distributed/ComponentInfoImpl.h>


namespace SCIRun {
  
  namespace Plume = sci::cca::plume;

  /**
   * \class CCAComponentInfo
   *
   */
  
  template<class Base>
  class CCAComponentInfoImpl : public ComponentInfoImpl<Base>
  {
  public:
    typedef Plume::CCAComponentInfo::pointer pointer;
    typedef typename ComponentInfoImpl<Base>::PortMap PortMap;

    using typename ComponentInfoImpl<Base>::ports_lock;
    using typename ComponentInfoImpl<Base>::ports;

    CCAComponentInfoImpl(const Distributed::DistributedFramework::pointer &framework,
			 const std::string& instanceName,
			 const std::string& className,
			 const sci::cca::TypeMap::pointer& properties,
			 const sci::cca::Component::pointer& component);

    virtual ~CCAComponentInfoImpl();

    // from cca.Services
    virtual void releasePort(const std::string&);
    virtual void registerUsesPort(const std::string&, const std::string&, const sci::cca::TypeMap::pointer &);
    virtual void unregisterUsesPort(const std::string&);
    virtual void addProvidesPort(const sci::cca::Port::pointer &, 
				 const std::string&, const std::string&, const sci::cca::TypeMap::pointer &);
    virtual void registerForRelease( const sci::cca::ComponentRelease::pointer &);
    virtual void removeProvidesPort(const std::string&);

    virtual sci::cca::Port::pointer getPort(const std::string&);
    virtual sci::cca::Port::pointer getPortNonblocking(const std::string&);
    virtual sci::cca::TypeMap::pointer createTypeMap();
    virtual sci::cca::TypeMap::pointer getPortProperties(const std::string &);
    virtual sci::cca::ComponentID::pointer getComponentID();
    
  private:
    // prevent using these directly
    CCAComponentInfoImpl<Base>(const CCAComponentInfoImpl<Base>&);
    CCAComponentInfoImpl<Base>& operator=(const CCAComponentInfoImpl<Base>&);
  };
  
} // end namespace SCIRun

#endif
