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
 *  FixedPortServiceInfoBase.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Core_FixedPortServiceFactoryBase_h
#define SCIRun_Core_FixedPortServiceFactoryBase_h

#include <Core/Thread/Mutex.h>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::core;
  
  /**
   * \class FixedPortServiceFactoryBase
   *
   */
  
  template<class Interface>
  class FixedPortServiceFactoryBase : public Interface
  {
  public:
    typedef ServiceFactory::pointer pointer;
    
    FixedPortServiceFactoryBase(const CoreFramework::pointer &framework,
				const std::string& serviceName,
				const Port::pointer &server);
    virtual ~FixedPortServiceFactoryBase();
    
    /*
     * sci.cca.core.FixedPortServiceFactory interface
     */
    
    virtual std::string getName() { return name; }
    virtual PortInfo::pointer getService(const std::string &serviceName);
    virtual void releaseService(const std::string &portName);

  protected:
    PortInfo::pointer service;
    std::string name;
    int uses;
    Mutex lock;

    // prevent using these directly
    FixedPortServiceFactoryBase(const FixedPortServiceFactoryBase&);
    FixedPortServiceFactoryBase& operator=(const FixedPortServiceFactoryBase&);
  };
  
} // end namespace SCIRun

#endif
