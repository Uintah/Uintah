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
 *  SingletonServiceInfo.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 * Adapted from the Plume framework implementation by:
 *   Ayla Khan
 *   SCI Institute
 *   University of Utah
 *   July 2007
 *
 */

#ifndef Framework_Core_SingletonServiceFactory_h
#define Framework_Core_SingletonServiceFactory_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <Core/Util/Assert.h>

#include <string>

// Babel includes
#include <sidl.hxx>

#include "scijump.hxx"
#include "sci_cca.hxx"
#include "gov_cca.hxx"

// Babel includes

namespace scijump {
  namespace core {

    using namespace SCIRun;

    class ServiceFactory {
    public:
      virtual ~ServiceFactory() {}
      virtual std::string getName() = 0;
      virtual sci::cca::core::PortInfo get(const std::string &serviceName) = 0;
      virtual void release(const std::string &portName) = 0;
    };


    /**
     * \class SingletonServiceFactory
     *
     * Wrapper for a framework service.
     * Enforces singleton pattern.
     */

    template<class Service>
    class SingletonServiceFactory : public ServiceFactory {
    public:
      SingletonServiceFactory(const scijump::SCIJumpFramework &framework, const std::string &serviceName)
        : framework(framework), serviceName(serviceName), uses(0), lock("SingletonServiceFactory") {}

      virtual ~SingletonServiceFactory() {
        //framework = 0;
        //service = 0;
      }

      std::string getName() { return serviceName; }
      //virtual PortInfo::pointer getService(const std::string &serviceName, const ComponentInfo::pointer &requester);
      virtual sci::cca::core::PortInfo get(const std::string &serviceName) {
        Guard guard(&lock);
        if ( service._is_nil() ) {
          //service = PortInfo::pointer( new PortInfoImpl( serviceName, serviceName, TypeMap::pointer(0), Service::create(framework), ProvidePort));

          service = scijump::BabelPortInfo::_create();
          sci::cca::core::FrameworkService fs = Service::create(framework);
          gov::cca::Port port = sidl::babel_cast<gov::cca::Port>(fs);

          // Although we probably should not have asserts in production code,
          // a babel_cast failure here would indicate a fairly serious problem.
          ASSERT(port._not_nil());

          service.initialize(port, serviceName, serviceName, sci::cca::core::PortType_ProvidesPort, scijump::TypeMap::_create());
        }
        uses++;
        return service;
      }

      virtual void release(const std::string &portName) {
        Guard guard(&lock);
        uses--;
      }

    protected:
      scijump::SCIJumpFramework framework; // Babelized framework
      scijump::BabelPortInfo service;

      std::string serviceName;
      int uses;
      Mutex lock;

      // prevent using these directly
      SingletonServiceFactory(const SingletonServiceFactory&);
      SingletonServiceFactory& operator=(const SingletonServiceFactory&);
    };

  } // end namespace
} // end namespace

#endif
