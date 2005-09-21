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
 *  ComponentIDImpl.h: Implementation of the SCI CCA Extension
 *                    ComponentID interface for SCIRun
 *
 *  Written by:
 *   Yarden Livnat
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   Sept 2005
 *
 *  Copyright (C) 2005 SCI Institute
 *
 */

#ifndef SCIRun_ComponentIDImpl_h
#define SCIRun_ComponentIDImpl_h

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Distributed/DistributedFramework.h>

namespace SCIRun {

  namespace Distributed = sci::cca::distributed;

  template<class Base>
  class ComponentIDImpl : public Base 
  {
  public:
    typedef sci::cca::ComponentID::pointer pointer;

    ComponentIDImpl( const Distributed::DistributedFramework::pointer &framework,
		     const std::string &instanceName)
      : framework(framework), instanceName(instanceName)
    {}

    virtual ~ComponentIDImpl() {}
    
    /*
     * cca::ComponentID interface
     */
    virtual std::string getInstanceName() { return instanceName; }
    virtual std::string getSerialization() { return framework->getFrameworkID()->getString()+"/"+instanceName; }

    /* 
     * internal function
     */

    virtual Distributed::DistributedFramework::pointer getFramework() { return framework;}

  protected:
    Distributed::DistributedFramework::pointer framework;
    std::string instanceName;

  private:
    // prevent using these directly
    ComponentIDImpl(const ComponentIDImpl&);
    ComponentIDImpl& operator=(const ComponentIDImpl&);

  };
  
} // namespace SCIRun

#endif
