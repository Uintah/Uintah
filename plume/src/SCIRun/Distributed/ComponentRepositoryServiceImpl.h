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
 *  ComponentRepositoryServiceImpl.h: Implementation of the CCA ComponentRegistry interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_ComponentRepositoryServiceImpl_h
#define SCIRun_ComponentRepositoryServiceImpl_h

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Distributed/ServiceImpl.h>

namespace SCIRun
{

  /**
   * \class ComponentRepositoryServiceImplService
   *
   * An implementation of a CCA ComponentRepositoryServiceImpl for SCIRun.  The
   * ComponentRepositoryServiceImpl handles 
   *
   */

  template<class Base>
  class ComponentRepositoryServiceImpl : public ServiceImpl<Base>
  {
  public:
    ComponentRepositoryServiceImpl(DistributedFramework*);
    virtual ~ComponentRepositoryServiceImpl();

    /** ? */
    sci::cca::Port::pointer getService(const std::string&);
    
    /** Returns a list of ComponentClassDescriptions that represents all of the
	component class types that may be instantiated in this framework.  In
	other words, calling getComponentClassName on each element in this list
	gives all of the components that the framework knows how to create. */

    virtual SSIDL::array1<sci::cca::ComponentClassDescription::pointer>  getAvailableComponentClasses();
    
  private:
    DistributedFramework *framework;
  };
  
} // end namespace SCIRun

#include <SCIRun/Distributed/ComponentRepositoryServiceImpl.code>

#endif
