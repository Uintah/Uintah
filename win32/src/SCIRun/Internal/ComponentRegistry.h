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
 *  ComponentRegistry.h: Implementation of the CCA ComponentRegistry interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_ComponentRegistry_h
#define SCIRun_ComponentRegistry_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>
#include <vector>

namespace SCIRun
{
class SCIRunFramework;

/**
 * \class ComponentRegistry
 *
 * An implementation of a CCA ComponentRepository for SCIRun.  The
 * ComponentRegistry handles 
 *
 */
class ComponentRegistry : public sci::cca::ports::ComponentRepository,
                          public InternalFrameworkServiceInstance
{
public:
  virtual ~ComponentRegistry();
  
  /** Factory method for allocating new ComponentRegistry objects.  Returns
      a smart pointer to the newly allocated object registered to the framework
      \em fwk with the instance name \em name. */
  static InternalFrameworkServiceInstance* create(SCIRunFramework* fwk);
  
  /** Returns this service (?) - overrides InternalComponentInstance::getService. */
  sci::cca::Port::pointer getService(const std::string&);

  /** Returns a list of ComponentClassDescriptions that represents all of the
      component class types that may be instantiated in this framework.  In
      other words, calling getComponentClassName on each element in this list
      gives all of the components that the framework knows how to create. */
  virtual SSIDL::array1<sci::cca::ComponentClassDescription::pointer>
  getAvailableComponentClasses();

  virtual void addComponentClass(const std::string& componentClassName);

private:
  ComponentRegistry(SCIRunFramework* fwk);
};

} // end namespace SCIRun

#endif
