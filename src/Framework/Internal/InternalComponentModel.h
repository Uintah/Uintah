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
 *  InternalComponentModel.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_InternalComponentModel_h
#define SCIRun_Framework_InternalComponentModel_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <SCIRun/ComponentModel.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <map>
#include <string>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace SCIRun
{
class ComponentDescription;
class InternalFrameworkServiceDescription;
class SCIRunFramework;

/**
 * \class InternalComponentModel
 *
 */
class InternalComponentModel : public ComponentModel
{
public:
    InternalComponentModel(SCIRunFramework* framework);
    virtual ~InternalComponentModel();

  /** */
  virtual const std::string getName() const { return "Internal"; }

  /** */
  virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
				     bool);

  /** */
  sci::cca::Port::pointer getFrameworkService(const std::string& type,
					      const std::string& componentName);

  /** */
  bool releaseFrameworkService(const std::string& type,
			       const std::string& componentName);

private:
  typedef std::map<std::string, InternalFrameworkServiceDescription*> FrameworkServicesMap;

  FrameworkServicesMap frameworkServices;
  SCIRun::Mutex lock_frameworkServices;

  void addService(InternalFrameworkServiceDescription* cd);
  //void addService(InternalComponenServicetDescription* cd);

  // Inherited ComponentModel functions that don't make sense for the
  // internal component model to implement:
  virtual void setComponentDescription(const std::string& type, const std::string& library)
  {
#if DEBUG
    std::cerr << "Warning: InternalComponentModel does not implement setComponentDescription"
              << std::endl;
#endif
  }
  virtual void destroyComponentList()
  {
#if DEBUG
    std::cerr << "Warning: InternalComponentModel does not implement destroyComponentList"
              << std::endl;
#endif
  }
  virtual void buildComponentList(const StringVector& files=StringVector())
  {
#if DEBUG
  std::cerr << "Warning: InternalComponentModel does not implement buildComponentList"
	    << std::endl;
#endif
  }
  virtual bool haveComponent(const std::string& type)
  {
#if DEBUG
  std::cerr << "Warning: InternalComponentModel does not implement haveComponent"
	<< std::endl;
#endif
    return false;
  }
  virtual ComponentInstance* createInstance(const std::string& name,
                                            const std::string& type,
                                            const sci::cca::TypeMap::pointer &tm)
  {
#if DEBUG
    std::cerr << "InternalComponentModel does not implement ComponentModel::createInstance(..)." << std::endl;
#endif
    return 0;
  }
  virtual bool destroyInstance(ComponentInstance *ci)
  {
#if DEBUG
    std::cerr << "InternalComponentModel does not implement ComponentModel::destroyInstance(..)." << std::endl;
#endif
    return true;
  }


  InternalComponentModel(const InternalComponentModel&);
  InternalComponentModel& operator=(const InternalComponentModel&);
};


} //End of namespace SCIRun

#endif
