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
 *  CorbaComponentModel.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Corba/CorbaComponentModel.h>
#include <SCIRun/Corba/CorbaComponentDescription.h>
#include <SCIRun/Corba/CorbaComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/soloader.h>
#include <Core/Util/sci_system.h>
#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <iostream>
#include <SCIRun/Corba/Port.h>
#include <SCIRun/Corba/Component.h>
#include <SCIRun/Corba/Services.h>

extern "C" {
 #include <string.h>
}

namespace SCIRun {

const std::string CorbaComponentModel::DEFAULT_PATH("/CCA/Components/CORBA/xml");

CorbaComponentModel::CorbaComponentModel(SCIRunFramework* framework,
				     const StringVector& xmlPaths)
  : ComponentModel("corba", framework),
    lock_components("CorbaComponentModel::components lock")
{
  // move to framework properties
  // Record the path containing DLLs for components.
  const char *dll_path = getenv("SIDL_DLL_PATH");
  if (dll_path != 0) {
    this->setSidlDLLPath(std::string(dll_path));
  } else {
    this->setSidlDLLPath(sci_getenv("SCIRUN_OBJDIR") + std::string("/lib"));
  }

  buildComponentList(xmlPaths);
}

CorbaComponentModel::~CorbaComponentModel()
{
  destroyComponentList();
}

void CorbaComponentModel::destroyComponentList()
{
  SCIRun::Guard g1(&lock_components);

  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void CorbaComponentModel::buildComponentList(const StringVector& files)
{
  destroyComponentList();

  if (files.empty()) {
    StringVector xmlPaths_;
    getXMLPaths(framework, xmlPaths_);
    for (StringVector::iterator iter = xmlPaths_.begin(); iter != xmlPaths_.end(); iter++) {
      parseComponentModelXML(*iter, this);
    }
  } else {
    for (StringVector::const_iterator iter = files.begin(); iter != files.end(); iter++) {
      parseComponentModelXML(*iter, this);
    }
  }
}

void
CorbaComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
  // library will actually be a path to an executable
  CorbaComponentDescription* cd = new CorbaComponentDescription(this, type, library);
  Guard g(&lock_components);
  componentDB_type::iterator iter = components.find(cd->getType());
  if (iter != components.end()) {
    std::cerr << "WARNING: Multiple definitions exist for " << cd->getType() << std::endl;
  } else {
    components[cd->getType()] = cd;
  }
}

bool CorbaComponentModel::haveComponent(const std::string& type)
{
  SCIRun::Guard g1(&lock_components);
  return components.find(type) != components.end();
}

ComponentInstance*
CorbaComponentModel::createInstance(const std::string& name,
				    const std::string& type,
				    const sci::cca::TypeMap::pointer &tm)
{
  corba::Component *component;

  lock_components.lock();
  componentDB_type::iterator iter = components.find(type);
  if (iter == components.end()) { // could not find this component
    return 0;
  }
  lock_components.unlock();

  std::string exec_name = iter->second->getExecPath();

  // If the component library does not exist
  // do not create the component instance.
  struct stat buf;
  if (LSTAT(exec_name.c_str(), &buf) < 0) {
    if (errno == ENOENT) {
      throw InternalError("File " + exec_name + " does not exist.", __FILE__, __LINE__);
    } else {
      throw InternalError("LSTAT on " + exec_name + " failed.", __FILE__, __LINE__);
    }
  }

  component = new corba::Component();
  corba::Services *svc=new corba::Services(component);
  component->setServices(svc);

  sci::cca::CorbaServices::pointer services(svc);
  services->addReference();
  std::string svc_url = services->getURL().getString();

  /*
	pid_t child_id=fork();
	if(child_id==0){
	//this is child process
	execl(exec_name.c_str(), exec_name.c_str(), svc_url.c_str(), NULL);
	exit(0);
	}else{
	std::cout<<"**** main process is still running ****"<<std::endl;
	//this is parent process
	//do nothing
	}
  */
  std::string cmdline = exec_name + " " + svc_url + "&";
  const int status = sci_system(cmdline.c_str());
  if (status != 0) { // failed
    throw InternalError("Corba service " + cmdline + " is not available.", __FILE__, __LINE__);
  }

  services->check();

  //TODO: do we really need a "component" here?
  CorbaComponentInstance* ci =
    new CorbaComponentInstance(framework, name, type, tm, component);
  return ci;
}

bool CorbaComponentModel::destroyInstance(ComponentInstance *ci)
{
  //TODO: pre-deletion clearance.
  delete ci;
  return true;
}

const std::string CorbaComponentModel::getName() const
{
  return "Corba";
}

void CorbaComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list, bool /*listInternal*/)
{
  SCIRun::Guard g1(&lock_components);
  for (componentDB_type::iterator iter=components.begin(); iter != components.end(); iter++) {
    list.push_back(iter->second);
  }
}

} // end namespace SCIRun
