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
 *  VtkComponentModel.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Vtk/VtkComponentModel.h>
#include <SCIRun/Vtk/VtkComponentDescription.h>
#include <SCIRun/Vtk/VtkComponentInstance.h>
#include <SCIRun/Vtk/Port.h>
#include <SCIRun/Vtk/Component.h>
#include <SCIRun/SCIRunFramework.h>
#include <Core/Util/soloader.h>
#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>

#include <iostream>

extern "C" {
#  include <string.h>
}

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace SCIRun {

const std::string VtkComponentModel::DEFAULT_PATH("/CCA/Components/VTK/xml");


VtkComponentModel::VtkComponentModel(SCIRunFramework* framework,
				     const StringVector& xmlPaths)
  : ComponentModel("vtk", framework),
    lock_components("VtkComponentModel::components lock")
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

VtkComponentModel::~VtkComponentModel()
{
  destroyComponentList();
}

void VtkComponentModel::destroyComponentList()
{
  SCIRun::Guard g1(&lock_components);
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++) {
    delete iter->second;
  }
  components.clear();
}

void VtkComponentModel::buildComponentList(const StringVector& files)
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

void VtkComponentModel::setComponentDescription(const std::string& type, const std::string& library)
{
  // Register this component
  VtkComponentDescription* cd = new VtkComponentDescription(this, type, library);
  Guard g1(&lock_components);
  componentDB_type::iterator iter = components.find(cd->getType());
  if (iter != components.end()) {
    std::cerr << "WARNING: Multiple definitions exist for " << cd->getType() << std::endl;
  } else {
    components[cd->getType()] = cd;
  }
}

bool VtkComponentModel::haveComponent(const std::string& type)
{
  SCIRun::Guard g1(&lock_components);
  return components.find(type) != components.end();
}

ComponentInstance*
VtkComponentModel::createInstance(const std::string& name,
				  const std::string& type,
				  const sci::cca::TypeMap::pointer& tm)
{
  vtk::Component *component;

  Guard g1(&lock_components);
  componentDB_type::iterator iter = components.find(type);
  if (iter == components.end()) { // could not find this component
    return 0;
  }

  // Get the list of DLL paths to search for the appropriate component library
  std::vector<std::string> possible_paths = splitPathString(this->getSidlDLLPath());
  LIBRARY_HANDLE handle;

  for (std::vector<std::string>::iterator it = possible_paths.begin();
       it != possible_paths.end(); it++) {
    std::string so_name = *it + "/" + iter->second->getLibrary();
    handle = GetLibraryHandle(so_name.c_str());
    if (handle)  {  break;   }
  }

  if ( !handle ) {
    std::cerr << "Could not find component DLL: " << iter->second->getLibrary()
	      << " for type " << type << std::endl;
    std::cerr << SOError() << std::endl;
    return 0;
  }

  std::string makername = "make_"+type;
  for(int i = 0; i < static_cast<int>(makername.size()); i++) {
    if (makername[i] == '.') { makername[i]='_'; }
  }

#if DEBUG
  std::cerr << "looking for symbol:" << makername << std::endl;
#endif
  void* maker_v = GetHandleSymbolAddress(handle, makername.c_str());
  if(!maker_v) {
    std::cerr <<"Cannot load component symbol " << makername << std::endl;
    std::cerr << SOError() << std::endl;
    return 0;
  }
  vtk::Component* (*maker)() = (vtk::Component* (*)())(maker_v);
  component = (*maker)();

  VtkComponentInstance* ci =
      new VtkComponentInstance(framework, name, type, tm, component);
  return ci;
}

bool VtkComponentModel::destroyInstance(ComponentInstance *ci)
{
  //TODO: pre-deletion clearance.
  delete ci;
  return true;
}

const std::string VtkComponentModel::getName() const
{
  return "Vtk";
}

void VtkComponentModel::listAllComponentTypes(std::vector<ComponentDescription*>& list, bool /*listInternal*/)
{
  SCIRun::Guard g1(&lock_components);
  for(componentDB_type::iterator iter=components.begin();
      iter != components.end(); iter++){
    list.push_back(iter->second);
  }
}


} // end namespace SCIRun
