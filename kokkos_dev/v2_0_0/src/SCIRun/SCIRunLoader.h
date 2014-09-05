/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SCIRunLoader.h: An instance of the SCIRun Parallel Component Loader
 *
 *  Written by:
 *   Kosta  & Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#ifndef SCIRun_SCIRunLoader_h
#define SCIRun_SCIRunLoader_h

#include <Core/CCA/spec/cca_sidl.h>
#include <vector>
#include <map>
#include <string>
#include <SCIRun/ComponentModel.h>
#include <SCIRun/CCA/CCAComponentDescription.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/resourceReference.h>
namespace SCIRun {

  class SCIRunLoader : public sci::cca::Loader{
  public:
    SCIRunLoader(const std::string& loaderName, const std::string& frameworkURL);
    virtual ~SCIRunLoader();

    virtual int createPInstance(const ::std::string& componentType, const std::string& componentName, SSIDL::array1<std::string>& componentURLs);
    virtual int createInstance(const std::string& componentType, const std::string& componentName, std::string &componentURL);

    virtual int destroyInstance(const std::string& componentName, float timeout);

    virtual int getAllComponentTypes(::SSIDL::array1< ::std::string>& componentTypes);
    
    virtual int shutdown(float timeout);

    int mpi_rank;
    int mpi_size;
    
  private:
    void buildComponentList();
    void readComponentDescription(const std::string& file);
    void destroyComponentList();
    std::string masterFrameworkURL;
    typedef std::map<std::string, CCAComponentDescription*> componentDB_type;
    componentDB_type components;
  };
}

#endif

