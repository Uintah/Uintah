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
 *  SCIRunFramework.h: An instance of the SCIRun framework
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_SCIRunFramework_h
#define SCIRun_Framework_SCIRunFramework_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/resourceReference.h>
#include <vector>
#include <map>
#include <string>

namespace SCIRun {
  class ComponentDescription;
  class ComponentModel;
  class ComponentInstance;
  class InternalComponentModel;
  class CCAComponentModel;
  class BabelComponentModel;
  class SCIRunFramework : public sci::cca::AbstractFramework {
  public:
    SCIRunFramework();
    virtual ~SCIRunFramework();

    virtual sci::cca::TypeMap::pointer createTypeMap();
    virtual sci::cca::Services::pointer getServices(const ::std::string& selfInstanceName,
						    const ::std::string& selfClassName,
						    const sci::cca::TypeMap::pointer& selfProperties);
    virtual void releaseServices(const sci::cca::Services::pointer& svc);
    virtual void shutdownFramework();
    virtual sci::cca::AbstractFramework::pointer createEmptyFramework();


    // Semi-private:
    // Used by builderService
    sci::cca::ComponentID::pointer 
      createComponentInstance( const std::string& name, const std::string& type, const sci::cca::TypeMap::pointer properties);
  
    
    void destroyComponentInstance(const sci::cca::ComponentID::pointer &cid, float timeout );

    sci::cca::Port::pointer getFrameworkService(const std::string& type,
						const std::string& componentName);
    bool releaseFrameworkService(const std::string& type,
				 const std::string& componentName);
    void registerComponent(ComponentInstance* ci, const std::string& name);




    ComponentInstance * unregisterComponent(const std::string& instanceName);
    void shutdownComponent(const std::string& name);
    void listAllComponentTypes(std::vector<ComponentDescription*>&,
			       bool);

    virtual int registerLoader(const ::std::string& slaveName, const ::SSIDL::array1< ::std::string>& slaveURLs);
    virtual int unregisterLoader(const ::std::string& slaveName);

    ComponentInstance* lookupComponent(const std::string& name);
    sci::cca::ComponentID::pointer lookupComponentID(const std::string& componentInstanceName);
    //do not delete the following 2 lines
    //void share(const sci::cca::Services::pointer &svc);
    //std::string createComponent(const std::string& name, const std::string& type);
    int destroyLoader(const std::string &loaderName);
  protected:
    friend class Services;
    friend class BuilderService;
    // Put these in a private structure to avoid #include bloat?
    std::vector<ComponentModel*> models;
    std::vector<sci::cca::ConnectionID::pointer> connIDs;
    std::vector<sci::cca::ComponentID::pointer> compIDs;
    std::map<std::string, ComponentInstance*> activeInstances;
    InternalComponentModel* internalServices;
    CCAComponentModel* cca;
    BabelComponentModel* babel;
    //Semaphore d_slave_sema;
    
  };
}

#endif

