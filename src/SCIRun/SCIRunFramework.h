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

#include <SCIRun/Dataflow/SCIRunComponentModel.h>
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
  class VtkComponentModel;
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
    //Dataflow component model needs to be public to give access to the scheduler service
    SCIRunComponentModel* dflow; 
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
    VtkComponentModel* vtk;
    //Semaphore d_slave_sema;
    
  };
}

#endif

