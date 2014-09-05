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
class ComponentEvent;
class InternalComponentModel;
class CCAComponentModel;
class BabelComponentModel;
class VtkComponentModel;
class CorbaComponentModel;

/**
 * \class SCIRunFramework
 * 
 * \brief An implementation of the CCA AbstractFramework for SCIRun2.
 *
 * The SCIRunFramework is a wrapper around a set of services for creating and
 * connecting components (the underlying framework).  Most interaction with the
 * underlying framework is actually done through the BuilderService port, but the
 * BuilderService relies on some of the methods and ivars implemented
 * in this class.  BuilderService is obtained through a CCA Services object,
 * which is the standard link between a CCA component and its framework.
 * A Services object is retrieved via SCIRunFramework::getServices.
 *
 * The BuilderService class is a standard CCA object responsible for
 * instantiating and connecting components.  In order to do this, it relies on
 * much of the functionality implemented in the SCIRunFramework class and in
 * the various ComponentModel classes contained in the SCIRunFramework.
 *
 * The SCIRunFramework maintains a list of ComponentModels that have
 * information about components available for instantiaion and use in the
 * framework.  ComponentModels are of various types (CCA, Vtk, Bridge, etc.)
 * but all have a standard API that the SCIRunFramework (acting as proxy for the
 * BuilderService) uses to instantiate components registered in that component
 * model.
 *
 * 
 * \sa BuilderService AbstractFramework 
 * \todo createTypeMap is not fully implemented
 * \todo releaseServices is not fully implemented
 * \todo shutdownFramework is not fully implemented
 * \todo implement shutdownComponent?
 * \todo createEmptyFramework is not fully implemented
 * \todo register a creation event for component (in registerComponent)  */
class SCIRunFramework : public sci::cca::AbstractFramework
{
public:
  SCIRunFramework();
  virtual ~SCIRunFramework();
  
  /** Standard CCA::AbstractFramework method that creates and returns an empty
      CCA Map object appropriate for use in an ensuing call to getServices. */
  virtual sci::cca::TypeMap::pointer createTypeMap();

  /** Returns a handle to the services provided by this framework (getting and
      releasing ports, registering ports, querying port properties, etc.).  The
      framework services are encapsulated in a standard CCA interface called a
      Services object.  The Services object is the real interface to the
      underlying framework and is used to retrieve ports, including the 
      BuilderService Port, which handles connections among components.

      The getServices method creates a new Services object that appears as a
      component instance in the Framework, effectively registering an image of
      the calling program as a component instance within the framework.
      Multiple Services objects may be created with different names from the
      same or different calling programs.

      The \em selfInstanceName parameter is the name given to the component
      instance. The \em selfClassName parameter is the component type of the
      calling program, as it will appear in the framework.  The
      \em selfProperties are the properties of the component instance (and may
      be null).  If the selfInstanceName is already in use by another component
      in the framework, this method will throw a CCAException.	 */
  virtual sci::cca::Services::pointer getServices(
                 const ::std::string& selfInstanceName,
                 const ::std::string& selfClassName,
                 const sci::cca::TypeMap::pointer& selfProperties);

  /** Informs the framework that the Services object referenced by \em svc is
      no longer needed by the calling program.  This method invalidates the
      associated component image in the underlying framework and also
      invalidates any ComponentIDs or ConnectionIDs contained in the referenced
      Services  object. */
  virtual void releaseServices(const sci::cca::Services::pointer& svc);

  /** Tells the framework it is no longer needed and to clean up after itself. */  
  virtual void shutdownFramework();

  /** Creates a new SCIRunFramework instance.  The new framework instance is
      not a copy of the existing framework and does not contain any of the
      user-instantiated components from the original.  This method exists as a
      kind of object factory for creating a new SCIRunFramework using an
      abstract API. */ 
  virtual sci::cca::AbstractFramework::pointer createEmptyFramework();
  
  /** Creates an instance of the component defined by the string ``type'',
      which must uniquely define the type of the component.  The component
      instance is given the name ``name''.   If the instance name is not
      specified (i.e. the method is passed an empty string), then the component
      will be assigned a unique name automatically.

      This method is ``semi-private'' and intended to be called only by the
      BuilderService class.  It works by searching the list of ComponentModels
      (the ivar \em models) for a matching registered type, and then calling
      the createInstance method on the appropriate ComponentModel object. */
  sci::cca::ComponentID::pointer createComponentInstance(
                                     const std::string& name,
                                     const std::string& type,
                                     const sci::cca::TypeMap::pointer properties);
  
  /** Eliminates the component instance ``cid'' from the scope of the
      framework.  The ``timeout'' parameter specifies the maximum allowable
      wait time for this operation.  A timeout of 0 leaves the wait time up to
      the framework.  If the destroy operation is not completed in the maximum
      allowed number of seconds, or the referenced component does not exist,
      then a CCAException is thrown.

      Like createComponentInstance, this method is only intended to be called
      by the BuilderService class.  It searches the list of registered
      components (compIDs) for the matching component ID, unregisters it, finds
      the correct ComponentModel for the type, then calls
      ComponentModel::destroyInstance to properly destroy the component.
  */
  void destroyComponentInstance(const sci::cca::ComponentID::pointer &cid,
                                float timeout );

  /** ? */
  sci::cca::Port::pointer getFrameworkService(const std::string& type,
                                              const std::string& componentName);

  /** ? */
  bool releaseFrameworkService(const std::string& type,
                               const std::string& componentName);

  /** Adds a description of a component instance (class ComponentInstance) to
      the list of active components.  The component instance description
      includes the component type name, the instance name, and the pointer to
      the allocated component.  When a \em name conflicts with an existing
      registered component instance name, this method will automatically append
      an integer to create a new, unique instance name.*/ 
  void registerComponent(ComponentInstance* ci, const std::string& name);

  /** Removes a component instance description from the list of active
      framework components.  Returns the pointer to the component description
      that was successfully unregistered. */
  ComponentInstance * unregisterComponent(const std::string& instanceName);

  /** This method is unimplemented. */
  //void shutdownComponent(const std::string& name);

  /** Compiles a list of all ComponentDescriptions in all ComponentModels
      contained in this framework.*/
  void listAllComponentTypes(std::vector<ComponentDescription*>&,
                             bool);

  /** ?  */
  virtual int registerLoader(const ::std::string& slaveName,
                             const ::SSIDL::array1< ::std::string>& slaveURLs);
  
  /** ? */
  virtual int unregisterLoader(const ::std::string& slaveName);

  /** ? */
  int destroyLoader(const std::string &loaderName);

  /** Returns a component instance description (ComponentInstance) from the
      list of active framework components.  Returns a null pointer if the
      instance \em name is not found. */
  ComponentInstance* lookupComponent(const std::string& name);

  /** ? */
  sci::cca::ComponentID::pointer lookupComponentID(const std::string&
                                                   componentInstanceName);

  /** ? */
  ComponentModel* lookupComponentModel(const std::string& name);

  //do not delete the following 2 lines
  //void share(const sci::cca::Services::pointer &svc);
  //std::string createComponent(const std::string& name, const std::string& type);

  /** A pointer to the framework's SCIRun dataflow component model.  This
      pointer is also in the \em models list.  The \em dflow ivar is public so
      that it can be accessed by the SCIRun scheduler service.  */
  SCIRunComponentModel* dflow; 

protected:
  friend class Services;
  friend class BuilderService;

  /** ? */
  void emitComponentEvent(ComponentEvent* event);

  // Put these in a private structure to avoid #include bloat?
  /** A list of component models available in this framework. */
  std::vector<ComponentModel*> models;

  /** ? */
  std::vector<sci::cca::ConnectionID::pointer> connIDs;

  /** ? */
  std::vector<sci::cca::ComponentID::pointer> compIDs;

  /** The set of registered components available in the framework, indexed by
      their instance names. */
  std::map<std::string, ComponentInstance*> activeInstances;

  /** A pointer to the special SCIRun \em internal component model, whose
  framework-related components are not intended to be exposed to the user.
  These components include the BuilderService port. */
  InternalComponentModel* internalServices;

  /** A pointer to the CCA component model. This pointer is also stored in the
      \em models list. */
  CCAComponentModel* cca;

  /** A pointer to the babel componet model.  This pointer is also stored in
      the \em models list. (Can this ivar be removed?)*/
  BabelComponentModel* babel;

  /** A poitner to the Vtk component model.  This pointer is also stored in the
      \em models list. (Can this ivar be removed?) */
  VtkComponentModel* vtk;

  CorbaComponentModel* corba;
  
  //Semaphore d_slave_sema;
};

} // end namespace SCIRun

#endif
