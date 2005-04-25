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
 *  BuilderService.h: Implementation of the CCA BuilderService interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_BuilderService_h
#define SCIRun_BuilderService_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentInstance.h>
#include <SCIRun/Bridge/AutoBridge.h>

namespace SCIRun {

class SCIRunFramework;
class ConnectionEvent;

/**
 * \class BuilderService
 *
 * \brief The BuilderService class is a CCA interface used to instantiate and
 * connect components in a framework.
 *
 * A BuilderService port's interface is standard for any CCA-compliant
 * AbstractFramework.  This implementation of the BuilderService port is
 * specific for SCIRun2 and relies on much of the functionality in the
 * SCIRunFramework class and the various SCIRun component model classes.
 *
 *
 * \todo getComponentProperties is not finished?
 * \todo setComponentProperties is not finished?
 * \todo getDeserialization is not finished?
 * \todo addComponentClasses not implemented
 * \todo removeComponentClasses not implemented
 *
 * \sa SCIRunFramework
 * \sa InternalComponentModel
 */
class BuilderService : public sci::cca::ports::BuilderService,
                       public InternalComponentInstance
{
  public:
  virtual ~BuilderService();

  /** Factory method for creating an instance of a BuilderService class.
      Returns a reference counted pointer to a newly-allocated BuilderService
      port.  The \em framework parameter is a pointer to the relevent framework
      and the \em name parameter will become the unique name for the new port.*/
  static InternalComponentInstance*
  create(SCIRunFramework* framework, const std::string& name);
  
  /** Creates an instance of the component of type \em className.  The
      parameter \em instanceName is the unique name of the newly created
      instance. This method is implemented through a createComponentInstance
      call to the SCIRunFramework. */
  virtual sci::cca::ComponentID::pointer
  createInstance(const std::string& instanceName,
                 const std::string& className,
                 const sci::cca::TypeMap::pointer &properties);

  /** Returns a list (array) of CCA ComponentIDs that exist in the
      BuilderService's framework. */
  virtual SSIDL::array1<sci::cca::ComponentID::pointer> getComponentIDs();

  /** Returns a CCA TypeMap (i.e. a map) that represents any properties
      associated with component \em cid */
  virtual sci::cca::TypeMap::pointer
  getComponentProperties(const sci::cca::ComponentID::pointer &cid);

  /** Associates the properties specified in \em map with an existing framework
      component \em cid.*/
  virtual void
  setComponentProperties(const sci::cca::ComponentID::pointer &cid,
                         const sci::cca::TypeMap::pointer &map);

  /** Returns the Component ID (opaque reference to a component instantiation)
      for the serialized component reference \em s. */
  virtual sci::cca::ComponentID::pointer
  getDeserialization(const std::string &s);

  /** Returns the Component ID (opaque reference to a component instantiation)
      for the component instance named \em componentInstanceName. */
  virtual sci::cca::ComponentID::pointer
  getComponentID(const std::string &componentInstanceName);

  /** Removed the component instance \em toDie from the scope of the framework.
       The \em timeout parameter gives the maximum allowable wait (in seconds)
       for the operation to be completed.  An exception is thrown if the \em
       toDie component does not  exist, or cannot be destroyed in the given \em
       timeout period. */
  virtual void
  destroyInstance(const sci::cca::ComponentID::pointer &toDie, float timeout);

  /** Returns a list of \em provides ports for the given component instance \em
      cid.*/
  virtual SSIDL::array1<std::string>
  getProvidedPortNames(const sci::cca::ComponentID::pointer &cid);

  /** */
  virtual SSIDL::array1<std::string>
  getUsedPortNames(const sci::cca::ComponentID::pointer &cid);

  /** */
  virtual sci::cca::TypeMap::pointer
  getPortProperties(const sci::cca::ComponentID::pointer &cid,
                    const std::string& portname);
  
  /** */
  virtual void
  setPortProperties(const sci::cca::ComponentID::pointer &cid,
                    const std::string &portname,
                    const sci::cca::TypeMap::pointer &map);

  /** */
  virtual sci::cca::ConnectionID::pointer
  connect(const sci::cca::ComponentID::pointer &user,
          const std::string &usingPortName,
          const sci::cca::ComponentID::pointer &provider,
          const ::std::string &providingPortName);

  /** */
  virtual SSIDL::array1<sci::cca::ConnectionID::pointer>
  getConnectionIDs(const SSIDL::array1<sci::cca::ComponentID::pointer>
                       &componentList);

  /** */
  virtual sci::cca::TypeMap::pointer
  getConnectionProperties(const sci::cca::ConnectionID::pointer &connID);

  /** */
  virtual void
  setConnectionProperties(const sci::cca::ConnectionID::pointer &connID,
                          const sci::cca::TypeMap::pointer &map);

  /** */
  virtual void
  disconnect(const sci::cca::ConnectionID::pointer &connID, float timeout);

  /** */
  virtual void
  disconnectAll(const sci::cca::ComponentID::pointer &id1,
                const sci::cca::ComponentID::pointer &id2,
                float timeout);

  /** */
  virtual SSIDL::array1<std::string>
  getCompatiblePortList(const sci::cca::ComponentID::pointer &c1,
                        const std::string &port1,
                        const sci::cca::ComponentID::pointer &c2);


  // Bridge methods 
  /** */
  virtual SSIDL::array1<std::string>
  getBridgablePortList(const sci::cca::ComponentID::pointer &c1,
                       const std::string &port1,
                       const sci::cca::ComponentID::pointer &c2);

  /** */
  virtual std::string
  generateBridge(const sci::cca::ComponentID::pointer &c1,
                 const string &port1,
                 const sci::cca::ComponentID::pointer &c2,
                 const string &port2);
  // END Bridge methods

  
  /** */
  sci::cca::Port::pointer getService(const std::string &);

  /** */
  int addComponentClasses(const std::string &loaderName);

  /** */
  int removeComponentClasses(const std::string &loaderName);

  /** */
  int
  addLoader(const std::string &loaderName, const std::string &user,
            const std::string &domain, const std::string &loaderPath);

  /** */
  int removeLoader(const std::string &name);

  //virtual void registerFramework(const std::string &frameworkURL); 
  //virtual void registerServices(const sci::cca::Services::pointer &svc);

  private:
    BuilderService(SCIRunFramework* fwk, const std::string& name);
    /** Returns the URL of this BuilderService component's framework. */
    std::string getFrameworkURL();
    /** ? */
    void emitConnectionEvent(ConnectionEvent* event);

    std::vector<sci::cca::Services::pointer> servicesList;  
    AutoBridge autobr;
  };
}

#endif
