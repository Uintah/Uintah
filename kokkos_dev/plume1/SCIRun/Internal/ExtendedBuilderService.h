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
 *  ExtendedBuilderService.h: Implementation of the CCA ExtendedBuilderService interface for SCIRun
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#ifndef SCIRun_ExtendedBuilderService_h
#define SCIRun_ExtendedBuilderService_h

#include <SCIRun/Internal/BuilderService.h>

namespace SCIRun {

class SCIRunFramework;
class ConnectionEvent;

/**
 * \class ExtendedBuilderService
 *
 * \brief The ExtendedBuilderService class is a CCA interface used to instantiate and
 * connect components in a framework.
 *
 * A ExtendedBuilderService port's interface extends the standard cca BuilderService.
 * This implementation of the ExtendedBuilderService port is
 * specific for SCIRun2 and relies on much of the functionality in the
 * SCIRunFramework class and the various SCIRun component model classes.
 *
 * \sa SCIRunFramework
 * \sa InternalComponentModel
 */
class ExtendedBuilderService : public sci::cca::ports::BridgeExtendedBuilderService,
                       public InternalFrameworkServiceInstance
{
  public:
  virtual ~ExtendedBuilderService();
  
  /** Factory method for creating an instance of a ExtendedBuilderService class.
      Returns a reference counted pointer to a newly-allocated ExtendedBuilderService
      port.  The \em framework parameter is a pointer to the relevent framework
      and the \em name parameter will become the unique name for the new port.*/
  static InternalFrameworkServiceInstance *create(SCIRunFramework* framework);
  
  /** Creates an instance of the component of type \em className.  The
      parameter \em instanceName is the unique name of the newly created
      instance. This method is implemented through a createComponentInstance
      call to the SCIRunFramework. */
  virtual sci::cca::ComponentID::pointer
  createInstance(const std::string& instanceName,
                 const std::string& className,
                 const sci::cca::TypeMap::pointer &properties);

  /** Returns a list (array) of CCA ComponentIDs that exist in the
      ExtendedBuilderService's framework. */
  virtual SSIDL::array1<sci::cca::ComponentID::pointer> getComponentIDs();

  /** Returns a CCA TypeMap (i.e. a map) that represents any properties
      associated with component \em cid.
      key                    value          meaning
      cca.className          string         component type (standard key)
      x                      int            component x position
      y                      int            component y position
      LOADER NAME            string         loader name
      np                     int            number of nodes
      bridge                 bool           is a bridge
    */
  virtual sci::cca::TypeMap::pointer
  getComponentProperties(const sci::cca::ComponentID::pointer &cid);

  /** Associates the properties specified in \em map with an existing framework
      component \em cid.
      Null TypeMaps are not allowed. */
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

  /** Returns a list of \em uses ports for the component instance \em cid */
  virtual SSIDL::array1<std::string>
  getUsedPortNames(const sci::cca::ComponentID::pointer &cid);

  /** Returns a map of port properties exposed by the framework. */
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
          const std::string &usesPortName,
          const sci::cca::ComponentID::pointer &provider,
          const ::std::string &providesPortName);

  /** */
  virtual SSIDL::array1<sci::cca::ConnectionID::pointer>
  getConnectionIDs(const SSIDL::array1<sci::cca::ComponentID::pointer>
                       &componentList);


  /** Returns a CCA TypeMap that represents any properties associated
      with \em connID.
      key                    value          meaning
      user                   string         unique uses component name
      provider               string         unique provides component name
      uses port              string         uses port name
      provides port          string         provides port name
      bridge                 bool           is a bridge
    */
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
  getCompatiblePortList(const sci::cca::ComponentID::pointer &user,
                        const std::string &usesPortName,
                        const sci::cca::ComponentID::pointer &provider);


  // Bridge methods 
  /** */
  virtual SSIDL::array1<std::string>
  getBridgablePortList(const sci::cca::ComponentID::pointer &c1,
                       const std::string &port1,
                       const sci::cca::ComponentID::pointer &c2);

  /** */
  virtual std::string
  generateBridge(const sci::cca::ComponentID::pointer &c1,
                 const std::string &port1,
                 const sci::cca::ComponentID::pointer &c2,
                 const std::string &port2);
  // END Bridge methods
  
  /** */
  sci::cca::Port::pointer getService(const std::string &);

  /** */
  int addComponentClasses(const std::string &loaderName);

  /** */
  int removeComponentClasses(const std::string &loaderName);

  private:
    friend class ConnectionEventService;
    ExtendedBuilderService(SCIRunFramework* fwk);
    /** Returns the URL of this ExtendedBuilderService component's framework. */
    std::string getFrameworkURL();
    /** ? */
    void emitConnectionEvent(ConnectionEvent* event);

#ifdef HAVE_RUBY
    AutoBridge autobr;
#endif
  };
}

#endif
