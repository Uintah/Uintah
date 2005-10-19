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
 *  BuilderServiceBase.h: Implementation of the CCA BuilderService interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_BuilderServiceBase_h
#define SCIRun_BuilderServiceBase_h

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::core;

  /**
   * \class BuilderServiceBase
   *
   * \brief The BuilderService class is a CCA interface used to instantiate and
   * connect components in a framework.
   *
   * A BuilderService port's interface is standard for any CCA-compliant
   * AbstractFramework.  This implementation of the BuilderService port is
   * specific for SCIRun2 and relies on much of the functionality in the
   * SCIRunFramework class and the various SCIRun component model classes.
   *
   */
  
  template<class Interface>
  class BuilderServiceBase : public Interface
  {
  public:
    virtual ~BuilderServiceBase();
  
    /** ? */
    PortInfo::pointer getService(const std::string& ); 

    /** Creates an instance of the component of type \em className.  The
	parameter \em instanceName is the unique name of the newly created
	instance. This method is implemented through a createComponentInstance
	call to the SCIRunFramework. */
    virtual ComponentID::pointer createInstance(const std::string& instanceName,
						const std::string& className,
						const TypeMap::pointer &properties);
    
    /** Returns a list (array) of CCA ComponentIDs that exist in the
	BuilderService's framework. */
    virtual SSIDL::array1<ComponentID::pointer> getComponentIDs();
    
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
    virtual TypeMap::pointer getComponentProperties(const ComponentID::pointer &cid);
    
    /** Associates the properties specified in \em map with an existing framework
	component \em cid.
	Null TypeMaps are not allowed. */
    virtual void setComponentProperties(const ComponentID::pointer &cid,
					const TypeMap::pointer &map);
    
    /** Returns the Component ID (opaque reference to a component instantiation)
	for the serialized component reference \em s. */
    virtual ComponentID::pointer getDeserialization(const std::string &s);

    /** Returns the Component ID (opaque reference to a component instantiation)
	for the component instance named \em componentInstanceName. */
    virtual ComponentID::pointer getComponentID(const std::string &componentInstanceName);
    
    /** Removed the component instance \em toDie from the scope of the framework.
	The \em timeout parameter gives the maximum allowable wait (in seconds)
	for the operation to be completed.  An exception is thrown if the \em
	toDie component does not  exist, or cannot be destroyed in the given \em
	timeout period. */
    virtual void destroyInstance(const ComponentID::pointer &component, float timeout);
    
    /** Returns a list of \em provides ports for the given component instance \em
	cid.*/
    virtual SSIDL::array1<std::string> getProvidedPortNames(const ComponentID::pointer &cid);
    
    /** Returns a list of \em uses ports for the component instance \em cid */
    virtual SSIDL::array1<std::string> getUsedPortNames(const ComponentID::pointer &cid);

    /** Returns a map of port properties exposed by the framework. */
    virtual TypeMap::pointer getPortProperties(const ComponentID::pointer &cid,
					       const std::string& portname);
    
    /** */
    virtual void setPortProperties(const ComponentID::pointer &cid,
				   const std::string &portname,
				   const TypeMap::pointer &map);
    
    /** */
    virtual ConnectionID::pointer connect(const ComponentID::pointer &user,
					  const std::string &usesPortName,
					  const ComponentID::pointer &provider,
					  const std::string &providesPortName);
    
    /** */
    virtual SSIDL::array1<ConnectionID::pointer> getConnectionIDs(const SSIDL::array1<ComponentID::pointer> &);
    
    
    /** Returns a CCA TypeMap that represents any properties associated
	with \em connID.
	key                    value          meaning
	user                   string         unique uses component name
	provider               string         unique provides component name
	uses port              string         uses port name
	provides port          string         provides port name
	bridge                 bool           is a bridge
    */
    virtual TypeMap::pointer getConnectionProperties(const ConnectionID::pointer &connID);
    
    /** */
    virtual void setConnectionProperties(const ConnectionID::pointer &connID,
					 const TypeMap::pointer &map);

    /** */
    virtual void disconnect(const ConnectionID::pointer &connID, float timeout);
    
    /** */
    virtual void disconnectAll(const ComponentID::pointer &id1,
			       const ComponentID::pointer &id2,
			       float timeout);

  protected:
    CoreFramework::pointer framework;
    PortInfo::pointer portInfo;

    BuilderServiceBase(const CoreFramework::pointer &framework);

  };
}

#endif
