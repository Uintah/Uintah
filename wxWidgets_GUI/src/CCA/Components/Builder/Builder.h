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

#ifndef CCA_Components_Builder_Builder_h
#define CCA_Components_Builder_Builder_h

#include <CCA/Components/Builder/wxSCIRunApp.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Thread/Mutex.h>
#include <wx/gdicmn.h>

#include <string>
#include <map>

// check \#if wxUSE_STATUSBAR, wxUSE_MENUS, wxUSE_THREADS, wxUSE_STREAMS, wxUSE_STD_IOSTREAM...

namespace GUIBuilder {

class Builder : public sci::cca::GUIBuilder {
public:
  enum PortType { Uses = 0, Provides };

  Builder();
  virtual ~Builder();

  virtual void setServices(const sci::cca::Services::pointer &svc);
  virtual std::string getFrameworkURL() { return frameworkURL; }
  virtual void getPortInfo(const sci::cca::ComponentID::pointer& cid, const std::string& portName, std::string& model, std::string& type);

  virtual void getComponentClassDescriptions(SSIDL::array1<sci::cca::ComponentClassDescription::pointer>& descArray);

  virtual sci::cca::ComponentID::pointer createInstance(const std::string& className, const sci::cca::TypeMap::pointer& properties);
  virtual void destroyInstance(const sci::cca::ComponentID::pointer& cid, float timeout);
  virtual int destroyInstances(const SSIDL::array1<sci::cca::ComponentID::pointer>& cidArray, float timeout);

  virtual void getUsedPortNames(const sci::cca::ComponentID::pointer& cid,
				SSIDL::array1<std::string>& nameArray);
  virtual void getProvidedPortNames(const sci::cca::ComponentID::pointer& cid,
				    SSIDL::array1<std::string>& nameArray);

  virtual void getCompatiblePortList(const sci::cca::ComponentID::pointer& user,
				     const std::string& usesPortName,
				     const sci::cca::ComponentID::pointer& provider,
				     SSIDL::array1<std::string>& portArray);


  virtual sci::cca::ConnectionID::pointer
  connect(const sci::cca::ComponentID::pointer &usesCID, const std::string &usesPortName,
          const sci::cca::ComponentID::pointer &providesCID, const ::std::string &providesPortName);
  virtual void
  disconnect(const sci::cca::ConnectionID::pointer &connID, float timeout);

  virtual bool connectGoPort(const std::string& usesName, const std::string& providesPortName,
                             const sci::cca::ComponentID::pointer &cid, std::string& usesPortName);
  virtual void disconnectGoPort(const std::string& goPortName);
  virtual int go(const std::string& goPortName);

  virtual bool connectUIPort(const std::string& usesName, const std::string& providesPortName,
                             const sci::cca::ComponentID::pointer &cid, std::string& usesPortName);
  virtual void disconnectUIPort(const std::string& uiPortName);
  virtual int ui(const std::string& uiPortName);


  // Note: make both setPortColor functions static when support for static functions is available

  // Use if color is in wxColorDatabase and is being used by another port, if not returns false.
  virtual bool setPortColor(const std::string& portName, const std::string& colorName);

  // Add color using RGB values if color has not been used before.  Returns false if wxWidgets
  // is unable to create the color, or if the color is beign used by another port.
  // Opaque type is wxColor.

  // see Bugzilla bug #2834:
  //virtual bool setPortColor(const std::string& portName, void* color);

  // Get stored port colour, if it doesn't exist then return a default.
  //void Builder::getPortColor(const std::string& portName, char& red, char& green, char& blue);
  void* Builder::getPortColor(const std::string& portName);

  virtual void connectionActivity(const sci::cca::ports::ConnectionEvent::pointer &e);
  virtual void componentActivity(const sci::cca::ports::ComponentEvent::pointer &e);

  static void setApp(wxSCIRunApp& a) { app = &a; }

private:
  Builder(const Builder &);
  Builder& operator=(const Builder &);

  typedef std::map<std::string, sci::cca::ConnectionID::pointer> ConnectionMap;
  typedef std::map<std::string, wxColor> PortColorMap;

  // Note: make setDefaultPortColors static when support for static methods is available
  void setDefaultPortColors();
  bool connectPort(const std::string& providesPortName, const std::string& usesPortName,
                   const std::string& portType, const sci::cca::ComponentID::pointer &cid);
  void disconnectPort(const std::string& usesPortName);

  sci::cca::Services::pointer services;
  std::string frameworkURL;
  // Uses port names will be unique since they are generated from unique component instance names.
  ConnectionMap connectionMap;

  // Set of port colours: the Builder will set up standard SCIRun2 ports (see SCIRun2Ports.sidl),
  // or component authors can add their own.
  // Note: make this map static when support for static functions is available
  // Note: implement using wxColorDatabase instead?
  PortColorMap portColors;

  static const std::string guiThreadName;
  static SCIRun::Mutex builderLock;
  static wxSCIRunApp* app;
};

}

#endif
