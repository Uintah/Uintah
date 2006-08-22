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

#ifndef CCA_Components_GUIBuilder_GUIBuilder_h
#define CCA_Components_GUIBuilder_GUIBuilder_h

#include <CCA/Components/GUIBuilder/wxSCIRunApp.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Thread/Mutex.h>
#include <SCIRun/StandAlone/sr2_version.h>

#include <wx/gdicmn.h>

#include <string>
#include <map>

namespace GUIBuilder {

class GUIBuilder : public sci::cca::GUIBuilder {
public:
  enum PortType { Uses = 0, Provides };

  GUIBuilder();
  virtual ~GUIBuilder();

  virtual void setServices(const sci::cca::Services::pointer &svc);
  virtual std::string getFrameworkURL() { return frameworkURL; }
  virtual std::string getConfigDir() { return configDir; }
  virtual void getPortInfo(const sci::cca::ComponentID::pointer& cid,
                           const std::string& portName, std::string& model, std::string& type);

  virtual void getComponentClassDescriptions(SSIDL::array1<sci::cca::ComponentClassDescription::pointer>& descArray);

  virtual sci::cca::ComponentID::pointer
  createInstance(const sci::cca::ComponentClassDescription::pointer& cd);
  virtual void destroyInstance(const sci::cca::ComponentID::pointer& cid, float timeout);
  virtual int destroyInstances(const SSIDL::array1<sci::cca::ComponentID::pointer>& cidArray, float timeout);

  virtual sci::cca::ComponentID::pointer
  generateBridge(const sci::cca::ComponentID::pointer &user,
                 const std::string& usesPortName,
                 const sci::cca::ComponentID::pointer &provider,
                 const std::string& providesPortName,
                 sci::cca::ConnectionID::pointer& connID1,
                 sci::cca::ConnectionID::pointer& connID2);

  virtual void getUsedPortNames(const sci::cca::ComponentID::pointer& cid,
                                SSIDL::array1<std::string>& nameArray);
  virtual void getProvidedPortNames(const sci::cca::ComponentID::pointer& cid,
                                    SSIDL::array1<std::string>& nameArray);

  virtual void getCompatiblePortList(const sci::cca::ComponentID::pointer& user,
                                     const std::string& usesPortName,
                                     const sci::cca::ComponentID::pointer& provider,
                                     SSIDL::array1<std::string>& portArray);
  virtual void getBridgeablePortList(const sci::cca::ComponentID::pointer &user,
                                     const std::string& usesPortName,
                                     const sci::cca::ComponentID::pointer &provider,
                                     SSIDL::array1<std::string>& portArray);


  virtual sci::cca::ConnectionID::pointer
  connect(const sci::cca::ComponentID::pointer &usesCID, const std::string &usesPortName,
          const sci::cca::ComponentID::pointer &providesCID, const ::std::string &providesPortName);
  virtual void
  disconnect(const sci::cca::ConnectionID::pointer &connID, float timeout);

  virtual void addComponentFromXML(const std::string& filePath, const std::string& componentModel);

  virtual void addFrameworkProxy(const std::string &loaderName, const std::string &user, const std::string &domain, const std::string &loaderPath);
  virtual void removeFrameworkProxy(const std::string &loaderName);

  virtual bool connectGoPort(const std::string& usesName, const std::string& providesPortName,
                             const sci::cca::ComponentID::pointer &cid, std::string& usesPortName);
  virtual void disconnectGoPort(const std::string& goPortName);
  virtual int go(const std::string& goPortName);

  virtual bool connectUIPort(const std::string& usesName, const std::string& providesPortName,
                             const sci::cca::ComponentID::pointer &cid, std::string& usesPortName);
  virtual void disconnectUIPort(const std::string& uiPortName);
  virtual int ui(const std::string& uiPortName);

  // progress
  virtual void updateProgress(const sci::cca::ComponentID::pointer& cid, int progressPercent);

  virtual bool connectComponentIcon(const std::string& usesName, const std::string& providesPortName,
                                    const sci::cca::ComponentID::pointer &cid, std::string& usesPortName);
  virtual void disconnectComponentIcon(const std::string& ciPortName);


  // events
  virtual void processEvent(const std::string& topicName, const sci::cca::Event::pointer& theEvent);


  // Note: make both setPortColor functions static when support for static functions is available

  // Use if color is in wxColorDatabase and is being used by another port, if not returns false.
  virtual bool setPortColor(const std::string& portName, const std::string& colorName);

  // Add color using RGB values if color has not been used before.  Returns false if wxWidgets
  // is unable to create the color, or if the color is beign used by another port.
  // Opaque type is wxColor.

  // see Bugzilla bug #2834:
  //virtual bool setPortColor(const std::string& portName, void* color);

  // Get stored port colour, if it doesn't exist then return a default.
  //void getPortColor(const std::string& portName, char& red, char& green, char& blue);
  void* getPortColor(const std::string& portName);

  // test ApplicationLoader
  virtual bool applicationFileExists();
  virtual void saveApplication();

  static void setApp(wxSCIRunApp& a) { app = &a; }

  static const std::string DEFAULT_SRC_DIR;
  static const std::string DEFAULT_OBJ_DIR;
  static const std::string DEFAULT_CCA_COMP_DIR;
  static const std::string GOPORT;
  static const std::string UIPORT;
  static const std::string PROGRESS_PORT;
  static const std::string COMPONENTICON_PORT;
  static const std::string APP_EXT_WILDCARD;

private:
  GUIBuilder(const GUIBuilder &);
  GUIBuilder& operator=(const GUIBuilder &);

  typedef std::map<std::string, sci::cca::ConnectionID::pointer> ConnectionMap;
  typedef std::map<std::string, wxColor> PortColorMap;

  // Note: make setDefaultPortColors static when support for static methods is available
  void setDefaultPortColors();
  bool connectPort(const std::string& usesPortName, const std::string& providesPortName,
                   const std::string& portType, const sci::cca::ComponentID::pointer &cid);
  void disconnectPort(const std::string& usesPortName);

  sci::cca::Services::pointer services;
  std::string frameworkURL;
  std::string configDir;

  // Uses port names will be unique since they are generated from unique component instance names.
  ConnectionMap connectionMap;

  // Set of port colours: the GUIBuilder will set up standard SCIRun2 ports (see SCIRun2Ports.sidl),
  // or component authors can add their own.
  // Note: make this map static when support for static functions is available
  // Note: implement using wxColorDatabase instead?
  PortColorMap portColors;

  static const std::string GUI_THREAD_NAME;
  static SCIRun::Mutex builderLock;
  static wxSCIRunApp* app;
};

}

#endif
