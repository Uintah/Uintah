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

#ifndef Builder_h
#define Builder_h

#include <wx/app.h>
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
 #include <wx/wx.h>
#endif

#include <Core/CCA/spec/cca_sidl.h>
//#include <Core/Thread/Mutex.h>
#include <CCA/Components/Builder/BuilderWindow.h>

#include <string>
#include <map>

// check \#if wxUSE_STATUSBAR, wxUSE_MENUS, wxUSE_THREADS, wxUSE_STREAMS, wxUSE_STD_IOSTREAM...

class SCIRun::Semaphore;
class SCIRun::Mutex;

class wxColor;

typedef std::map<std::string, sci::cca::ConnectionID::pointer> ConnectionMap;
typedef std::map<std::string, wxColor> PortColorMap;

namespace GUIBuilder {

// wxApp is a singleton class, has private copy ctor, assgn. op. (see wx/app.h, wx/def.h)
class wxSCIRunApp : public wxApp {
public:
  virtual bool OnInit();
  // virtual int OnExit() { return wxApp::OnExit(); }
  void AddTopWindow(const sci::cca::BuilderComponent::pointer& bc);
  BuilderWindow* GetTopBuilderWindow() const;

  static void SetTopBuilder(const sci::cca::BuilderComponent::pointer& bc) { topBuilder = bc; }

  static void semDown() { sem.down(); }
  static void semUp() { sem.up(); }

private:
  static SCIRun::Mutex appLock;
  static SCIRun::Semaphore sem;
  // keep track of inital Builder component (instantiated from main)
  static sci::cca::BuilderComponent::pointer topBuilder;
  //static std::vector<sci::cca::BuilderComponent> activeBuilders;
};

DECLARE_APP(wxSCIRunApp)

class Builder : public sci::cca::BuilderComponent {
public:
  enum PortType { Uses = 0, Provides };

  Builder();
  virtual ~Builder();

  virtual void setServices(const sci::cca::Services::pointer &svc);
  virtual std::string getFrameworkURL() { return frameworkURL; }
  virtual void getPortInfo(const sci::cca::ComponentID::pointer& cid, const std::string& portName, std::string& model, std::string& type);

  virtual sci::cca::ComponentID::pointer createInstance(const std::string& className, const sci::cca::TypeMap::pointer& properties);
  virtual void destroyInstance(const sci::cca::ComponentID::pointer& cid, float timeout);
  virtual int destroyInstances(const SSIDL::array1<sci::cca::ComponentID::pointer>& cidArray, float timeout);

  virtual void getUsedPortNames(const sci::cca::ComponentID::pointer& cid,
				SSIDL::array1<std::string>& nameArray);
  virtual void getProvidedPortNames(const sci::cca::ComponentID::pointer& cid,
				    SSIDL::array1<std::string>& nameArray);
  virtual void getComponentClassDescriptions(SSIDL::array1<sci::cca::ComponentClassDescription::pointer>& descArray);

  virtual void getCompatiblePortList(const sci::cca::ComponentID::pointer& user,
				     const std::string& usesPortName,
				     const sci::cca::ComponentID::pointer& provider,
				     SSIDL::array1<std::string>& portArray);


  virtual sci::cca::ConnectionID::pointer
  connect(const sci::cca::ComponentID::pointer &usesCID, const std::string &usesPortName,
          const sci::cca::ComponentID::pointer &providesCID, const ::std::string &providesPortName);
  virtual void
  disconnect(const sci::cca::ConnectionID::pointer &connID, float timeout);

  virtual bool
  registerGoPort(const std::string& usesName, const sci::cca::ComponentID::pointer &cid,
                 bool isSciPort, std::string& usesPortName);
  virtual void unregisterGoPort(const std::string& goPortName);
  virtual int go(const std::string& goPortName);

  virtual void connectionActivity(const sci::cca::ports::ConnectionEvent::pointer &e);
  virtual void componentActivity(const sci::cca::ports::ComponentEvent::pointer &e);

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

  static void setApp(wxSCIRunApp& a) { app = &a; }

private:
  Builder(const Builder &);
  Builder& operator=(const Builder &);

  // Note: make setDefaultPortColors static when support for static methods is available
  void setDefaultPortColors();

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
