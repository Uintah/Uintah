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
 *  BuilderWindow.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 *  Ported to wxWidgets by:
 *   Ayla Khan
 *   January 2006
 */

#ifndef CCA_Components_GUIBuilder_BuilderWindow_h
#define CCA_Components_GUIBuilder_BuilderWindow_h


#include <sci_wx.h>
#include <sci_defs/framework_defs.h>
#include <Core/CCA/spec/cca_sidl.h>

#include <string>

class wxEvtHandler;
class wxFrame;
class wxSashLayoutWindow;
class wxSashEvent;
class wxTextCtrl;
class wxMenuBar;

namespace GUIBuilder {

class BuilderWindow;
class MiniCanvas;
class NetworkCanvas;
class ComponentIcon;

class MenuTree : public wxEvtHandler {
public:
  enum { // user specified ids for widgets, menus
    ID_MENU_COMPONENTS = wxID_HIGHEST,
    ID_MENUTREE_HIGHEST = ID_MENU_COMPONENTS + 1,
  };

  MenuTree(BuilderWindow* bw, const wxString &url);
  virtual ~MenuTree();

  void add(const std::vector<wxString>& name,
           int nameindex,
           const sci::cca::ComponentClassDescription::pointer& desc,
           const wxString& fullname);
  void coalesce();
  void populateMenu(wxMenu* menu);
  void clear();
  const wxString& getURL() const { return url; }
  int getID() const { return id; }

  void OnInstantiateComponent(wxCommandEvent& event);


private:
  BuilderWindow* builderWindow;
  sci::cca::ComponentClassDescription::pointer cd;
  std::map<wxString, MenuTree*> child;
  wxString url;
  int id;
};

class BuilderWindow : public wxFrame {
public:
  //typedef std::map<std::string, int> IntMap;
  typedef std::map<wxString, MenuTree*> MenuTreeMap;
  typedef std::map<int, wxMenu*> MenuMap;
  typedef SSIDL::array1<sci::cca::ComponentClassDescription::pointer> ClassDescriptionList;

  enum { // user specified ids for widgets, menus
    ID_WINDOW_LEFT = MenuTree::ID_MENUTREE_HIGHEST,
    ID_WINDOW_RIGHT,
    ID_WINDOW_BOTTOM,
    ID_NET_WINDOW,
    ID_MINI_WINDOW,
    ID_TEXT_WINDOW,
    ID_MENU_LOAD,
    ID_MENU_INSERT,
    ID_MENU_CLEAR,
    ID_MENU_CLEAR_MESSAGES,
    //ID_MENU_EXECALL,
    ID_MENU_COMPONENT_WIZARD,
    ID_MENU_WIZARDS,
    ID_MENU_ADDINFO,
    ID_MENU_ADD_SIDLXML,
    ID_MENU_ADD_PROXY,
    ID_MENU_REMOVE_PROXY,
    ID_BUILDERWINDOW_HIGHEST = ID_MENU_REMOVE_PROXY + 1,
  };

  BuilderWindow(const sci::cca::GUIBuilder::pointer& bc, wxWindow *parent);
  virtual ~BuilderWindow();

  // two-step creation
  bool Create(wxWindow* parent, wxWindowID id,
              const wxString& title = wxString(wxT("SCIJump GUI Builder")),
              const wxPoint& pos = wxDefaultPosition,
              const wxSize& size = wxSize(WIDTH, HEIGHT),
              long style = wxDEFAULT_FRAME_STYLE,
              const wxString& name = wxString(wxT("SCIJump")));

  // set builder only if builder is null
  bool SetBuilder(const sci::cca::GUIBuilder::pointer& bc);
  void BuildAllPackageMenus();
  ComponentIcon* GetComponentIcon(const std::string& instanceName);

  // Event handlers
  void OnQuit(wxCommandEvent& event);
  void OnAbout(wxCommandEvent& event);
  void OnSize(wxSizeEvent& event);
  void OnSashDrag(wxSashEvent& event);
  void OnLoad(wxCommandEvent& event);
  void OnSave(wxCommandEvent& event);
  void OnSaveAs(wxCommandEvent& event);
  void OnClear(wxCommandEvent& event);
  void OnCompWizard(wxCommandEvent& event);
  void OnSidlXML(wxCommandEvent& event);
  void OnClearMessages(wxCommandEvent& event);
  void OnAddFrameworkProxy(wxCommandEvent& event);
  void OnRemoveFrameworkProxy(wxCommandEvent& event);

  void InstantiateComponent(const sci::cca::ComponentClassDescription::pointer& cd);

  void RedrawMiniCanvas();
  void DisplayMessage(const wxString& line);
  void DisplayErrorMessage(const wxString& line);
  void DisplayMessages(const std::vector<wxString>& lines);
  void DisplayErrorMessages(const std::vector<wxString>& lines);
  void DisplayMousePosition(const wxString& widgetName, const wxPoint& p);

  //const MenuMap& GetComponentMenus() { return menus; }

  static int GetNextID() { return ++IdCounter; }
  static int GetCurrentID() { return IdCounter; }

  static const wxColor BACKGROUND_COLOUR;

protected:
  BuilderWindow() { Init(); }
  // common initialization
  void Init();
  void SetMenus();
  void SetLayout();

  MiniCanvas* miniCanvas;
  wxTextCtrl* textCtrl;
  NetworkCanvas* networkCanvas;

  wxSashLayoutWindow* leftWindow;
  wxSashLayoutWindow* rightWindow;
  wxSashLayoutWindow* bottomWindow;

  wxMenuBar* menuBar;
  wxStatusBar* statusBar;

  //MenuTreeMap menuTrees;
  MenuMap menus;

private:
  DECLARE_DYNAMIC_CLASS(BuilderWindow)
  // This class handles events
  DECLARE_EVENT_TABLE()

  static const int MIN = 4;
  static const int WIDTH = 1000;
  static const int HEIGHT = 800;
  static const int TOP_HEIGHT = 300;
  static const int BOTTOM_HEIGHT = 500;
  static const int MINI_WIDTH = 350;
  static const int TEXT_WIDTH = 650;
  static int IdCounter;

  sci::cca::GUIBuilder::pointer builder;
  std::string url;
  wxString pointerLocationX;
  wxString pointerLocationY;


  // Component menus:
  // Need to build the menu bar and network window popup menu items
  // separately, since they will be owned by different parents.
  void buildPackageMenus(const ClassDescriptionList& list);
  void buildNetworkPackageMenus(const ClassDescriptionList& list);
  void doSaveAs();
  void setDefaultText();
};

std::vector<wxString> split_string(const wxString& str, char sep);

}

#endif
