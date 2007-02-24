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
 *  BuilderWindow.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *  Ported to wxWidgets by:
 *   Ayla Khan
 *   Scientific Computing and Imaging Institute
 *   January 2006
 *
 */

#include <wx/event.h>
#include <wx/frame.h>
#include <wx/image.h>
#include <wx/textctrl.h>
#include <wx/laywin.h>
#include <wx/string.h>

#include <vector>
#include <map>
#include <iostream>

#include <sci_metacomponents.h>
#include <CCA/Components/GUIBuilder/BuilderWindow.h>

#include <Core/Thread/Thread.h>

# include <Framework/TypeMap.h>

#include <CCA/Components/GUIBuilder/GUIBuilder.h>
#include <CCA/Components/GUIBuilder/MiniCanvas.h>
#include <CCA/Components/GUIBuilder/NetworkCanvas.h>
#include <CCA/Components/GUIBuilder/ComponentIcon.h>
#include <CCA/Components/GUIBuilder/ComponentWizardDialog.h>
#include <CCA/Components/GUIBuilder/XMLPathDialog.h>
#include <CCA/Components/GUIBuilder/FrameworkProxyDialog.h>

namespace GUIBuilder {

using namespace SCIRun;

// from Core/Containers/StringUtil:
std::vector<wxString>
split_string(const wxString& str, char sep)
{
  std::vector<wxString> result;
  wxString s(str);
  while(! s.empty()){
    unsigned long first = s.find(sep);
    if(first < s.size()){
      result.push_back(s.substr(0, first));
      s = s.substr(first+1);
    } else {
      result.push_back(s);
      break;
    }
  }
  return result;
}

MenuTree::MenuTree(BuilderWindow* bw, const wxString &url) : builderWindow(bw), url(url), id(0)
{
}

MenuTree::~MenuTree()
{
  for (std::map<wxString, MenuTree*>::iterator iter = child.begin(); iter != child.end(); iter++) {
    delete iter->second;
  }
  if (! cd.isNull()) {
    // event handler cleanup
    this->Disconnect(id,  wxEVT_COMMAND_MENU_SELECTED,
                     wxCommandEventHandler(MenuTree::OnInstantiateComponent));
    builderWindow->RemoveEventHandler(this);
  }
}

void MenuTree::add(const std::vector<wxString>& name, int nameindex,
                   const sci::cca::ComponentClassDescription::pointer& desc,
                   const wxString& fullname)
{
  if (desc->getComponentClassName() == "SCIRun.GUIBuilder"
      || desc->getComponentClassName() == "SCIRun.TxtBuilder") return;

  if (nameindex == (int) name.size()) {
    if ( !cd.isNull() ) {
      // warning - should be displayed?
      builderWindow->DisplayMessage(wxT("Duplicate component: ") + fullname);
    } else {
      cd = desc;
      id = BuilderWindow::GetNextID();
    }
  } else {
    const wxString& n = name[nameindex];
    std::map<wxString, MenuTree*>::iterator iter = child.find(n);
    if(iter == child.end()) {
      child[n] = new MenuTree(builderWindow, url);
    }
    child[n]->add(name, nameindex + 1, desc, fullname);
  }
}

// Consolidate component class names from the bottom up.
void MenuTree::coalesce()
{
  for (std::map<wxString, MenuTree*>::iterator iter = child.begin();
       iter != child.end(); iter++) {
    MenuTree* c = iter->second;
    while (c->child.size() == 1) {
      std::map<wxString, MenuTree*>::iterator grandchild = c->child.begin();
      wxString newname = iter->first + wxT(".") + grandchild->first;

      MenuTree* gc = grandchild->second;
      c->child.clear(); // So that grandchild won't get deleted...
      delete c;

      child.erase(iter);
      child[newname] = gc;
      iter = child.begin();
      c = gc;
    }
    c->coalesce();
  }
}

void MenuTree::populateMenu(wxMenu* menu)
{
  for (std::map<wxString, MenuTree*>::iterator iter = child.begin();
       iter != child.end(); iter++) {
//     if (iter->first == "GUIBuilder" || iter->first == "TxtBuilder") {
//       child.erase(iter);
//       continue;
//     }
    if (iter->second->cd.isNull()) {
      wxMenu* submenu = new wxMenu(wxEmptyString, wxMENU_TEAROFF);
      //submenu->setFont(builderWindow->font());
      iter->second->populateMenu(submenu);
      menu->Append(ID_MENU_COMPONENTS, iter->first, submenu);
    } else {
      builderWindow->PushEventHandler(iter->second);
      menu->Append(iter->second->id, iter->first, iter->first);
      iter->second->Connect(iter->second->id, wxEVT_COMMAND_MENU_SELECTED,
                            wxCommandEventHandler(MenuTree::OnInstantiateComponent));
    }
  }
}

void MenuTree::clear()
{
  child.clear();
}

void MenuTree::OnInstantiateComponent(wxCommandEvent& event)
{
  // this shouldn't happen
  if (cd.isNull()) {
    // error should be logged
    builderWindow->DisplayMessage(wxT("Error: null component description!"));
    return;
  }
  builderWindow->InstantiateComponent(cd);
}



const wxColor BuilderWindow::BACKGROUND_COLOUR(0, 51, 102);

// Event table
BEGIN_EVENT_TABLE(BuilderWindow, wxFrame)
  EVT_MENU(wxID_ABOUT, BuilderWindow::OnAbout)
  EVT_MENU(wxID_EXIT, BuilderWindow::OnQuit)
  EVT_MENU(ID_MENU_LOAD, BuilderWindow::OnLoad)
  EVT_MENU(wxID_SAVE, BuilderWindow::OnSave)
  EVT_MENU(wxID_SAVEAS, BuilderWindow::OnSaveAs)
  EVT_MENU(ID_MENU_CLEAR, BuilderWindow::OnClear)
  EVT_MENU(ID_MENU_CLEAR_MESSAGES, BuilderWindow::OnClearMessages)
  EVT_MENU(ID_MENU_COMPONENT_WIZARD, BuilderWindow::OnCompWizard)
  EVT_MENU(ID_MENU_ADD_SIDLXML, BuilderWindow::OnSidlXML)
  EVT_MENU(ID_MENU_ADD_PROXY, BuilderWindow::OnAddFrameworkProxy)
  EVT_SIZE(BuilderWindow::OnSize)
  EVT_SASH_DRAGGED_RANGE(ID_WINDOW_LEFT, ID_WINDOW_BOTTOM, BuilderWindow::OnSashDrag)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(BuilderWindow, wxFrame)

int BuilderWindow::IdCounter = BuilderWindow::ID_BUILDERWINDOW_HIGHEST;

BuilderWindow::BuilderWindow(const sci::cca::GUIBuilder::pointer& bc, wxWindow *parent) : builder(bc), pointerLocationX("0"), pointerLocationY("0")
{
#if FWK_DEBUG
  std::cerr << "BuilderWindow::BuilderWindow(..): from thread " << Thread::self()->getThreadName() << " in framework " << builder->getFrameworkURL() << std::endl;
#endif

  Init();
  Create(parent, wxID_ANY);
}

bool BuilderWindow::Create(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
{
  if (!wxFrame::Create(parent, id, title, pos, size, style, name)) {
    // error message?
    return false;
  }
  url = builder->getFrameworkURL();
  SetLayout();
  SetMenus();
  setDefaultText();

  //SetFont(wxFont(11, wxDEFAULT, wxNORMAL, wxNORMAL, 0, wxT("Sans")));
#if FWK_DEBUG
  statusBar = CreateStatusBar(3, wxST_SIZEGRIP);
  int statusBarWidths[] = { 350, 150, -1 };
  statusBar->SetStatusWidths(3, statusBarWidths);
#else
  statusBar = CreateStatusBar(2, wxST_SIZEGRIP);
  int statusBarWidths[] = { 350, -1 };
  statusBar->SetStatusWidths(2, statusBarWidths);
#endif
  statusBar->SetStatusText(wxT("SCIJump started"), 0);
  return true;
}


bool BuilderWindow::SetBuilder(const sci::cca::GUIBuilder::pointer& bc)
{
  if (builder.isNull()) {
    builder = bc;
    return true;
  }

  return false;
}

void BuilderWindow::BuildAllPackageMenus() {
    ClassDescriptionList list;
    builder->getComponentClassDescriptions(list);
    buildPackageMenus(list);
    buildNetworkPackageMenus(list);
}

BuilderWindow::~BuilderWindow()
{
#if FWK_DEBUG
  std::cerr << "BuilderWindow::~BuilderWindow()" << std::endl;
#endif
  // framework shutdown instead!!!
  Thread::exitAll(0);
}

///////////////////////////////////////////////////////////////////////////
// manage child windows

void BuilderWindow::RedrawMiniCanvas()
{
  miniCanvas->Refresh();
}

ComponentIcon* BuilderWindow::GetComponentIcon(const std::string& instanceName)
{
  return networkCanvas->GetIcon(instanceName);
}

void BuilderWindow::DisplayMessage(const wxString& line)
{
  // Used to (temporarily - local scope) redirect all output sent to a C++ ostream object to a wxTextCtrl.
  //wxStreamToTextRedirector redirect(textCtrl);
  *textCtrl << line << wxT("\n");
}

void BuilderWindow::DisplayErrorMessage(const wxString& line)
{
  textCtrl->SetDefaultStyle(wxTextAttr(*wxRED));
  // Used to (temporarily - local scope) redirect all output sent to a C++ ostream object to a wxTextCtrl.
  //wxStreamToTextRedirector redirect(textCtrl);
  *textCtrl << line << wxT("\n");
  textCtrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
}

void BuilderWindow::DisplayMessages(const std::vector<wxString>& lines)
{
  //wxStreamToTextRedirector redirect(textCtrl);

  for (std::vector<wxString>::const_iterator iter = lines.begin(); iter != lines.end(); iter++) {
    *textCtrl << *iter << wxT("\n");
  }
}


void BuilderWindow::DisplayErrorMessages(const std::vector<wxString>& lines)
{
  textCtrl->SetDefaultStyle(wxTextAttr(*wxRED));
  DisplayMessages(lines);
  textCtrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
}

#if FWK_DEBUG
// TODO: should only be available in debug mode?
void BuilderWindow::DisplayMousePosition(const wxString& widgetName, const wxPoint& p)
{
  pointerLocationX.Printf("%d", p.x);
  pointerLocationY.Printf("%d", p.y);
  statusBar->SetStatusText(widgetName + ": " + pointerLocationX + ", " + pointerLocationY, 1);
}
#endif

///////////////////////////////////////////////////////////////////////////
// event handlers

void BuilderWindow::OnAbout(wxCommandEvent &event)
{
  wxString msg;
  msg.Printf(wxT("Copyright (c) 2006 Scientific Computing and Imaging Institute, University of Utah.\n\nSCIJump information is available at\nhttps://code.sci.utah.edu/SCIJump/index.php/Main_Page."));

  // show license
  wxMessageBox(msg, wxT("About SCIJump"), wxOK|wxICON_INFORMATION, this);
}

void BuilderWindow::OnQuit(wxCommandEvent &event)
{
#if FWK_DEBUG
  std::cerr << "BuilderWindow::OnQuit(..)" << std::endl;
#endif
  // Destroy the frame
  Close();
}

void BuilderWindow::OnSashDrag(wxSashEvent& event)
{
#if 0
// #if FWK_DEBUG
//   std::cerr << "BuilderWindow::OnSashDrag(..): event drag status="  << event.GetDragStatus() << std::endl;
//   std::cerr << "Drag rect = (" << event.GetDragRect().x << ", " << event.GetDragRect().y << ", " << event.GetDragRect().width << ", " << event.GetDragRect().height << ")" << std::endl;
// #endif
#endif
  if (event.GetDragStatus() == wxSASH_STATUS_OUT_OF_RANGE) {
    return;
  }

  switch (event.GetId()) {
    // If both the left and right windows attempt to handle dragging events,
    // the delivery of events can be error-prone, resulting in resizing
    // errors.
  case ID_WINDOW_LEFT:
    leftWindow->SetDefaultSize(wxSize(event.GetDragRect().width, MIN));
    //rightWindow->SetMinimumSizeX(MIN);
    break;
  case ID_WINDOW_BOTTOM:
    bottomWindow->SetDefaultSize(wxSize(MIN, event.GetDragRect().height));
    leftWindow->SetMinimumSizeY(MIN);
    rightWindow->SetMinimumSizeY(MIN);
    break;
  }

  wxLayoutAlgorithm layout;
  layout.LayoutFrame(this);

  // Leaves bits of itself behind sometimes
  Refresh();
}

void BuilderWindow::OnSize(wxSizeEvent& WXUNUSED(event))
{
  // recalc. window sizes so that bottom window gets more
  wxLayoutAlgorithm layout;
  layout.LayoutFrame(this);
  Refresh();
}

// network file handling will have to be moved outside of GUI classes
void BuilderWindow::OnLoad(wxCommandEvent& event)
{
  // need to save current app if user agrees and clear
  wxBusyCursor wait;
  wxString wildcard = STLTowxString(GUIBuilder::APP_EXT_WILDCARD);
  wxFileDialog fDialog(this,
                       wxT("Open application file"),
                       STLTowxString(GUIBuilder::DEFAULT_OBJ_DIR),
                       wxEmptyString,
                       wildcard,
                       wxOPEN|wxFILE_MUST_EXIST|wxCHANGE_DIR);
  statusBar->SetStatusText(wxT("Loading application file"), 0);

  if (fDialog.ShowModal() == wxID_OK) {
    wxString path = fDialog.GetPath();
    SSIDL::array1<sci::cca::ComponentID::pointer> cidList;
    SSIDL::array1<sci::cca::ConnectionID::pointer> connList;

    //Read XML and loads components and connections using BuilderService:
    builder->loadApplication(wxToSTLString(path), cidList, connList);

    //Graphical parts of loading:
    for (SSIDL::array1<sci::cca::ComponentID::pointer>::iterator cidIter = cidList.begin();
         cidIter != cidList.end(); cidIter++) {

      if (! (*cidIter).isNull()) {
        networkCanvas->AddIcon(*cidIter);
      }
    }

    for (SSIDL::array1<sci::cca::ConnectionID::pointer>::iterator connIDIter = connList.begin();
         connIDIter != connList.end(); connIDIter++) {

      if (! (*connIDIter).isNull()) {
        networkCanvas->Connect(*connIDIter);
      }

    }

    statusBar->SetStatusText(wxT("Application file loaded"), 0);
  } else {
    statusBar->SetStatusText(wxEmptyString, 0);
  }
}

void BuilderWindow::OnSave(wxCommandEvent& event)
{
  wxBusyCursor wait;
  bool exists = builder->applicationFileExists();
  if (exists) {
    // get file name?
    int answer = wxMessageBox(wxT("Overwrite current application file?"), wxT("Confirm"),
                              wxYES_NO|wxICON_QUESTION, this);
    if (answer == wxYES) {
      builder->saveApplication();
    }
  } else {
    doSaveAs();
  }
}

void BuilderWindow::OnSaveAs(wxCommandEvent& WXUNUSED(event))
{
  doSaveAs();
}

void BuilderWindow::OnCompWizard(wxCommandEvent& event)
{
  ComponentWizardDialog cwDialog(builder,
                                 this,
                                 wxID_ANY,
                                 wxT("Component wizard dialog"),
                                 wxPoint(100, 100),
                                 wxSize(600, 800),
                                 wxRESIZE_BORDER|wxCAPTION|wxSYSTEM_MENU);
  if (cwDialog.ShowModal() == wxID_OK){
    cwDialog.Generate();
  }
}

void BuilderWindow::OnSidlXML(wxCommandEvent& event)
{
  XMLPathDialog pathDialog(this, wxID_ANY);
  if (pathDialog.ShowModal() == wxID_OK) {
    builder->addComponentFromXML( wxToSTLString(pathDialog.GetFilePath() ),
                                  wxToSTLString( pathDialog.GetComponentModel() ) );
  }
}

void BuilderWindow::OnClearMessages(wxCommandEvent& event)
{
  // need option to not ask for confirmation -> save to config file
  wxMessageDialog mdialog(this, wxT("Clear all messages?"), wxT("Clear Messages Confirmation"));

  if (mdialog.ShowModal() == wxID_OK) {
    textCtrl->Clear();
  }
  setDefaultText();
}

void BuilderWindow::OnAddFrameworkProxy(wxCommandEvent& event)
{
  FrameworkProxyDialog fpDialog(this);
  if (fpDialog.ShowModal() == wxID_OK) {
    builder->addFrameworkProxy(wxToSTLString( fpDialog.GetLoader() ),
                               wxToSTLString( fpDialog.GetLogin()) ,
                               wxToSTLString( fpDialog.GetDomain() ),
                               wxToSTLString( fpDialog.GetPath() ) );
  }
}

void BuilderWindow::OnRemoveFrameworkProxy(wxCommandEvent& event)
{
  // not implemented yet
}

void BuilderWindow::InstantiateComponent(const sci::cca::ComponentClassDescription::pointer& cd)
{
  wxBusyCursor wait;
  statusBar->SetStatusText(wxT("Build component"), 0);

  sci::cca::ComponentID::pointer cid = builder->createInstance(cd);
  // Assumes that the GUI builder component class will be named "SCIRun.GUIBuilder".
  // Is there a better way to check if this is a GUI builder?
  if (! cid.isNull() && cd->getComponentClassName() != "SCIRun.GUIBuilder") {
#if FWK_DEBUG
    std::cerr << "wx: Got " << cid->getInstanceName() << std::endl;
#endif
    networkCanvas->AddIcon(cid);
  }
  statusBar->SetStatusText(wxT("Component built"), 0);
}

void BuilderWindow::OnClear(wxCommandEvent& WXUNUSED(event))
{
  networkCanvas->Clear();
  RedrawMiniCanvas();
}


///////////////////////////////////////////////////////////////////////////
// protected member functions

void BuilderWindow::Init()
{
  //SetFont(wxFont(11, wxDEFAULT, wxNORMAL, wxNORMAL, 0, wxT("Sans")));
  //SetToolTip(wxT("\"Test tooltip\""));
}

// constructor helpers

void BuilderWindow::SetMenus()
{
  // The "About" item should be in the help menu
  wxMenu *helpMenu = new wxMenu();
  helpMenu->Append(wxID_ABOUT, wxT("&About..."), wxT("Show about dialog"));

  wxMenu* compWizardMenu = new wxMenu(wxEmptyString, wxMENU_TEAROFF);
  compWizardMenu->Append(ID_MENU_COMPONENT_WIZARD, wxT("Component Wizard"), wxT("Create component skeleton"));

  wxMenu* fileMenu = new wxMenu();
  fileMenu->Append(ID_MENU_LOAD, wxT("&Load\tAlt-L"), wxT("Load saved application from file"));
  //fileMenu->Append(ID_MENU_INSERT, wxT("&Insert\tAlt-L"), wxT("Insert components from file"));
  fileMenu->Append(wxID_SAVE, wxT("&Save\tAlt-S"), wxT("Save application to file"));
  fileMenu->Append(wxID_SAVEAS, wxT("&Save As\tAlt-S"), wxT("Save application to file"));
  fileMenu->AppendSeparator();

  fileMenu->Append(ID_MENU_CLEAR, wxT("&Clear\tAlt-C"), wxT("Clear All"));

  fileMenu->Append(ID_MENU_CLEAR_MESSAGES, wxT("&Clear Messages"), wxT("Clear Text Messages"));
  //fileMenu->Append(wxID_SELECTALL, wxT("Select &All\tCtrl-A"), wxT("Select All"));
  //fileMenu->Append(ID_MENU_EXECALL, wxT("&Execute All\tCtrl-A"), wxT("Execute All"));
  fileMenu->AppendSeparator();
  fileMenu->Append(ID_MENU_WIZARDS, wxT("&Wizards\tAlt-W"), compWizardMenu);
  fileMenu->AppendSeparator();
  // need ability to add information to component applications
  //fileMenu->Append(ID_MENU_ADDINFO, wxT("&Add Info\tAlt-A"), wxT("Add information to application"));
  //fileMenu->AppendSeparator();
  fileMenu->Append(ID_MENU_ADD_SIDLXML, wxT("&Add Components from XML\tAlt-A"), wxT("Add a new component XML description file."));
  fileMenu->AppendSeparator();
  fileMenu->Append(wxID_EXIT, wxT("E&xit\tAlt-X"), wxT("Quit this program"));

#if 0
  // Disabled until xterm crashing problem is fixed.
//   wxMenu* proxyFwkMenu = new wxMenu();
//   proxyFwkMenu->Append(ID_MENU_ADD_PROXY, wxT("Add Proxy Framework"), wxT("Instantiate a new proxy framework to the master framework"));
//   //proxyFwkMenu->Append(ID_MENU_REMOVE_PROXY, wxT("Remove Proxy Framework"), wxT("Remove a new proxy framework from the master framework"));
#endif

  menuBar = new wxMenuBar();
  menuBar->Append(fileMenu, wxT("&File"));
#if 0
//   menuBar->Append(proxyFwkMenu, wxT("&Proxy Frameworks"));
#endif

  BuildAllPackageMenus();

  menuBar->Append(helpMenu, wxT("&Help"));
  SetMenuBar(menuBar);
}

void BuilderWindow::SetLayout()
{
  bottomWindow = new wxSashLayoutWindow(this, ID_WINDOW_BOTTOM, wxPoint(0, TOP_HEIGHT),
                                        wxSize(WIDTH, BOTTOM_HEIGHT), wxNO_BORDER|wxSW_3D|wxCLIP_CHILDREN);
  bottomWindow->SetDefaultSize(wxSize(MIN, BOTTOM_HEIGHT));
  bottomWindow->SetOrientation(wxLAYOUT_HORIZONTAL);
  bottomWindow->SetAlignment(wxLAYOUT_BOTTOM);
  bottomWindow->SetSashVisible(wxSASH_TOP, true);

  networkCanvas = new NetworkCanvas(builder, this, bottomWindow, ID_NET_WINDOW,
                                    wxPoint(0, TOP_HEIGHT), wxSize(WIDTH, BOTTOM_HEIGHT));

  // use wxCLIP_CHILDREN to eliminate flicker on Windows
  // A window to the left of the client window
  leftWindow = new wxSashLayoutWindow(this, ID_WINDOW_LEFT, wxPoint(0, 0),
                                      wxSize(MINI_WIDTH, TOP_HEIGHT), wxNO_BORDER|wxSW_3D|wxCLIP_CHILDREN);
  leftWindow->SetDefaultSize(wxSize(MINI_WIDTH, MIN));
  leftWindow->SetOrientation(wxLAYOUT_VERTICAL);
  leftWindow->SetAlignment(wxLAYOUT_LEFT);
  leftWindow->SetBackgroundColour(BACKGROUND_COLOUR);
  leftWindow->SetSashVisible(wxSASH_RIGHT, true);

  // add mini-canvas (scrolled window) to leftWindow
  miniCanvas = new MiniCanvas(leftWindow, this, networkCanvas, ID_MINI_WINDOW,
                              wxPoint(0, 0), wxSize(MINI_WIDTH, TOP_HEIGHT));

  // A window to the left of the client window
  rightWindow = new wxSashLayoutWindow(this, ID_WINDOW_RIGHT, wxPoint(MINI_WIDTH, 0),
                                       wxSize(TEXT_WIDTH, TOP_HEIGHT), wxNO_BORDER|wxSW_3D|wxCLIP_CHILDREN);
  rightWindow->SetDefaultSize(wxSize(TEXT_WIDTH, MIN));
  rightWindow->SetOrientation(wxLAYOUT_VERTICAL);
  rightWindow->SetAlignment(wxLAYOUT_LEFT);
  rightWindow->SetSashVisible(wxSASH_LEFT, false); // resizing in terms of leftWindow only

  textCtrl = new wxTextCtrl(rightWindow, ID_TEXT_WINDOW, wxEmptyString, wxPoint(MINI_WIDTH, 0),
                            wxSize(TEXT_WIDTH, TOP_HEIGHT),
                            wxSUNKEN_BORDER|wxTE_MULTILINE|wxTE_READONLY|wxTE_AUTO_URL|wxTE_CHARWRAP);
}

///////////////////////////////////////////////////////////////////////////
// private member functions

void BuilderWindow::buildPackageMenus(const ClassDescriptionList& list)
{
  wxBusyCursor wait;
  MenuTreeMap menuTrees;

  // build menu trees for component model and component types in model
  for (ClassDescriptionList::const_iterator iter = list.begin(); iter != list.end(); iter++) {
    // model name could be obtained somehow locally.
    // and we can assume that the remote component model is always "CCA"
    wxString model = STLTowxString((*iter)->getComponentModelName());
    wxString loaderName = STLTowxString((*iter)->getLoaderName());
    wxString name = STLTowxString((*iter)->getComponentClassName());

    // component class has a loader that is not in this address space?
    if (! loaderName.empty()) {
      size_t i = name.find_first_of('.');
      name.insert(i, wxT("@") + loaderName);
    }
    if (menuTrees.find(model) == menuTrees.end()) {
      menuTrees[model] = new MenuTree(this, STLTowxString(url));
    }
    std::vector<wxString> splitname = split_string(name, '.');
    menuTrees[model]->add(splitname, 0, *iter, name);
  }

  for (MenuTreeMap::iterator iter = menuTrees.begin();
       iter != menuTrees.end(); iter++) {
    iter->second->coalesce();
  }

  // build menus from menu trees
  for (MenuTreeMap::iterator iter = menuTrees.begin(); iter != menuTrees.end(); iter++) {
    wxMenu *menu = new wxMenu(wxEmptyString, wxMENU_TEAROFF);
    iter->second->populateMenu(menu);

    // must be tested after adding components at runtime
    int menuIndex = menuBar->FindMenu(iter->first);
    if (menuIndex == wxNOT_FOUND) {
      if (menuBar->Append(menu, iter->first)) {
        menus[menuIndex] = menu;
      } else {
        DisplayErrorMessage(wxT("Could not append menu ") + iter->first);
      }
    } else {
      menus[menuIndex] = menu;
      wxMenu* oldMenu = menuBar->Replace(menuIndex, menu, iter->first);
      delete oldMenu;
    }
  }
}

void BuilderWindow::buildNetworkPackageMenus(const ClassDescriptionList& list)
{
  wxBusyCursor wait;
  if (networkCanvas == 0) {
    DisplayErrorMessage(wxT("Cannot build network canvas menus: network canvas does not exist."));
    return;
  }

  MenuTreeMap menuTrees;
  // build menu trees for component model and component types in model
  for (ClassDescriptionList::const_iterator iter = list.begin(); iter != list.end(); iter++) {
    // model name could be obtained somehow locally.
    // and we can assume that the remote component model is always "CCA"
    wxString model = STLTowxString((*iter)->getComponentModelName());
    wxString loaderName = STLTowxString((*iter)->getLoaderName());
    wxString name = STLTowxString((*iter)->getComponentClassName());

    // component class has a loader that is not in this address space?
    if (! loaderName.empty()) {
      size_t i = name.find_first_of('.');
      name.insert(i, wxT("@") + loaderName);
    }
    if (menuTrees.find(model) == menuTrees.end()) {
      menuTrees[model] = new MenuTree(this, STLTowxString(url));
    }
    std::vector<wxString> splitname = split_string(name, '.');
    menuTrees[model]->add(splitname, 0, *iter, name);
  }

  for (MenuTreeMap::iterator iter = menuTrees.begin();
       iter != menuTrees.end(); iter++) {
    iter->second->coalesce();
  }

  // build menus from menu trees
  for (MenuTreeMap::iterator iter = menuTrees.begin(); iter != menuTrees.end(); iter++) {
    wxMenu *menu = new wxMenu(wxEmptyString, wxMENU_TEAROFF);
    iter->second->populateMenu(menu);

    // must be tested after adding components at runtime
    int menuID = networkCanvas->popupMenu->FindItem(iter->first);
    if (menuID == wxNOT_FOUND) {
      networkCanvas->popupMenu->Append(wxID_ANY, iter->first, menu);
    } else {
      networkCanvas->popupMenu->Destroy(menuID);
      networkCanvas->popupMenu->Append(menuID, iter->first, menu);
    }
  }
}

void BuilderWindow::setDefaultText()
{
  std::vector<wxString> v;
  wxString ver = wxT("SCIJump v") + STLTowxString(SCIJUMP_VERSION);
  wxString u = wxT("Framework URL: ") + STLTowxString(url.c_str());
  v.push_back(ver);
  v.push_back(u);
  v.push_back(wxT("--------------------\n"));
  DisplayMessages(v);
}

void BuilderWindow::doSaveAs()
{
  wxBusyCursor wait;
  wxString wildcard = STLTowxString(GUIBuilder::APP_EXT_WILDCARD);
  wxString extension = STLTowxString(GUIBuilder::APP_EXT);
  wxFileDialog fDialog(this,
                       wxT("Save application file"),
                       STLTowxString(GUIBuilder::DEFAULT_OBJ_DIR),
                       wxEmptyString,
                       wildcard,
                       wxSAVE|wxOVERWRITE_PROMPT|wxCHANGE_DIR);

  statusBar->SetStatusText(wxT("Saving application file"), 0);
  if(fDialog.ShowModal() == wxID_OK) {
    wxString path = fDialog.GetPath();
    wxString name = fDialog.GetFilename();
    if(name.Find('.') == -1) {
      path.Append('.');
      path.Append(extension);
    }
    builder->saveApplication(wxToSTLString(path));
    statusBar->SetStatusText(wxT("Application file saved"), 0);
  } else {
    statusBar->SetStatusText(wxEmptyString, 0);
  }
}

}
