/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
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
 * ComponentWizardDialog.h
 *
 * Written by:
 *  Ashwin Deepak Swaminathan  
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  August 2006
 *
 */

#ifndef CCA_Components_GUIBuilder_ComponentWizardDialog_h
#define CCA_Components_GUIBuilder_ComponentWizardDialog_h

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include <SCIRun/ComponentSkeletonWriter.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>
#include <vector>

class wxWindow;
class wxButton;
class wxStaticText;
class wxTextCtrl;
class wxGrid;

namespace GUIBuilder {

class AddPortDialog;

//To generate the Component Wizard Dialog layout
class ComponentWizardDialog : public wxDialog {
public:
  enum {
    ID_AddProvidesPort = wxID_HIGHEST,
    ID_AddUsesPort,
    ID_RemovePort,
    ID_PreviewCode,
    ID_Choose,
  };

  ComponentWizardDialog(const sci::cca::GUIBuilder::pointer& bc,
                        wxWindow *parent,
                        wxWindowID id,
                        const wxString &title,
                        const wxPoint& pos = wxDefaultPosition,
                        const wxSize& size = wxDefaultSize,
                        long style = wxDEFAULT_DIALOG_STYLE,
                        const wxString& name = "ComponentWizardDialogBox");

  ~ComponentWizardDialog();
  void Generate();
  void OnAddProvidesPort( wxCommandEvent &event );
  void OnAddUsesPort( wxCommandEvent &event );
  void OnRemovePort( wxCommandEvent &event );
  void OnPreviewCode( wxCommandEvent &event );
  void OnSize(wxSizeEvent &event);
  void OnChoose( wxCommandEvent &event );
  virtual bool Validate();
  std::string getTempDirectoryName();
private:
  std::string getTempDirName();
  std::string getCompDirName();
  //Helper function to add a Provides port or Uses port
  void addPort(const std::string &portType);
  wxTextCtrl  *componentName;
  wxTextCtrl *location;
  wxGrid *listofPorts;
  wxCheckBox *portInfo;
  
  int count_table;
  bool isPreviewed;
  bool isWithSidl;
  sci::cca::GUIBuilder::pointer builder;

  wxString GetText();
  wxString GetLocation();
  //To store the list of ports added by the user
  std::vector <PortDescriptor*> pp;
  std::vector <PortDescriptor*> up;

  const static int NUM_ROWS = 5;
  const static int NUM_COLS = 4;
  const static int COL_PADDING = 20;

  DECLARE_EVENT_TABLE()
};

//To generate the AddPorts Dialog Layout
class AddPortDialog : public wxDialog {
public:
  AddPortDialog(wxWindow *parent,wxWindowID id,const wxString &title,const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE, const wxString& name = "Port dialog box");

  std::string GetPortNameText() const;
  std::string GetDataTypeText() const;
  std::string GetDescriptionText() const;
  virtual bool Validate();
  
private:
  wxTextCtrl  *pname;
  wxTextCtrl  *dtype;
  wxTextCtrl  *desc;
  wxStaticText *lname;
  wxStaticText *ldtype;
  wxStaticText *ldesc;
};

}
#endif
