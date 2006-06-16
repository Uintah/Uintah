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

#ifndef CCA_Components_GUIBuilder_ComponentWizardDialog_h
#define CCA_Components_GUIBuilder_ComponentWizardDialog_h

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#include<wx/grid.h>
#endif

#include <CCA/Components/GUIBuilder/ComponentSkeletonWriter.h>
#include <vector>

class wxWindow;
class wxButton;
class wxStaticText;
class wxTextCtrl;
class wxGrid;

namespace GUIBuilder {

class AddPortDialog;

//To generate the Component Wizard Dialog layout
class ComponentWizardDialog: public wxDialog
{
public:
  enum {
    ID_AddProvidesPort=wxID_HIGHEST,
    ID_AddUsesPort,
    ID_RemovePort,
    ID_PreviewCode,
  };

  ComponentWizardDialog(wxWindow *parent, wxWindowID id,
			const wxString &title,
			const wxPoint& pos = wxDefaultPosition,
			const wxSize& size = wxDefaultSize,
			long style = wxDEFAULT_DIALOG_STYLE,
			const wxString& name = "ComponentWizardDialogBox");

  ~ComponentWizardDialog();

  void OnOk( wxCommandEvent &event );
  void OnAddProvidesPort( wxCommandEvent &event );
  void OnAddUsesPort( wxCommandEvent &event );
  void OnRemovePort( wxCommandEvent &event );
  void OnPreviewCode( wxCommandEvent &event );
  void OnSize(wxSizeEvent &event);
  virtual bool Validate();

private:
  wxTextCtrl  *componentName;
  wxStaticText *lcomponentName;
  wxButton *AddProvidesPort;
  wxButton *AddUsesPort;
  wxButton *Preview;
  wxGrid *listofPorts;
  int count_table;
  bool isPreviewed;
  const static std::string DIR_SEP;

  wxString GetText();
  

  //To store the list of ports added by the user
  std::vector <PortDescriptor*> pp;
  std::vector <PortDescriptor*> up;

  DECLARE_EVENT_TABLE()

};

//To generate the AddPorts Dialog Layout
class AddPortDialog: public wxDialog
{
public:
  AddPortDialog(wxWindow *parent,wxWindowID id,const wxString &title,const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE, const wxString& name = "Port dialog box");

  std::string GetPortNameText() const;
  std::string GetDataTypeText() const;
  std::string GetDescriptionText() const;

private:
  wxTextCtrl  *pname;
  wxTextCtrl  *dtype;
  wxTextCtrl  *desc;
  wxStaticText *lname;
  wxStaticText *ldtype;
  wxStaticText *ldesc;
};

/* class CodePreviewDialog : public wxDialog */
/* { */
  
/*  public: */
/*   enum */
/*     { */
/*       ID_ViewSourceFileCode = wxID_HIGHEST, */
/*       ID_ViewHeaderFileCode, */
/*       ID_ViewMakeFileCode */
/*     }; */
/*   CodePreviewDialog(wxWindow *parent,wxWindowID id,const wxString &title,const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE, const wxString& name = "Code Preview Dialog Box"); */
  
/*  private: */
/*   wxTextCtrl *codePreview; */
/*   wxButton *viewSourceFileCode; */
/*   wxButton *viewHeaderFileCode; */
/*   wxButton *viewMakeFileCode; */
  
/*   void OnViewSourceFileCode( wxCommandEvent &event ); */
/*   void OnViewHeaderFileCode( wxCommandEvent &event ); */
/*   void OnViewMakeFileCode( wxCommandEvent &event ); */
/*   DECLARE_EVENT_TABLE() */
/* }; */

}
#endif
