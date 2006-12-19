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
 * ComponentWizardDialog.cc
 *
 * Written by:
 *  Ashwin Deepak Swaminathan
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  August 2006
 *
 */

#include <CCA/Components/GUIBuilder/ComponentWizardDialog.h>
#include <CCA/Components/GUIBuilder/CodePreviewDialog.h>
#include <CCA/Components/GUIBuilder/ComponentWizardHelper.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <SCIRun/TypeMap.h>
#include <Core/Util/Environment.h>

#include <wx/grid.h>
#include <wx/file.h>
#include <wx/textfile.h>
#include <wx/gbsizer.h>
#include <iostream>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(ComponentWizardDialog, wxDialog)
  EVT_BUTTON( ID_AddProvidesPort, ComponentWizardDialog::OnAddProvidesPort )
  EVT_BUTTON( ID_AddUsesPort, ComponentWizardDialog::OnAddUsesPort )
  EVT_BUTTON( ID_RemovePort, ComponentWizardDialog::OnRemovePort )
  EVT_BUTTON( ID_PreviewCode, ComponentWizardDialog::OnPreviewCode )
  EVT_BUTTON( ID_Choose, ComponentWizardDialog::OnChoose )
  EVT_SIZE  (ComponentWizardDialog::OnSize)
END_EVENT_TABLE()

ComponentWizardDialog::ComponentWizardDialog(const sci::cca::GUIBuilder::pointer& bc,
                                             wxWindow *parent,
                                             wxWindowID id,
                                             const wxString &title,
                                             const wxPoint& pos,
                                             const wxSize& size,
                                             long style,
                                             const wxString& name)
  : wxDialog(parent, id, title, pos, size, style), count_table(0), isPreviewed(false), builder(bc)
{
  const int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int leftFlags = wxALIGN_LEFT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  wxGridBagSizer *topSizer = new wxGridBagSizer();
  int i=1;
  topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Component Name")), wxGBPosition(i,1) ,wxGBSpan(1,1), leftFlags);
  componentName = new wxTextCtrl( this, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(150, wxDefaultSize.GetHeight()));
  componentName->SetToolTip(wxT("The Name of this component\nUsually begins with a capital letter\nExample: Hello, Linsolver etc."));
  topSizer->Add(componentName, wxGBPosition(1,2) ,wxGBSpan(1,2), centerFlags);
  i+=2;
  std::string compdir(getCompDirName());
  topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Location")), wxGBPosition(i,1) ,wxGBSpan(1,1), leftFlags);
  location = new wxTextCtrl( this, wxID_ANY, wxT(compdir), wxDefaultPosition, wxSize(150, wxDefaultSize.GetHeight()));
  topSizer->Add(location, wxGBPosition(i,2) ,wxGBSpan(1,2), centerFlags);
  wxBitmap bitmap(wxT("/home/collab/ashwinds/ashwin/SCIRun2/src/CCA/Components/GUIBuilder/load.xpm"), wxBITMAP_TYPE_XPM);
  topSizer->Add(new wxBitmapButton(this, ID_Choose ,bitmap,wxDefaultPosition, wxSize(30,30) ), wxGBPosition(i,4) ,wxGBSpan(1,1), centerFlags);
  i+=2;
  portInfo = new wxCheckBox(this,wxID_ANY,wxT("Do not create seperate classes for ports"));
  topSizer->Add(portInfo,wxGBPosition(i,0),wxGBSpan(1,3),centerFlags);
  i+=2;
  topSizer->Add(new wxButton(this, ID_AddProvidesPort, wxT("Add &Provides Port")),wxGBPosition(i,1) ,wxGBSpan(1,1),centerFlags);
  topSizer->Add(new wxButton(this, ID_AddUsesPort, wxT("Add &Uses Port")),wxGBPosition(i,2), wxDefaultSpan, centerFlags);
  topSizer->Add(new wxButton(this, ID_RemovePort, wxT("&Remove Port")),wxGBPosition(i,3), wxDefaultSpan, centerFlags);
  i+=2;
  listofPorts = new wxGrid(this, wxID_ANY,wxDefaultPosition ,wxSize(500,140));
  listofPorts->CreateGrid(5,4);
  for (int i=0;i<4;i++)
    listofPorts->SetColSize(i,listofPorts->GetDefaultColSize()+20);
  listofPorts->SetColLabelValue(0,"Port Class");
  listofPorts->SetColLabelValue(1,"Data Type");
  listofPorts->SetColLabelValue(2,"Port Name");
  listofPorts->SetColLabelValue(3,"Port Type");
  listofPorts->SetMargins(-10,-10);
  topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("List of Ports")), wxGBPosition(i,2) ,wxGBSpan(1,1), wxALIGN_CENTER);
  i++;
  topSizer->Add(listofPorts, wxGBPosition(i,0) ,wxGBSpan(5,5), wxEXPAND,2);
  i+=6;
  topSizer->Add( new wxButton( this, ID_PreviewCode, wxT("P&review") ), wxGBPosition(i,1) ,wxGBSpan(1,1),centerFlags);
  topSizer->Add( new wxButton( this, wxID_OK, wxT("&Generate") ), wxGBPosition(i,2) ,wxGBSpan(1,1),centerFlags);
  topSizer->Add( new wxButton( this, wxID_CANCEL, wxT("&Cancel") ), wxGBPosition(i,3) ,wxGBSpan(1,1),centerFlags);
  for (int i=0; i<15;i++)
    topSizer->AddGrowableRow(i);
  for (int i=0; i<6;i++)
    topSizer->AddGrowableCol(i);
  SetAutoLayout( TRUE );     // tell dialog to use sizer
  SetSizer( topSizer );      // actually set the sizer
  topSizer->Fit( this );           // set size to minimum size as calculated by the sizer
  topSizer->SetSizeHints( this );   // set size hints to honour mininum size
}

ComponentWizardDialog::~ComponentWizardDialog()
{
  for (unsigned int i = 0;i < pp.size(); i++) {
    delete pp[i];
  }

  for (unsigned int i = 0; i < up.size(); i++) {
    delete up[i];
  }
}

void ComponentWizardDialog::OnSize(wxSizeEvent& event)
{
  wxString val = listofPorts->GetCellValue(4,1);
  int numberofCols = listofPorts->GetNumberCols();
  int colSize =  listofPorts->GetSize().GetWidth()/(numberofCols+1);
  listofPorts->BeginBatch();
  for (int i = 0; i < numberofCols; i++) {
    listofPorts->SetColSize(i, colSize);
  }
  listofPorts->SetRowLabelSize(colSize);
  listofPorts->EndBatch();
  listofPorts->ForceRefresh();
  event.Skip();
}

void ComponentWizardDialog::Generate()
{
  isWithSidl = portInfo->GetValue();
  pp.clear();
  up.clear();
  for (int i = 0;i < count_table; i++) {
    if (listofPorts->GetCellValue(i, 3).compare("ProvidesPort") == 0) {
      pp.push_back(new PortDescriptor(listofPorts->GetCellValue(i, 0), listofPorts->GetCellValue(i, 1), listofPorts->GetCellValue(i, 2)));
    }
    if (listofPorts->GetCellValue(i, 3).compare("UsesPort") == 0) {
      up.push_back(new PortDescriptor(listofPorts->GetCellValue(i, 0), listofPorts->GetCellValue(i,1), listofPorts->GetCellValue(i, 2)));
    }
  }
  std::string compName(componentName->GetValue());
  std::string compDir(GetLocation().c_str() + compName);
  ComponentWizardHelper newHelper(builder, compName, compDir, pp, up, isWithSidl);
  newHelper.createComponent();
  return;
}

//Returns the name of the Component
wxString ComponentWizardDialog::GetText()
{
  return componentName->GetValue();
}

bool ComponentWizardDialog::Validate()
{
  //check if component Name field is empty
  if ((componentName->GetValue()).empty()) {
    wxString msg;
    msg.Printf(wxT("Component name field is Empty"));
    wxMessageBox(msg, wxT("Create Component"),
                 wxOK | wxICON_INFORMATION, this);
    return FALSE;
  }

  for (int i=0; i<listofPorts->GetNumberRows(); i++) {
    //if one cell is empty all cells must be empty
    if (listofPorts->GetCellValue(i,0).empty()) {
      if (!(listofPorts->GetCellValue(i,1).empty() &&
            listofPorts->GetCellValue(i,2).empty() &&
            listofPorts->GetCellValue(i,3).empty())) {
        wxMessageBox(wxT("Incomplete Entries"), wxT("Create Component"),
                     wxOK | wxICON_INFORMATION, this);
        return FALSE;
      }
    }

    if (!listofPorts->GetCellValue(i,0).empty()) {
      if (listofPorts->GetCellValue(i,1).empty() ||
          listofPorts->GetCellValue(i,2).empty() ||
          listofPorts->GetCellValue(i,3).empty()) {
        wxMessageBox(wxT("Incomplete Entries"), wxT("Create Component"),
                     wxOK | wxICON_INFORMATION, this);
        return FALSE;

      }
    }
  }

  //Check if another component with the same name already exists
  std::string compName(componentName->GetValue());
  std::string compDirName = GetLocation();
  Dir destDir = Dir(compDirName + ComponentSkeletonWriter::DIR_SEP + compName);
  if (destDir.exists()) {
    if (wxNO == wxMessageBox(wxT("Another component with the same name exists.\nDo you wan't to overwrite ?"), wxT("Create Component"), wxYES | wxNO | wxICON_INFORMATION | wxNO_DEFAULT, this))
      return FALSE;
  }
  return TRUE;
}

// TODO: would it make sense to put common functionality from
// OnAddProvidesPort and OnAddUsesPort in a helper funtion?


void ComponentWizardDialog::OnAddProvidesPort(wxCommandEvent& event)
{
  std::string portType("ProvidesPort");
  addPort(portType);
}
void ComponentWizardDialog::OnAddUsesPort(wxCommandEvent& event)
{
  std::string portType("UsesPort");
  addPort(portType);
}
void ComponentWizardDialog::addPort(const std::string &portType)
{
  AddPortDialog addpport(this, wxID_ANY, std::string("Add")+portType, wxPoint(10, 20), wxSize(600, 600), wxRESIZE_BORDER);
  if (addpport.ShowModal() == wxID_OK) {
    bool emptyRowFlag = true;
    // To find the number of rows - Checking for the first non empty row from the last
    for (int i=listofPorts->GetNumberRows() - 1; (i >= 0) && (emptyRowFlag == true); i--) {
      if (!(listofPorts->GetCellValue(i,0).empty() &&
            listofPorts->GetCellValue(i,1).empty() &&
            listofPorts->GetCellValue(i,2).empty() &&
            listofPorts->GetCellValue(i,3).empty())) {
        emptyRowFlag = false;
        count_table = i+1;
      }
    }
    count_table++;
    int row = count_table - 1;
    listofPorts->InsertRows(row, 1);
    listofPorts->SetCellValue(row, 0, addpport.GetPortNameText());
    listofPorts->SetCellValue(row, 1, addpport.GetDataTypeText());
    listofPorts->SetCellValue(row, 2, addpport.GetDescriptionText());
    listofPorts->SetCellValue(row, 3, portType);
  }

}
// TODO: would it make sense to put common functionality from
// OnRemoveProvidesPort and OnRemoveUsesPort in a helper funtion?

//To Remove a Port from the List of Ports Added

void ComponentWizardDialog::OnRemovePort(wxCommandEvent& event)
{
  bool emptyRowFlag = true;
  //To find the number of rows - Checking for the first non empty row from the last
  for (int i=listofPorts->GetNumberRows()-1; (i >= 0) && (emptyRowFlag == true); i--) {
    if (!(listofPorts->GetCellValue(i,0).empty() &&
          listofPorts->GetCellValue(i,1).empty() &&
          listofPorts->GetCellValue(i,2).empty() &&
          listofPorts->GetCellValue(i,3).empty())) {
      emptyRowFlag = false;
      count_table = i+1;
    }
  }
  wxArrayInt sel_rows = listofPorts->GetSelectedRows();
  for (int row_num=0;row_num<(int)sel_rows.Count();row_num++) {
    listofPorts->DeleteRows(sel_rows.Item(row_num),1);
    count_table--;
    listofPorts->InsertRows(listofPorts->GetNumberRows()-1,1);
  }
}

//To Preview the Files generated for the Component
void ComponentWizardDialog::OnPreviewCode(wxCommandEvent& event)
{
  isWithSidl = portInfo->GetValue();
  if (Validate() == FALSE)
    return;
  pp.clear();
  up.clear();

  for (int i=0;i<count_table;i++) {
    if (listofPorts->GetCellValue(i,3).compare("ProvidesPort") == 0) {
      pp.push_back(new PortDescriptor(listofPorts->GetCellValue(i,0),listofPorts->GetCellValue(i,1),listofPorts->GetCellValue(i,2)));
    }
    if (listofPorts->GetCellValue(i,3).compare("UsesPort") == 0) {
      up.push_back(new PortDescriptor(listofPorts->GetCellValue(i,0),listofPorts->GetCellValue(i,1),listofPorts->GetCellValue(i,2)));
    }
  }
  isPreviewed=true;
  ComponentWizardHelper newHelper(builder,componentName->GetValue(),GetLocation(),pp,up,isWithSidl);
  std::string sopf;
  std::string sosf;
  std::string somf;
  std::string sosidlf;
  newHelper.previewCode(sopf,sosf,somf,sosidlf);
  CodePreviewDialog codepreview (sopf, sosf, somf, sosidlf, isWithSidl, this, wxID_ANY, "Preview Generated Code", wxPoint(100, 20), wxSize(700, 500),wxRESIZE_BORDER);
  codepreview.ShowModal();
}

void ComponentWizardDialog::OnChoose(wxCommandEvent &e)
{
  std::string compDir = getCompDirName();
  wxDirDialog dialog(this, wxT("Testing directory picker"),compDir, wxDD_NEW_DIR_BUTTON);
  if (dialog.ShowModal() == wxID_OK){
    wxString path = dialog.GetPath();
    location->SetValue(path);
  }
}

std::string ComponentWizardDialog::getCompDirName()
{
  std::string srcDir(sci_getenv("SCIRUN_SRCDIR"));
  std::string compDir(srcDir + "/CCA/Components/");
  return compDir;
}

std::string ComponentWizardDialog::getTempDirName()
{
  std::string tmp(builder->getConfigDir());
  std::string home (getenv("HOME"));
  std::string tmpDirName = std::string(tmp  + ComponentSkeletonWriter::DIR_SEP + "ComponentGenerationWizard");
  return tmpDirName;
}

wxString ComponentWizardDialog::GetLocation()
{
  std::string path = location->GetValue();
  int end = path.length()-1;
  if (path.at(end) != '/') {
    path.append(std::string("/"));
  }
  return path;
}


//////////////////////////////////////////////////////////////////////////
// AddPortDialog helper class

AddPortDialog::AddPortDialog(wxWindow *parent, wxWindowID id, const wxString &title,
                             const wxPoint& pos, const wxSize& size, long style, const wxString& name)
  : wxDialog( parent, id, title, pos, size, style)
{
  wxBoxSizer *topSizer = new wxBoxSizer( wxVERTICAL );
  topSizer->AddSpacer(10);
  const int leftFlags = wxALIGN_LEFT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int rightFlags = wxALIGN_RIGHT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;

  wxBoxSizer *nameSizer = new wxBoxSizer( wxHORIZONTAL );
  lname = new wxStaticText(this, wxID_ANY, wxT("Port Class"));
  nameSizer->Add(lname, 1, leftFlags, 2);
  pname = new wxTextCtrl(this,  wxID_ANY, wxT(""), wxDefaultPosition, wxSize(150, wxDefaultSize.GetHeight()));
  pname->SetToolTip(wxT("The name of the class that this Port belongs to.Usually has the name of the component as a prefix.\nExample: HelloUIPort, WorldGoPort..etc"));
  nameSizer->Add(pname, 1, rightFlags, 2);
  topSizer->Add( nameSizer, 1, wxALIGN_CENTER, 2 );
  topSizer->AddSpacer(10);

  wxBoxSizer *datatypeSizer = new wxBoxSizer( wxHORIZONTAL );
  ldtype = new wxStaticText(this, wxID_ANY, "Datatype");
  datatypeSizer->Add(ldtype, 1, leftFlags, 2);
  dtype = new wxTextCtrl(this, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(150, wxDefaultSize.GetHeight()));
  dtype->SetToolTip(wxT("A SIDL type that derives from cca.Port.\nExample: StringPort, GoPort..etc"));
  datatypeSizer->Add(dtype, 1, rightFlags, 2);
  topSizer->Add( datatypeSizer, 1, wxALIGN_CENTER, 2 );
  topSizer->AddSpacer(10);

  wxBoxSizer *descSizer = new wxBoxSizer( wxHORIZONTAL );
  ldesc = new wxStaticText(this, wxID_ANY, wxT("Port Name"));
  descSizer->Add(ldesc, 1, rightFlags, 2);
  desc= new wxTextCtrl(this, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(150, wxDefaultSize.GetHeight()));
  desc->SetToolTip(wxT("The name of this port, which should be unique over both Uses and Provides ports. Example: ui, go, string..etc"));
  descSizer->Add(desc, 1, rightFlags, 2);
  topSizer->Add( descSizer, 1, wxALIGN_CENTER, 2 );
  topSizer->AddSpacer(30);

  wxButton *okbutton = new wxButton(this, wxID_OK, wxT("&OK"));
  wxButton *cancelbutton = new wxButton(this, wxID_CANCEL, wxT("&Cancel"));

  wxBoxSizer *okCancelSizer = new wxBoxSizer( wxHORIZONTAL );
  okCancelSizer->Add(okbutton, 1, leftFlags, 2);
  okCancelSizer->Add(cancelbutton, 1, rightFlags, 2);
  topSizer->Add( okCancelSizer, 1, wxALIGN_CENTER, 2 );
  topSizer->AddSpacer(10);

  SetAutoLayout( TRUE );     // tell dialog to use sizer
  SetSizer( topSizer );      // actually set the sizer

  topSizer->Fit( this );            // set size to minimum size as calculated by the sizer
  topSizer->SetSizeHints( this );   // set size hints to honour mininum size
}

//Returns the Port Class name
std::string AddPortDialog::GetPortNameText() const
{
  return std::string(pname->GetValue().c_str());
}

//Return the Port Type
std::string AddPortDialog::GetDataTypeText() const
{
  return std::string(dtype->GetValue().c_str());
}

//Returns the unique name for the Port
std::string AddPortDialog::GetDescriptionText() const
{
  return std::string(desc->GetValue().c_str());
}

bool AddPortDialog::Validate()
{
  if ((GetPortNameText().empty())) {
    std::cout << "\nPort name is empty";
    wxString msg;
    msg.Printf(wxT("Port name field is Empty"));

    wxMessageBox(msg, wxT("Add Provides Port"),
                 wxOK | wxICON_INFORMATION, this);
    return FALSE;
  }
  if (GetDataTypeText().empty()) {
    wxString msg;
    msg.Printf(wxT("Port type field is Empty"));

    wxMessageBox(msg, wxT("Add Uses Port"),
                 wxOK | wxICON_INFORMATION, this);

    return FALSE;
  }
  if (GetDescriptionText().empty()) {
    std::cout << "\nPort description is empty";
    wxString msg;
    msg.Printf(wxT("Port Description field is Empty"));

    wxMessageBox(msg, wxT("Add Uses Port"),
                 wxOK | wxICON_INFORMATION, this);
    return FALSE;
  }
  return TRUE;
}

}
