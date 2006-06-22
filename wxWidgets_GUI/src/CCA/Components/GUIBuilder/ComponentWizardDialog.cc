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
 * ComponentWizardDialog.cc
 *
 * Written by:
 *  <author>
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  <date>
 *
 */

#include <CCA/Components/GUIBuilder/ComponentWizardDialog.h>
#include <CCA/Components/GUIBuilder/CodePreviewDialog.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <SCIRun/TypeMap.h>

#include <wx/grid.h>
#include <wx/file.h>
#include <wx/textfile.h>

#include <iostream>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(ComponentWizardDialog, wxDialog)
  EVT_BUTTON( wxID_OK, ComponentWizardDialog::OnOk )
  EVT_BUTTON( ID_AddProvidesPort, ComponentWizardDialog::OnAddProvidesPort )
  EVT_BUTTON( ID_AddUsesPort, ComponentWizardDialog::OnAddUsesPort )
  EVT_BUTTON( ID_RemovePort, ComponentWizardDialog::OnRemovePort )
  EVT_BUTTON( ID_PreviewCode, ComponentWizardDialog::OnPreviewCode )
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
  wxFlexGridSizer *topSizer = new wxFlexGridSizer(4, 1, 2, 2);

  topSizer->AddSpacer(20);
  wxBoxSizer *componentSizer = new wxBoxSizer( wxHORIZONTAL );
  const int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int leftFlags = wxALIGN_LEFT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int rightFlags = wxALIGN_RIGHT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;

  componentSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Component Name")), 1, centerFlags, 2);
  componentName = new wxTextCtrl( this, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(150, wxDefaultSize.GetHeight()));
  componentName->SetToolTip(wxT("The Name of this component\nUsually begins with a capital letter\nExample: Hello, Linsolver etc."));
  componentSizer->Add(componentName, 1, rightFlags, 2);
  topSizer->Add( componentSizer, 1, wxALIGN_CENTER, 0 );
  topSizer->AddSpacer(15);

  wxBoxSizer *portsSizer = new wxBoxSizer( wxHORIZONTAL );
  portsSizer->Add(new wxButton(this, ID_AddProvidesPort, wxT("Add &Provides Port")), 1, leftFlags, 4);
  portsSizer->Add(new wxButton(this, ID_AddUsesPort, wxT("Add &Uses Port")), 1, rightFlags, 4);
  portsSizer->Add(new wxButton(this, ID_RemovePort, wxT("&Remove Port")), 1, rightFlags, 4);
  topSizer->Add( portsSizer, 1, wxALIGN_CENTER, 0 );
  topSizer->AddSpacer(20);

  wxBoxSizer *gridSizer = new wxBoxSizer(wxVERTICAL);
  listofPorts = new wxGrid(this, wxID_ANY, wxDefaultPosition, wxSize(500, 140));
  listofPorts->CreateGrid(NUM_ROWS, NUM_COLS);
  for (int i = 0; i < NUM_COLS; i++) {
    listofPorts->SetColSize(i, listofPorts->GetDefaultColSize() + COL_PADDING);
  }
  listofPorts->SetColLabelValue(0, "Port Class");
  listofPorts->SetColLabelValue(1, "Data Type");
  listofPorts->SetColLabelValue(2, "Port Name");
  listofPorts->SetColLabelValue(3, "Port Type");
  listofPorts->SetMargins(-10, -10);

  gridSizer->Add(new wxStaticText(this, wxID_ANY, wxT("List of Ports")), 0, centerFlags, 2);
  gridSizer->AddSpacer(10);
  gridSizer->Add(listofPorts, 1, wxEXPAND, 0);
  //topSizer->Add( gridSizer, 0, wxALIGN_CENTER, 0 );
  topSizer->Add( gridSizer, 1, wxEXPAND, 0 );
  topSizer->AddSpacer(20);


  wxBoxSizer *okCancelSizer = new wxBoxSizer( wxHORIZONTAL );
  okCancelSizer->Add( new wxButton( this, ID_PreviewCode, wxT("P&review") ), 1, leftFlags, 4 );
  okCancelSizer->Add( new wxButton( this, wxID_OK, wxT("&Generate") ), 1, leftFlags, 4 );
  okCancelSizer->Add( new wxButton( this, wxID_CANCEL, wxT("&Cancel") ), 1, rightFlags, 4 );
  topSizer->Add( okCancelSizer, 1, wxALIGN_CENTER, 0 );
  topSizer->AddSpacer(5);


  topSizer->AddGrowableRow(0);
  topSizer->AddGrowableRow(1);
  topSizer->AddGrowableRow(2);
  topSizer->AddGrowableRow(3);
  topSizer->AddGrowableRow(4);
  topSizer->AddGrowableCol(0);
  topSizer->SetFlexibleDirection(wxBOTH);
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

  std::string tmpDir(builder->getConfigDir() + ComponentSkeletonWriter::DIR_SEP + "ComponentGenerationWizard");
  Dir d1(tmpDir);
  if (d1.exists()) {
    d1.remove(std::string("tempheader.txt"));
    d1.remove(std::string("tempsource.txt"));
    d1.remove(std::string("tempsubmake.txt"));
    d1.remove();
  }
}

void ComponentWizardDialog::OnSize(wxSizeEvent& event)
{
  //  int numberofRows = listofPorts->GetNumberRows();
  int numberofCols = listofPorts->GetNumberCols();
  int colSize =  listofPorts->GetSize().GetWidth()/(numberofCols+1);
  //  int rowSize =  listofPorts->GetSize().GetHeight()/(numberofRows+1);
  listofPorts->BeginBatch();
  for (int i = 0; i < numberofCols; i++) {
    listofPorts->SetColSize(i, colSize);
  }
  listofPorts->SetRowLabelSize(colSize);
  listofPorts->EndBatch();
  //    if (int i=0;i<numberofRows;i++)
  //      listofPorts->SetRowSize(i, rowSize);
  //   listofPorts->SetColLabelSize(rowSize);
  listofPorts->ForceRefresh();
  event.Skip();
}


void ComponentWizardDialog::OnOk(wxCommandEvent& event)
{
  if ((componentName->GetValue()).empty()) {
#if DEBUG
    std::cout<<"\nComponent Name is Empty\n";
#endif
    wxString msg;
    msg.Printf(wxT("Component name field is Empty"));

    wxMessageBox(msg, wxT("Create Component"),
                 wxOK | wxICON_INFORMATION, this);
    return;
  }

  // TODO: allow user to set component directory

  std::string compName(componentName->GetValue());
  std::string compsDir(GUIBuilder::DEFAULT_CCA_COMP_DIR + ComponentSkeletonWriter::DIR_SEP + compName);

  pp.clear();
  up.clear();

  for (int i = 0; i < count_table; i++) {
    if (listofPorts->GetCellValue(i, 3).compare("ProvidesPort") == 0) {
      pp.push_back(new PortDescriptor(listofPorts->GetCellValue(i, 0), listofPorts->GetCellValue(i, 1), listofPorts->GetCellValue(i, 2)));
    }

    if (listofPorts->GetCellValue(i, 3).compare("UsesPort") == 0) {
      up.push_back(new PortDescriptor(listofPorts->GetCellValue(i, 0), listofPorts->GetCellValue(i, 1), listofPorts->GetCellValue(i, 2)));
    }
  }

  std::string tmpDir(builder->getConfigDir() + ComponentSkeletonWriter::DIR_SEP + "ComponentGenerationWizard");
  Dir temp(tmpDir);
  if (temp.exists() && isPreviewed) {
    try {
      Dir destDir = Dir(compsDir + compName);
      if (!destDir.exists()) {
        destDir.create(compsDir);
      }
      destDir = Dir(compsDir + ComponentSkeletonWriter::DIR_SEP + compName + std::string(".h") );
      temp.copy("tempheader.txt", destDir);
      destDir = Dir(compsDir + ComponentSkeletonWriter::DIR_SEP + compName + std::string(".cc") );
      temp.copy("tempsource.txt", destDir);
      destDir = Dir(compsDir + ComponentSkeletonWriter::DIR_SEP + std::string("sub.mk") );
      temp.copy("tempsubmake.txt", destDir);
    }
    catch (const sci::cca::CCAException::pointer &e) {
      std::cout << e->getNote() << std::endl;
    }
    event.Skip();
  } else {
    ComponentSkeletonWriter newComponent(componentName->GetValue(), compsDir, pp, up);
    newComponent.GenerateCode();
    event.Skip();
  }
}

//Returns the name of the Component
wxString ComponentWizardDialog::GetText()
{
  return componentName->GetValue();
}

bool ComponentWizardDialog::Validate()
{
  return TRUE;
}

void ComponentWizardDialog::OnAddProvidesPort(wxCommandEvent& event)
{
  AddPortDialog addpport (this,  wxID_ANY, "Add provides port", wxPoint(10, 20), wxSize(600, 600), wxRESIZE_BORDER);
  if (addpport.ShowModal() == wxID_OK) {
    if ((addpport.GetPortNameText().empty())) {
#if DEBUG
      std::cout << "\nPort name is empty";
#endif
      wxString msg;
      msg.Printf(wxT("Port name field is Empty"));

      wxMessageBox(msg, wxT("Add Provides Port"),
                   wxOK | wxICON_INFORMATION, this);
    } else if (addpport.GetDataTypeText().empty()) {
#if DEBUG
      std::cout << "\nPort type is empty";
#endif
      wxString msg;
      msg.Printf(wxT("Port type field is Empty"));

      wxMessageBox(msg, wxT("Add Provides Port"),
                   wxOK | wxICON_INFORMATION, this);

    } else if (addpport.GetDescriptionText().empty()) {
#if DEBUG
      std::cout << "\nPort description is empty";
#endif
      wxString msg;
      msg.Printf(wxT("Port Description field is Empty"));

      wxMessageBox(msg, wxT("Add Povides Port"),
                   wxOK | wxICON_INFORMATION, this);

    } else {
      count_table++;
      int row = count_table - 1;
      listofPorts->InsertRows(row, 1);
      listofPorts->SetCellValue(row, 0, addpport.GetPortNameText());
      listofPorts->SetCellValue(row, 1, addpport.GetDataTypeText());
      listofPorts->SetCellValue(row, 2, addpport.GetDescriptionText());
      listofPorts->SetCellValue(row, 3, "ProvidesPort");
    }
  }
}


void ComponentWizardDialog::OnAddUsesPort(wxCommandEvent& event)
{
  AddPortDialog addpport (this, wxID_ANY, "Add uses port", wxPoint(10, 20), wxSize(600, 600), wxRESIZE_BORDER);
  if (addpport.ShowModal() == wxID_OK) {

    if ((addpport.GetPortNameText().empty())) {
#if DEBUG
      std::cout << "\nPort name is empty" << std::endl;
#endif
      wxString msg;
      msg.Printf(wxT("Port name field is Empty"));

      wxMessageBox(msg, wxT("Add Uses Port"),
                   wxOK | wxICON_INFORMATION, this);
    } else if (addpport.GetDataTypeText().empty()) {
      std::cout << "\nPort type is empty" << std::endl;
      wxString msg;
      msg.Printf(wxT("Port type field is Empty"));

      wxMessageBox(msg, wxT("Add Uses Port"),
                   wxOK | wxICON_INFORMATION, this);

    } else if (addpport.GetDescriptionText().empty()) {
#if DEBUG
      std::cout << "\nPort description is empty";
#endif
      wxString msg;
      msg.Printf(wxT("Port Description field is Empty"));

      wxMessageBox(msg, wxT("Add Uses Port"),
                   wxOK | wxICON_INFORMATION, this);
    } else {
      count_table++;
      int row = count_table - 1;
      listofPorts->InsertRows(row, 1);
      listofPorts->SetCellValue(row, 0, addpport.GetPortNameText());
      listofPorts->SetCellValue(row, 1, addpport.GetDataTypeText());
      listofPorts->SetCellValue(row, 2, addpport.GetDescriptionText());
      listofPorts->SetCellValue(row, 3, "UsesPort");
    }
  }
}


void ComponentWizardDialog::OnRemovePort(wxCommandEvent& event)
{
  count_table--;
  wxArrayInt sel_rows = listofPorts->GetSelectedRows();
  for (unsigned int row_num = 0; row_num < sel_rows.Count(); row_num++) {
    listofPorts->DeleteRows(sel_rows.Item(row_num), 1);
  }
}

void ComponentWizardDialog::OnPreviewCode(wxCommandEvent& event)
{
  if ((componentName->GetValue()).empty()) {
#if DEBUG
    std::cout<<"\nComponent Name is Empty\n";
#endif
    wxString msg;
    msg.Printf(wxT("Component name field is Empty"));

    wxMessageBox(msg, wxT("Create Component"),
                 wxOK | wxICON_INFORMATION, this);
    return;
  }

  pp.clear();
  up.clear();

  for (int i = 0; i < count_table; i++) {
    if (listofPorts->GetCellValue(i, 3).compare("ProvidesPort") == 0) {
      pp.push_back(new PortDescriptor(listofPorts->GetCellValue(i, 0), listofPorts->GetCellValue(i, 1), listofPorts->GetCellValue(i, 2)));
    }

    if (listofPorts->GetCellValue(i, 3).compare("UsesPort") == 0) {
      up.push_back(new PortDescriptor(listofPorts->GetCellValue(i, 0), listofPorts->GetCellValue(i, 1), listofPorts->GetCellValue(i, 2)));
    }
  }
  isPreviewed = true;
  std::string tmpDir(builder->getConfigDir() + ComponentSkeletonWriter::DIR_SEP + "ComponentGenerationWizard");
  Dir temp(tmpDir);
  if (! temp.exists()) {
    Dir::create(tmpDir);
  }

  ComponentSkeletonWriter newComponent(componentName->GetValue(), tmpDir, pp, up);
  newComponent.GenerateTempCode("tempheader.txt", "tempsource.txt", "tempsubmake.txt");
  event.Skip();

  CodePreviewDialog codepreview(wxT(tmpDir + ComponentSkeletonWriter::DIR_SEP + "tempheader.txt"),
                                wxT(tmpDir + ComponentSkeletonWriter::DIR_SEP + "tempsource.txt"),
                                wxT(tmpDir + ComponentSkeletonWriter::DIR_SEP + "tempsubmake.txt"),
                                this, wxID_ANY, wxT("Preview Generated Code"),
                                wxPoint(100, 20), wxSize(700, 500), wxRESIZE_BORDER);
  codepreview.ShowModal();
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

}
