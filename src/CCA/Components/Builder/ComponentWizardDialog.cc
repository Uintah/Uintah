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

#include <CCA/Components/Builder/ComponentWizardDialog.h>
#include <Core/Containers/StringUtil.h>

#include <wx/file.h>
#include <wx/textfile.h>

#include <iostream>

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(ComponentWizardDialog, wxDialog)
  EVT_BUTTON( wxID_OK, ComponentWizardDialog::OnOk )
  EVT_BUTTON( ID_AddProvidesPort, ComponentWizardDialog::OnAddProvidesPort )
  EVT_BUTTON( ID_AddUsesPort, ComponentWizardDialog::OnAddUsesPort )
END_EVENT_TABLE()

ComponentWizardDialog::ComponentWizardDialog(wxWindow *parent, wxWindowID id, const wxString &title,
                                             const wxPoint& pos, const wxSize& size,
                                             long style, const wxString& name)
    : wxDialog( parent, id, title, pos, size, style)
{
  wxString dimensions, s;
  wxPoint p;
  wxSize  sz;

  sz.SetWidth(size.GetWidth() - 500);    //set size of text control
  sz.SetHeight(size.GetHeight() - 570);

  p.x = pos.x; p.y = pos.y;          //set x y position for text control
  p.y += sz.GetHeight() + 100;
  p.x += 100;
  componentName = new wxTextCtrl( this,-1,"",p,wxSize(100,30),wxTE_MULTILINE);
  lcomponentName = new wxStaticText (this,-1,"component", wxPoint(p.x-100,p.y), wxSize(150,50));
  p.y +=  100;
  p.x -= 100;
  AddProvidesPort = new wxButton(this,ID_AddProvidesPort,"Add Provides Port",p,wxDefaultSize);
  p.x += 130;
  AddUsesPort = new wxButton(this,ID_AddUsesPort,"Add Uses Port",p,wxDefaultSize);
  p.x -= 100;
  p.y += 100;
  wxButton * b = new wxButton( this, wxID_OK,     "OK",p, wxDefaultSize);
  p.x += 100;
  wxButton * c = new wxButton( this, wxID_CANCEL, "Cancel", p, wxDefaultSize);

}


void ComponentWizardDialog::OnOk(wxCommandEvent& event)
{
  
  std::cout<<"here in cwd \nprovides portname: "<<pp[0]->GetName()<<"\nUses port name: "<<up[0]->GetName()<<"\n";
   ComponentSkeletonWriter newComponent(componentName->GetValue(),pp,up);
   newComponent.GenerateCode();
   event.Skip();
}

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
  
  AddPortDialog addpport (this, -1, "Add provides port", wxPoint(10, 20), wxSize(400, 400));
  if (addpport.ShowModal() == wxID_OK) {
      PortDescriptor p(addpport.GetPortNameText(), addpport.GetDataTypeText());
   
    std::cout << p.GetName() << "\t" << p.GetType() << std::endl;
      pp.push_back(new PortDescriptor(addpport.GetPortNameText(), addpport.GetDataTypeText()));
 
  }
}
void ComponentWizardDialog::OnAddUsesPort(wxCommandEvent& event)
{
  
  AddPortDialog addpport (this, -1, "Add uses port", wxPoint(10, 20), wxSize(400, 400));
  if (addpport.ShowModal() == wxID_OK) {
      PortDescriptor p(addpport.GetPortNameText(), addpport.GetDataTypeText());
   
    std::cout << p.GetName() << "\t" << p.GetType() << std::endl;
      up.push_back(new PortDescriptor(addpport.GetPortNameText(), addpport.GetDataTypeText()));
 
  }

}


//////////////////////////////////////////////////////////////////////////
// AddPortDialog helper class

AddPortDialog::AddPortDialog(wxWindow *parent,wxWindowID id, const wxString &title,
			       const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : wxDialog( parent, id, title, pos, size, style)
{

  wxPoint p;
  p.x = pos.x;
  p.y = pos.y;
  p.x += 50;
  p.y += 50;
  lname = new wxStaticText(this, -1,"Name", p, wxSize(100,50));
  pname = new wxTextCtrl(this, -1, "", wxPoint((p.x+100),p.y), wxSize(100,30), wxTE_MULTILINE);
  p.y += 50;
  ldtype = new wxStaticText(this, -1, "Datatype", p, wxSize(100,50));
  dtype = new wxTextCtrl(this, -1, "", wxPoint((p.x+100),p.y), wxSize(100,30), wxTE_MULTILINE);

  p.y += 100;
  wxButton *okbutton = new wxButton(this, wxID_OK, "OK", wxPoint((p.x+50), p.y), wxDefaultSize);
  wxButton *cancelbutton = new wxButton(this, wxID_CANCEL, "Cancel", wxPoint((p.x+150), p.y), wxDefaultSize);
}

std::string AddPortDialog::GetPortNameText() const
{
  return std::string(pname->GetValue().c_str());
}

std::string AddPortDialog::GetDataTypeText() const
{
  return std::string(dtype->GetValue().c_str());
}

}
