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
  /*wxString dimensions, s;
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
  wxButton * b = new wxButton( this, wxID_OK, "OK",p, wxDefaultSize);
  p.x += 100;
  wxButton * c = new wxButton( this, wxID_CANCEL,"Cancel", p, wxDefaultSize);*/

  
  
    wxBoxSizer *topsizer = new wxBoxSizer( wxVERTICAL );
    wxGridSizer *gridsizer = new wxGridSizer(2,5,10);
    wxBoxSizer *button_sizer = new wxBoxSizer( wxHORIZONTAL );

    componentName = new wxTextCtrl( this,-1,"");

    gridsizer->Add(new wxStaticText(this,-1,"Component Name"),
                   wxSizerFlags().Align(wxALIGN_RIGHT | wxALIGN_CENTER_VERTICAL));
    gridsizer->Add(componentName,wxSizerFlags(1).Align( wxGROW |wxALIGN_CENTER_VERTICAL));
    
    gridsizer->AddSpacer(10);
    gridsizer->AddSpacer(10);
    
    gridsizer->Add(new wxButton(this,ID_AddProvidesPort ,"Add Provides Port"),
                   wxSizerFlags().Align(wxALIGN_RIGHT |wxALIGN_CENTER_VERTICAL));
    gridsizer->Add(new wxButton(this,ID_AddUsesPort,"Add Uses Port"),
                  wxSizerFlags(1).Align(wxALIGN_RIGHT |wxALIGN_CENTER_VERTICAL));
    

    gridsizer->AddSpacer(10);
    gridsizer->AddSpacer(10);

    button_sizer->Add( new wxButton( this, wxID_OK, "OK" ), 0, wxALL,10 );
    button_sizer->Add( new wxButton( this, wxID_CANCEL, "Cancel" ), 0, wxALL, 10 );
   
    topsizer->Add(gridsizer,wxSizerFlags().Proportion(1).Expand().Border(wxALL, 10));
    topsizer->Add( button_sizer, 1, wxALIGN_CENTER,50 );


  SetAutoLayout( TRUE );     // tell dialog to use sizer
  SetSizer( topsizer );      // actually set the sizer

  topsizer->Fit( this );            // set size to minimum size as calculated by the sizer
  topsizer->SetSizeHints( this );   // set size hints to honour mininum size

}

  ComponentWizardDialog::~ComponentWizardDialog()
 {
   for(unsigned int i=0;i<pp.size();i++)
	delete pp[i];
   for(unsigned int i=0;i<up.size();i++)
   delete up[i];
   }
void ComponentWizardDialog::OnOk(wxCommandEvent& event)
{
  
  
    if(componentName->GetValue()=="")
    {
       std::cout<<"\nComponent Name is Empty\n";
       wxString msg;
       msg.Printf(wxT("Component name field is Empty"));
             
       wxMessageBox(msg, wxT("Create Component"),
               wxOK | wxICON_INFORMATION, this);
    }
    else
    {
        ComponentSkeletonWriter newComponent(componentName->GetValue(),pp,up);
	newComponent.GenerateCode();
	event.Skip();
    }
  
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
  
  AddPortDialog addpport (this, -1, "Add provides port", wxPoint(10, 20), wxSize(400, 400), wxRESIZE_BORDER);
  if (addpport.ShowModal() == wxID_OK) 
    {
       if((addpport.GetPortNameText()==""))
       {
	 std::cout << "\nPort name is empty";
	 wxString msg;
	 msg.Printf(wxT("Port name field is Empty"));
             
         wxMessageBox(msg, wxT("Add Provides Port"),
               wxOK | wxICON_INFORMATION, this);
       }
      
       else if (addpport.GetDataTypeText()=="")
       { 
	 std::cout << "\nPort type is empty";
	 wxString msg;
	 msg.Printf(wxT("Port type field is Empty"));
             
         wxMessageBox(msg, wxT("Add Provides Port"),
               wxOK | wxICON_INFORMATION, this);
	      
       }
        else if (addpport.GetDescriptionText()=="")
       { 
	 std::cout << "\nPort description is empty";
	 wxString msg;
	 msg.Printf(wxT("Port Description field is Empty"));
             
         wxMessageBox(msg, wxT("Add Povides Port"),
               wxOK | wxICON_INFORMATION, this);
	      
       }
       else
	{
	   PortDescriptor p(addpport.GetPortNameText(), addpport.GetDataTypeText(),addpport.GetDescriptionText());

	   std::cout << p.GetName() << "\t" << p.GetType() << "\t" << p.GetDesc() << std::endl;
	   pp.push_back(new PortDescriptor(addpport.GetPortNameText(), addpport.GetDataTypeText(),addpport.GetDescriptionText()));
	}
    }
}
void ComponentWizardDialog::OnAddUsesPort(wxCommandEvent& event)
{
  
  
  AddPortDialog addpport (this, -1, "Add uses port", wxPoint(10, 20), wxSize(400, 400),wxRESIZE_BORDER);
  if (addpport.ShowModal() == wxID_OK) 
    {
        

       if((addpport.GetPortNameText()==""))
       {
	 std::cout << "\nPort name is empty";
	 wxString msg;
	 msg.Printf(wxT("Port name field is Empty"));
             
         wxMessageBox(msg, wxT("Add Uses Port"),
               wxOK | wxICON_INFORMATION, this);
       }
       else if (addpport.GetDataTypeText()=="")
       { 
	 std::cout << "\nPort type is empty";
	 wxString msg;
	 msg.Printf(wxT("Port type field is Empty"));
             
         wxMessageBox(msg, wxT("Add Uses Port"),
               wxOK | wxICON_INFORMATION, this);
	      
       }
      else if (addpport.GetDescriptionText()=="")
       { 
	 std::cout << "\nPort description is empty";
	 wxString msg;
	 msg.Printf(wxT("Port Description field is Empty"));
             
         wxMessageBox(msg, wxT("Add Uses Port"),
               wxOK | wxICON_INFORMATION, this);
	      
       }
      else
      {
	PortDescriptor p(addpport.GetPortNameText(), addpport.GetDataTypeText(),addpport.GetDescriptionText());
	std::cout << p.GetName() << "\t" << p.GetType() << "\t" << p.GetDesc() << std::endl;
	up.push_back(new PortDescriptor(addpport.GetPortNameText(), addpport.GetDataTypeText(), addpport.GetDescriptionText()));

      }
    }

}


//////////////////////////////////////////////////////////////////////////
// AddPortDialog helper class

AddPortDialog::AddPortDialog(wxWindow *parent,wxWindowID id, const wxString &title,
			       const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : wxDialog( parent, id, title, pos, size, style)
{

  wxPoint p;

  wxBoxSizer *topsizer = new wxBoxSizer( wxVERTICAL );
  wxBoxSizer *button_sizer = new wxBoxSizer( wxHORIZONTAL );
  wxGridSizer *gridsizer = new wxGridSizer(2,2,5,5);
  
  p.x = pos.x;
  p.y = pos.y;
  p.x += 50;
  p.y += 50;
  
  lname= new wxStaticText(this,-1,"Name");
   pname = new wxTextCtrl(this, -1,"");

   p.y += 50;
   ldtype = new wxStaticText(this, -1, "Datatype");
   dtype= new wxTextCtrl(this, -1,"");
  
  p.y += 100;
  
   ldesc = new wxStaticText(this, -1, "Description");
   desc= new wxTextCtrl(this, -1,"");
  
  wxButton *okbutton = new wxButton(this, wxID_OK, "OK");
  wxButton *cancelbutton = new wxButton(this, wxID_CANCEL, "Cancel");
 
  button_sizer->Add(okbutton,0,wxALL,10);
  button_sizer->Add(cancelbutton,0,wxALL,10);
  
   gridsizer->Add(lname,1,wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL);
   gridsizer->Add(pname,1,wxGROW|wxALIGN_CENTER_VERTICAL);
   gridsizer->Add(ldtype,1,wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL);
   gridsizer->Add(dtype,1,wxGROW|wxALIGN_CENTER_VERTICAL);
   gridsizer->Add(ldesc,1,wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL);
   gridsizer->Add(desc,1,wxGROW|wxALIGN_CENTER_VERTICAL);
  
   topsizer->Add(gridsizer,wxSizerFlags().Proportion(1).Expand().Border(wxALL, 20));
   topsizer->Add( button_sizer, 1, wxALIGN_CENTER,0 );


  SetAutoLayout( TRUE );     // tell dialog to use sizer
  SetSizer( topsizer );      // actually set the sizer

  topsizer->Fit( this );            // set size to minimum size as calculated by the sizer
  topsizer->SetSizeHints( this );   // set size hints to honour mininum size
  

}

std::string AddPortDialog::GetPortNameText() const
{
  return std::string(pname->GetValue().c_str());
}

std::string AddPortDialog::GetDataTypeText() const
{
  return std::string(dtype->GetValue().c_str());
}
std::string AddPortDialog::GetDescriptionText() const
{
  return std::string(desc->GetValue().c_str());
}
}
