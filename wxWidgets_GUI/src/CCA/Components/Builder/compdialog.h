#ifndef compdialog_h
#define compdialog_h


#include "wx/wx.h" 
#include "wx/file.h"
#include "wx/textfile.h"
#include<iostream>
#include<vector>
#include"compske.h"

namespace GUIBuilder{

class MyCustomDialog: public wxDialog
{
	public:
	MyCustomDialog(wxWindow *parent,wxWindowID id,const wxString &title,const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE, const wxString& name = "MyCustomDialogBox");
	wxTextCtrl  *mytextctrl;
	wxStaticText *l1;
	wxButton *app;
	wxButton *aup;
	 wxString GetText();
	 virtual bool Validate();
  	 private:
    	void OnOk( wxCommandEvent &event );
	void Onapp( wxCommandEvent &event );
	void Onaup( wxCommandEvent &event );
	std::vector <port> pp;
	DECLARE_EVENT_TABLE()

};
class Ppdialog:public wxDialog
{
	public:
	Ppdialog(wxWindow *parent,wxWindowID id,const wxString &title,const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE, const wxString& name = "Provides port dialog box");
	wxTextCtrl  *pname;
	wxTextCtrl  *dtype;
	wxStaticText *lname;
	wxStaticText *ldtype;
	wxButton *ok;
	wxButton *cancel; 
// 	DECLARE_EVENT_TABLE()

};

enum
{
	ID_app=1,
        ID_aup=2
};

BEGIN_EVENT_TABLE(MyCustomDialog, wxDialog)
    EVT_BUTTON( wxID_OK, MyCustomDialog::OnOk )
    EVT_BUTTON( ID_app, MyCustomDialog::Onapp )
   EVT_BUTTON( ID_aup, MyCustomDialog::Onaup )
END_EVENT_TABLE()
MyCustomDialog::MyCustomDialog(wxWindow *parent,wxWindowID id,const wxString &title,const wxPoint& pos , const wxSize& size , long style , const wxString& name): wxDialog( parent, id, title, pos, size, style)
{	
	wxString dimensions, s;
	wxPoint p;
	wxSize  sz;
	
	sz.SetWidth(size.GetWidth() - 500);    //set size of text control
	sz.SetHeight(size.GetHeight() - 570);
	
	p.x = pos.x; p.y = pos.y; 					//set x y position for text control
	p.y += sz.GetHeight() + 100;
	p.x+=100;
	mytextctrl = new wxTextCtrl	( this,-1,"",p,wxSize(100,30),wxTE_MULTILINE	);
	l1=new wxStaticText (this,-1,"component", wxPoint(p.x-100,p.y), wxSize(150,50));
	p.y +=  100; 
	p.x-=100;
	app=new wxButton(this,ID_app,"Add Provides Port",p,wxDefaultSize);
	p.x += 130;
	aup=new wxButton(this,ID_aup,"Add Uses Port",p,wxDefaultSize);
	p.x-=100;
	p.y+=100;
	wxButton * b = new wxButton( this, wxID_OK,     "OK",p, wxDefaultSize);
	p.x += 100;	
	wxButton * c = new wxButton( this, wxID_CANCEL, "Cancel", p, wxDefaultSize);
	
}

Ppdialog::Ppdialog(wxWindow *parent,wxWindowID id,const wxString &title,const wxPoint& pos , const wxSize& size , long style , const wxString& name): wxDialog( parent, id, title, pos, size, style)
{
	
	wxPoint p;
	p.x=pos.x;
	p.y=pos.y;
	p.x+=50;
	p.y+=50;
	lname=new wxStaticText(this,-1,"Name",p,wxSize(100,50));
	pname=new wxTextCtrl(this,-1,"",wxPoint((p.x+100),p.y),wxSize(100,30),wxTE_MULTILINE);
	p.y+=50;
	ldtype=new wxStaticText(this,-1,"Datatype",p,wxSize(100,50));
	dtype=new wxTextCtrl(this,-1,"",wxPoint((p.x+100),p.y),wxSize(100,30),wxTE_MULTILINE);
	
	p.y+=100;
	wxButton *okbutton = new wxButton(this, wxID_OK, "OK", wxPoint((p.x+50),p.y), wxDefaultSize);
	
	wxButton *cancelbutton = new wxButton(this, wxID_CANCEL, "Cancel", wxPoint((p.x+150),p.y), wxDefaultSize);
	
}

void MyCustomDialog::OnOk(wxCommandEvent& event)
{
	ske aa;
	std::string s=mytextctrl->GetValue();
	aa.pp=pp;
	aa.compname=s;
	aa.gen();
	event.Skip();
}
wxString MyCustomDialog::GetText()
{
	return mytextctrl->GetValue();
}
bool MyCustomDialog::Validate()
{
    return TRUE;
}
void MyCustomDialog::Onapp(wxCommandEvent& event)
{
// 	wxMessageBox("Add provides port","Hello World Sample", wxOK, this);
	Ppdialog addpport ( this,-1,"Add provides port",wxPoint(10,20),wxSize(400,400));
	if (addpport.ShowModal()==wxID_OK)
	{
		port p;
		p.name=addpport.pname->GetValue();
		p.type=addpport.dtype->GetValue();
		std::cout<<p.name<<"\t"<<p.type<<std::endl;
		pp.push_back(p);
	}
}
void MyCustomDialog::Onaup(wxCommandEvent& event)
{
	wxMessageBox("Add Uses port",
        "Hello World Sample", wxOK, this);
}
}
#endif
