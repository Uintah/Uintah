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

#include <CCA/Components/GUIBuilder/CodePreviewDialog.h>
 #include <SCIRun/Internal/FrameworkProperties.h>

#include <wx/file.h>
#include <wx/textfile.h>

#include <iostream>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(CodePreviewDialog, wxDialog)
  EVT_BUTTON( ID_ViewSourceFileCode, CodePreviewDialog::OnViewSourceFileCode )
  EVT_BUTTON( ID_ViewHeaderFileCode, CodePreviewDialog::OnViewHeaderFileCode )
  EVT_BUTTON( ID_ViewMakeFileCode, CodePreviewDialog::OnViewMakeFileCode )
END_EVENT_TABLE()

CodePreviewDialog::CodePreviewDialog(wxWindow *parent,wxWindowID id, const wxString &title, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
  : wxDialog( parent, id, title, pos, size, style)
{
  const int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  wxFlexGridSizer *topSizer = new wxFlexGridSizer(3,1,2,2);
  wxBoxSizer *textSizer = new wxBoxSizer( wxVERTICAL );
  wxBoxSizer *viewbuttonSizer = new wxBoxSizer( wxHORIZONTAL );
  wxBoxSizer *cancelSizer = new wxBoxSizer( wxHORIZONTAL );
  codePreview = new wxTextCtrl(this,wxID_ANY,wxT(""),wxDefaultPosition,wxSize(600,400),wxTE_MULTILINE);
  textSizer->Add(codePreview, 1, centerFlags, 2);
  viewSourceFileCode = new wxButton(this,ID_ViewSourceFileCode,wxT("View Source File Code"));
  viewHeaderFileCode = new wxButton(this,ID_ViewHeaderFileCode,wxT("View Header File Code"));
  viewMakeFileCode = new wxButton(this,ID_ViewMakeFileCode,wxT("View Make File Code"));
  viewbuttonSizer->Add(viewSourceFileCode, 1, centerFlags, 2);
  viewbuttonSizer->Add(viewHeaderFileCode, 1, centerFlags, 2);
  viewbuttonSizer->Add(viewMakeFileCode, 1, centerFlags, 2);
  cancelSizer->Add(new wxButton(this,wxID_CANCEL,wxT("&Cancel")), 1, centerFlags, 2);
  topSizer->AddSpacer(10);
  topSizer->Add(textSizer, 1, centerFlags, 2);
  topSizer->AddSpacer(20);
  topSizer->Add(viewbuttonSizer, 1, centerFlags, 2);
  topSizer->AddSpacer(20);
  topSizer->Add(cancelSizer, 1, centerFlags, 2);

  SetAutoLayout( TRUE );     // tell dialog to use sizer
  SetSizer( topSizer );      // actually set the sizer
  topSizer->Fit( this );           // set size to minimum size as calculated by the sizer
  topSizer->SetSizeHints( this );   // set size hints to honour mininum size
}
void CodePreviewDialog::OnViewSourceFileCode(wxCommandEvent& event)
{
  codePreview->Clear();
  std::ifstream tempfile;
  std::string home(getenv("HOME"));
  std::string tmp(FrameworkProperties::CONFIG_DIR);
  std::string tmpDir = std::string(home + tmp  + "/ComponentGenerationWizard");
  std::string currPath(tmpDir + "/tempsource.txt");
  tempfile.open(currPath.c_str());
  if(!tempfile)
    {
      //#if DEBUG
        std::cout << "unable to read file" << std::endl;
	//#endif
    }
    else
    {
      std::string line;
      codePreview->Clear();
      codePreview->AppendText(wxT("//Source File\n"));
      while(!tempfile.eof())
	 {
	   std::getline (tempfile,line);
	   codePreview->AppendText(wxString(line));
	   codePreview->AppendText(wxT("\n"));
	 }
       tempfile.close();
    }
}
void CodePreviewDialog::OnViewHeaderFileCode(wxCommandEvent& event)
{
  codePreview->Clear();
  std::ifstream tempfile;
  std::string home(getenv("HOME"));
  std::string tmp(FrameworkProperties::CONFIG_DIR);
  std::string tmpDir = std::string(home + tmp  + "/ComponentGenerationWizard");
  std::string currPath(tmpDir + "/tempheader.txt");
  tempfile.open(currPath.c_str());
  if(!tempfile)
    {
      //#if DEBUG
        std::cout << "unable to read file" << std::endl;
	//#endif
    }
    else
    {
      std::string line;
      codePreview->Clear();
      codePreview->AppendText(wxT("//Header File\n"));
      while(!tempfile.eof())
	 {
	   std::getline (tempfile,line);
	   codePreview->AppendText(wxString(line));
	   codePreview->AppendText(wxT("\n"));
	 }
       tempfile.close();
    }

}
void CodePreviewDialog::OnViewMakeFileCode(wxCommandEvent& event)
{
  codePreview->Clear();
  std::ifstream tempfile;
  std::string home(getenv("HOME"));
  std::string tmp(FrameworkProperties::CONFIG_DIR);
  std::string tmpDir = std::string(home + tmp  + "/ComponentGenerationWizard");
  std::string currPath(tmpDir + "/tempsubmake.txt");
  tempfile.open(currPath.c_str());
  if(!tempfile)
    {
      //#if DEBUG
        std::cout << "unable to read file" << std::endl;
	//#endif
    }
    else
    {
      std::string line;
      codePreview->Clear();
      codePreview->AppendText(wxT("#Make File\n"));
      while(!tempfile.eof())
	 {
	   std::getline (tempfile,line);
	   codePreview->AppendText(wxString(line));
	   codePreview->AppendText(wxT("\n"));
	 }
       tempfile.close();
    }
}

}

