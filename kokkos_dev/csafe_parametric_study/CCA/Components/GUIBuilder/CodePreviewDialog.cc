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
 * CodePreviewDialog.cc
 *
 * Written by:
 *  <author>
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  <date>
 *
 */

#include <CCA/Components/GUIBuilder/CodePreviewDialog.h>

#include<wx/grid.h>
#include <wx/file.h>
#include <wx/textfile.h>

#include <iostream>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace GUIBuilder {

BEGIN_EVENT_TABLE(CodePreviewDialog, wxDialog)
  EVT_BUTTON( ID_ViewSourceFileCode, CodePreviewDialog::OnViewSourceFileCode )
  EVT_BUTTON( ID_ViewHeaderFileCode, CodePreviewDialog::OnViewHeaderFileCode )
  EVT_BUTTON( ID_ViewMakeFileCode, CodePreviewDialog::OnViewMakeFileCode )
  EVT_BUTTON( ID_ViewSidlFileCode, CodePreviewDialog::OnViewSidlFileCode )
END_EVENT_TABLE()

CodePreviewDialog::CodePreviewDialog(const wxString& headerFile,
                                     const wxString& sourceFile,
                                     const wxString& submakeFile,
                                     const wxString& sidlFile,
                                     const bool isWithSidl,
                                     wxWindow *parent,
                                     wxWindowID id,
                                     const wxString &title,
                                     const wxPoint& pos,
                                     const wxSize& size,
                                     long style,
                                     const wxString& name)
  : wxDialog(parent, id, title, pos, size, style),
    headerFile(headerFile),
    sourceFile(sourceFile),
    submakeFile(submakeFile),
    sidlFile(sidlFile),
    isWithSidl(isWithSidl)
{
  const int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;

  wxFlexGridSizer *topSizer = new wxFlexGridSizer(3, 1, 2, 2);
  wxBoxSizer *textSizer = new wxBoxSizer( wxVERTICAL );
  wxBoxSizer *viewbuttonSizer = new wxBoxSizer( wxHORIZONTAL );
  wxBoxSizer *cancelSizer = new wxBoxSizer( wxHORIZONTAL );
  codePreview = new wxTextCtrl(this, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(600, 400), wxTE_MULTILINE);
  textSizer->Add(codePreview, 1, wxEXPAND, 2);
  viewSourceFileCode = new wxButton(this, ID_ViewSourceFileCode, wxT("View Source File Code"));
  viewHeaderFileCode = new wxButton(this, ID_ViewHeaderFileCode, wxT("View Header File Code"));
  viewMakeFileCode = new wxButton(this, ID_ViewMakeFileCode, wxT("View Makefile Code"));

  viewbuttonSizer->Add(viewSourceFileCode, 1, centerFlags, 2);
  viewbuttonSizer->Add(viewHeaderFileCode, 1, centerFlags, 2);
  viewbuttonSizer->Add(viewMakeFileCode, 1, centerFlags, 2);
  if(isWithSidl)
    viewbuttonSizer->Add(new wxButton(this, ID_ViewSidlFileCode, wxT("View Sidl File Code")), 1, centerFlags, 2);
  cancelSizer->Add(new wxButton(this, wxID_CANCEL, wxT("&Cancel")), 1, centerFlags, 2);
  topSizer->AddSpacer(10);
  topSizer->Add(textSizer, 1, wxEXPAND, 2);
  topSizer->AddSpacer(20);
  topSizer->Add(viewbuttonSizer, 1, centerFlags, 2);
  topSizer->AddSpacer(20);
  topSizer->Add(cancelSizer, 1, centerFlags, 2);

  topSizer->AddGrowableCol(0);
  topSizer->AddGrowableRow(0);
  topSizer->AddGrowableRow(1);
  topSizer->AddGrowableRow(2);
  SetAutoLayout( TRUE );     // tell dialog to use sizer
  SetSizer( topSizer );      // actually set the sizer
  topSizer->Fit( this );           // set size to minimum size as calculated by the sizer
  topSizer->SetSizeHints( this );   // set size hints to honour mininum size
  wxCommandEvent tmpEvent;
  OnViewSourceFileCode(tmpEvent);
}

void CodePreviewDialog::OnViewSourceFileCode(wxCommandEvent& event)
{
  previewCode(sourceFile, wxT("Preview Source File"));
}

void CodePreviewDialog::OnViewHeaderFileCode(wxCommandEvent& event)
{
  previewCode(headerFile, wxT("Preview Header File"));
}

void CodePreviewDialog::OnViewSidlFileCode(wxCommandEvent& event)
{
  previewCode(sidlFile, wxT("Preview SIDL File"));
}

void CodePreviewDialog::OnViewMakeFileCode(wxCommandEvent& event)
{
  previewCode(submakeFile, wxT("Preview Makefile"));
}

///////////////////////////////////////////////////////////////////////////
// private member functions

void CodePreviewDialog::previewCode(const wxString& filename, const wxString& title)
{
  codePreview->Clear();

  std::ifstream tempfile;
  tempfile.open(wxToSTLString(filename).c_str());
  if (!tempfile) {
#if DEBUG
    std::cout << "unable to read file " << wxToSTLString(filename) << std::endl;
#endif
    return;
    // TODO: error dialog or write error to builder window...
  }

  std::string line;
  //codePreview->Clear();
  SetTitle(title);
  while (!tempfile.eof()) {
    std::getline(tempfile, line);
    codePreview->AppendText(STLTowxString(line));
    codePreview->AppendText(wxT("\n"));
  }
  tempfile.close();
}

}
