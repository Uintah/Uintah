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
 * FrameworkProxyDialog.cc
 *
 * Written by:
 *  Ayla Khan
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  September 2004
 *
 *  Copyright (C) 2004 SCI Institute
 *
 */

#include <CCA/Components/GUIBuilder/FrameworkProxyDialog.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>

#include <wx/gbsizer.h>
#include <wx/combobox.h>
#include <wx/button.h>
#include <wx/radiobox.h>
#include <wx/textctrl.h>
#include <wx/stattext.h>
#include <wx/spinctrl.h>

#include <iostream>

namespace GUIBuilder {
//using namespace SCIRun;

BEGIN_EVENT_TABLE(FrameworkProxyDialog, wxDialog)
  EVT_BUTTON(ID_BUTTON_RESET, FrameworkProxyDialog::OnReset)
END_EVENT_TABLE()

FrameworkProxyDialog::FrameworkProxyDialog(wxWindow *parent,
                                           wxWindowID id,
                                           const wxString &title,
                                           const wxPoint& pos,
                                           const wxSize& size,
                                           long style,
                                           const wxString& name,
                                           const std::string& defaultProxy,
                                           const std::string& defaultDomain,
                                           const std::string& defaultLogin)
  : wxDialog(parent, id, title, pos, size, style),
    proxy(wxT(defaultProxy)),
    domain(wxT(defaultDomain)),
    login(wxT(defaultLogin)),
    path(wxT(GUIBuilder::DEFAULT_OBJ_DIR))
{
  SetLayout();
}

FrameworkProxyDialog::~FrameworkProxyDialog()
{
//   passwordTextCtrl->SetValue("");
}

void FrameworkProxyDialog::OnReset(wxCommandEvent& event)
{
  SetDefaultText();
}

wxString FrameworkProxyDialog::GetLoader() const
{
  return proxyComboBox->GetValue().Strip();
}

wxString FrameworkProxyDialog::GetDomain() const
{
  return domainComboBox->GetValue().Strip();
}

wxString FrameworkProxyDialog::GetLogin() const
{
  return loginComboBox->GetValue().Strip();
}

wxString FrameworkProxyDialog::GetPath() const
{
  return pathComboBox->GetValue().Strip();
}

// wxString FrameworkProxyDialog::GetPassword() const
// {
//   return passwordTextCtrl->GetValue().Strip();
// }

int FrameworkProxyDialog::GetCopiesNumber() const
{
  return copiesSpinCtrl->GetValue();
}

wxString FrameworkProxyDialog::GetMPIWhereString() const
{
  return mpiWhereRadioBox->GetStringSelection();
}

///////////////////////////////////////////////////////////////////////////
// protected member functions

void FrameworkProxyDialog::SetDefaultText()
{
  // TODO: make sure that user entries get preserved in combo boxes, config dir!
  proxyComboBox->SetValue(proxy);
  domainComboBox->SetValue(domain);
  loginComboBox->SetValue(login);
  pathComboBox->SetValue(path);

  // reset copiesSpinCtrl
}

// constructor helpers

void FrameworkProxyDialog::SetLayout()
{
  const int BORDER = 4;
  const int leftFlags = wxALIGN_CENTER_VERTICAL|wxALL|wxALIGN_LEFT;
  const int rightFlags = wxALIGN_CENTER_VERTICAL|wxALL|wxALIGN_RIGHT;

  wxGridBagSizer *topSizer = new wxGridBagSizer();

  topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Proxy framework name")), wxGBPosition(0, 0), wxGBSpan(1, 2), leftFlags, BORDER);
  proxyComboBox = new wxComboBox(this, wxID_ANY, proxy, wxDefaultPosition, wxSize(200, wxDefaultSize.GetHeight()), wxArrayString(), wxCB_DROPDOWN|wxCB_SORT);
  proxyComboBox->SetToolTip(wxT("Proxy framework (ploader) build directory"));
  topSizer->Add(proxyComboBox, wxGBPosition(0, 2), wxGBSpan(1, 2), wxGROW|leftFlags, BORDER);

  topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Domain Name")), wxGBPosition(1, 0), wxGBSpan(1, 2), leftFlags, BORDER);
  domainComboBox = new wxComboBox(this, wxID_ANY, domain, wxDefaultPosition, wxSize(TEXT_CTRL_WIDTH, wxDefaultSize.GetHeight()), wxArrayString(), wxCB_DROPDOWN|wxCB_SORT);
  topSizer->Add(domainComboBox, wxGBPosition(1, 2), wxGBSpan(1, 2), wxGROW|leftFlags, BORDER);

  topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Login")), wxGBPosition(2, 0), wxGBSpan(1, 2), leftFlags, BORDER);
  loginComboBox = new wxComboBox(this, wxID_ANY, login, wxDefaultPosition, wxSize(TEXT_CTRL_WIDTH, wxDefaultSize.GetHeight()), wxArrayString(), wxCB_DROPDOWN|wxCB_SORT);
  topSizer->Add(loginComboBox, wxGBPosition(2, 2), wxGBSpan(1, 2), wxGROW|leftFlags, BORDER);

  // Is getting a password a good idea?
//   topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Password")), 0, leftFlags, 2);
//   passwordTextCtrl = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition, wxSize(TEXT_CTRL_WIDTH, wxDefaultSize.GetHeight()), wxTE_PASSWORD);
//   topSizer->Add(passwordTextCtrl, 0, leftFlags, 2);

  topSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Proxy framework directory")), wxGBPosition(3, 0), wxGBSpan(1, 2), wxALIGN_CENTER_VERTICAL|wxALL|wxALIGN_LEFT, BORDER);
  pathComboBox = new wxComboBox(this, wxID_ANY, path, wxDefaultPosition, wxSize(TEXT_CTRL_WIDTH, wxDefaultSize.GetHeight()), wxArrayString(), wxCB_DROPDOWN|wxCB_SORT);

  topSizer->Add(pathComboBox, wxGBPosition(3, 2), wxGBSpan(1, 3), wxGROW|leftFlags, BORDER);

  copiesStaticText = new wxStaticText(this, wxID_ANY, wxT("Number of program copies (MPI)"));
  topSizer->Add(copiesStaticText, wxGBPosition(4, 0), wxGBSpan(1, 2), leftFlags, BORDER);
  // TODO: find mpi np limits?
  copiesSpinCtrl = new wxSpinCtrl(this, wxID_ANY, wxT("2"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS|wxSP_WRAP, 1, 10000, 1);
  topSizer->Add(copiesSpinCtrl, wxGBPosition(4, 2), wxDefaultSpan, leftFlags, BORDER);

  wxArrayString options;
  options.Add("&CPU");
  options.Add("&Node");
  mpiWhereRadioBox = new wxRadioBox(this, wxID_ANY, wxT("Execute on"), wxDefaultPosition, wxDefaultSize, options, wxRA_SPECIFY_ROWS);
  mpiWhereRadioBox->SetToolTip(wxT("Execute on which MPI CPUs or nodes?"));
  topSizer->Add(mpiWhereRadioBox, wxGBPosition(5, 2), wxDefaultSpan, leftFlags, BORDER);

  topSizer->Add(new wxButton(this, wxID_OK, wxT("&OK")), wxGBPosition(6, 2), wxDefaultSpan, rightFlags, BORDER);
  topSizer->Add(new wxButton(this, wxID_CANCEL, wxT("&Cancel")), wxGBPosition(6, 3), wxDefaultSpan, leftFlags, BORDER);

  resetButton = new wxButton(this, ID_BUTTON_RESET, wxT("&Reset"));
  topSizer->Add(resetButton, wxGBPosition(6, 4), wxDefaultSpan, rightFlags, BORDER);

  helpButton = new wxButton(this, wxID_HELP, wxT("&Help"));
  topSizer->Add(helpButton, wxGBPosition(6, 5), wxDefaultSpan, leftFlags, BORDER);

  SetAutoLayout(true);            // tell dialog to use sizer
  SetSizer(topSizer);            // actually set the sizer
  topSizer->Fit(this);           // set size to minimum size as calculated by the sizer
  topSizer->SetSizeHints(this);  // set size hints to honour minimum size


  // Disable MPI copies until mpi args supported in FrameworkProxyService
  copiesStaticText->Enable(false);
  copiesSpinCtrl->Enable(false);
  mpiWhereRadioBox->Enable(false);

  // Disable help until online help is available
  helpButton->Enable(false);
}

}

