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

#include <wx/sizer.h>
#include <wx/combobox.h>
#include <wx/button.h>
#include <wx/radiobox.h>
#include <wx/textctrl.h>
#include <wx/stattext.h>

#include <iostream>

namespace GUIBuilder {

FrameworkProxyDialog::FrameworkProxyDialog(const std::string& defaultLoader, const std::string& defaultDomain, const std::string& defaultLogin)
{
  SetLayout();
}

FrameworkProxyDialog::~FrameworkProxyDialog()
{
}

///////////////////////////////////////////////////////////////////////////
// protected member functions

// constructor helpers

void FrameworkProxyDialog::SetLayout()
{
  wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);
  topSizer->AddSpacer(10);
  const int leftFlags = wxALIGN_LEFT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int rightFlags = wxALIGN_RIGHT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;

  wxBoxSizer *loaderSizer = new wxBoxSizer(wxHORIZONTAL);
  loaderSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Proxy framework name")), 0, rightFlags, 2);
  topSizer->Add(loaderSizer, 0, wxALIGN_CENTER);
  topSizer->AddSpacer(4);

  wxBoxSizer *domainSizer = new wxBoxSizer(wxHORIZONTAL);
  domainSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Domain Name")), 0, rightFlags, 2);
  topSizer->Add(domainSizer, 0, wxALIGN_CENTER);
  topSizer->AddSpacer(4);

  wxBoxSizer *loginSizer = new wxBoxSizer(wxHORIZONTAL);
  loginSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Login")), 0, rightFlags, 2);
  topSizer->Add(loginSizer, 0, wxALIGN_CENTER);
  topSizer->AddSpacer(4);

  wxBoxSizer *passwordSizer = new wxBoxSizer(wxHORIZONTAL);
  passwordSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Password")), 0, rightFlags, 2);
  topSizer->Add(passwordSizer, 0, wxALIGN_CENTER);
  topSizer->AddSpacer(4);

  wxBoxSizer *pathSizer = new wxBoxSizer(wxHORIZONTAL);
  pathSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Path To proxy framework")), 0, rightFlags, 2);
  topSizer->Add(pathSizer, 0, wxALIGN_CENTER);
  topSizer->AddSpacer(4);

  wxBoxSizer *copiesSizer = new wxBoxSizer(wxHORIZONTAL);
  copiesSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Number of program copies (MPI)")), 0, rightFlags, 2);
  topSizer->Add(copiesSizer, 0, wxALIGN_CENTER);
  topSizer->AddSpacer(4);

  wxBoxSizer *mpiWhereSizer = new wxBoxSizer(wxHORIZONTAL);
  copiesSizer->Add(new wxStaticText(this, wxID_ANY, wxT("Execute on which CPUs or nodes?")), 0, rightFlags, 2);
  topSizer->Add(mpiWhereSizer, 0, wxALIGN_CENTER);
  topSizer->AddSpacer(4);

  wxBoxSizer *whereSizer = new wxBoxSizer(wxHORIZONTAL);
  wxStaticText* whereStaticText;

  SetAutoLayout(true);            // tell dialog to use sizer
  SetSizer(topSizer);            // actually set the sizer
  topSizer->Fit(this);           // set size to minimum size as calculated by the sizer
  topSizer->SetSizeHints(this);  // set size hints to honour minimum size
}

}
