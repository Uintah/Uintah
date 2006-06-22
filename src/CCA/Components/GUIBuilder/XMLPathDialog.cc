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
 * XMLPathDialog.cc
 *
 * Written by:
 *  Ayla Khan
 *  SCI
 *  University of Utah
 *  November 2004
 *  Ported to wxWidgets May 2006
 *
 */

#include <wx/sizer.h>
#include <wx/combobox.h>
#include <wx/button.h>
#include <wx/dirdlg.h>
#include <wx/valtext.h>

#include <CCA/Components/GUIBuilder/XMLPathDialog.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>
#include <include/sci_metacomponents.h>

namespace GUIBuilder {
//using namespace SCIRun;

BEGIN_EVENT_TABLE(XMLPathDialog, wxDialog)
  EVT_BUTTON(ID_BUTTON_FILE_PATH, XMLPathDialog::OnFilePath)
END_EVENT_TABLE()

XMLPathDialog::XMLPathDialog(wxWindow* parent,
                             wxWindowID id,
                             const wxString& caption,
                             const wxPoint& pos,
                             const wxSize& size,
                             long style)
  : wxDialog(parent, id, caption, pos, size, style)
{
  // The framework should have a way to query available component models
  // from a factory (available through the builder).
  // A dialog shouldn't know this much about framework's component models.
  componentModels.Add("CCA");
  // SCIRun Dataflow is discovered using SCIRun packages.
#if HAVE_BABEL
  componentModels.Add("Babel");
#endif
#if HAVE_VTK
  componentModels.Add("VTK");
#endif
#if HAVE_TAO
  componentModels.Add("CORBA");
  componentModels.Add("Tao");
#endif

  SetLayout();
}


///////////////////////////////////////////////////////////////////////////
// event handlers

void XMLPathDialog::OnFilePath(wxCommandEvent& event)
{
  wxDirDialog dDialog(this, wxT("Choose component XML path"), wxT(GUIBuilder::DEFAULT_SRC_DIR));
  if (dDialog.ShowModal() == wxID_OK) {
    fpTextCtrl->SetValue(dDialog.GetPath());
  }
}

///////////////////////////////////////////////////////////////////////////
// protected member functions

// constructor helpers

void XMLPathDialog::SetLayout()
{
  // Validators (customized):
  // * should check that directories exist
  // * component models shouldn't start with digits

  wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);
  topSizer->AddSpacer(10);
  const int leftFlags = wxALIGN_LEFT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int rightFlags = wxALIGN_RIGHT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;

  wxBoxSizer *fpSizer = new wxBoxSizer(wxHORIZONTAL);
  fpButton = new wxButton(this, ID_BUTTON_FILE_PATH, wxT("Get XML directory"));
  fpSizer->Add(fpButton, 0, leftFlags, 2);
  setTextCtrl(*fpSizer, centerFlags);
  cmComboBox = new wxComboBox(this, wxID_ANY, componentModels[0], wxDefaultPosition,
                              wxSize(100, wxDefaultSize.GetHeight()), componentModels,
                              wxCB_DROPDOWN|wxCB_SORT|wxCB_READONLY);
  fpSizer->Add(cmComboBox, 0, rightFlags, 2);
  cmComboBox->SetToolTip(wxT("Select component model for component classes described in XML file."));
  topSizer->Add(fpSizer, 1, wxALIGN_CENTER, 0);
  topSizer->AddSpacer(10);

  wxBoxSizer *okCancelSizer = new wxBoxSizer(wxHORIZONTAL);
  okCancelSizer->Add( new wxButton( this, wxID_OK, wxT("&OK") ), 1, leftFlags, 4 );
  okCancelSizer->Add( new wxButton( this, wxID_CANCEL, wxT("&Cancel") ), 1, rightFlags, 4 );
  topSizer->Add(okCancelSizer, 1, wxALIGN_CENTER, 0);
  topSizer->AddSpacer(10);

  SetAutoLayout(true);
  SetSizer(topSizer);

  topSizer->Fit(this);
  topSizer->SetSizeHints(this);
}

///////////////////////////////////////////////////////////////////////////
// private member functions

void XMLPathDialog::setTextCtrl(wxBoxSizer& sizer, const int flags)
{

  fpTextCtrl = new wxTextCtrl(this, wxID_ANY, wxT(""), wxDefaultPosition,
                              wxSize(320, wxDefaultSize.GetHeight()));
  fpTextCtrl->SetToolTip(wxT("Use the 'Get XML directory' button to set the directory with the component description XML file or enter the directory path here."));

  wxTextValidator val(wxFILTER_EXCLUDE_LIST, &filePath);
  wxArrayString valExcludes;
  valExcludes.Add("!");
  valExcludes.Add("%");
  val.SetExcludes(valExcludes);
  fpTextCtrl->SetValidator(val);
  sizer.Add(fpTextCtrl, 1, flags, 2);
}

}
