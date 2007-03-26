/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
 * ViewerDialog.h
 *
 * Written by:
 *
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *
 *
 */

#include <CCA/Components/Viewer/ViewerDialog.h>
#include <CCA/Components/Viewer/ViewerWindow.h>

namespace Viewer {

ViewerDialog::ViewerDialog(wxWindow* parent,
                           const SSIDL::array1<double> &nodes1d,
                           const SSIDL::array1<int> &triangles,
                           const SSIDL::array1<double> &solution,
                           wxWindowID id,
                           const wxString &caption,
                           const wxPoint &pos,
                           const wxSize &size,
                           long style)
  : wxDialog(parent, id, caption, pos, size, style)
{
  SetLayout(nodes1d, triangles, solution);
}

ViewerDialog::~ViewerDialog()
{
  delete colorMap;
  delete viewerWindow;
}

///////////////////////////////////////////////////////////////////////////
// protected member functions

// constructor helpers

void ViewerDialog::SetLayout(const SSIDL::array1<double> &nodes1d, const SSIDL::array1<int> &triangles, const SSIDL::array1<double> &solution)
{
  wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);
  topSizer->AddSpacer(10);
  const int leftFlags = wxALIGN_LEFT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int rightFlags = wxALIGN_RIGHT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  //const int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;

  wxBoxSizer *vwSizer = new wxBoxSizer(wxHORIZONTAL);
  colorMap = new Colormap(this, wxT("Color"));
  viewerWindow = new ViewerWindow(this, colorMap, nodes1d, triangles, solution);
  vwSizer->Add(viewerWindow, 1, leftFlags, 2);
  vwSizer->Add(colorMap, 0, rightFlags, 2);
  topSizer->Add(vwSizer, 1, wxALIGN_CENTER, 0);
  topSizer->AddSpacer(10);

  wxBoxSizer *okCancelSizer = new wxBoxSizer(wxHORIZONTAL);
  okCancelSizer->Add( new wxButton( this, wxID_OK, wxT("&OK") ), 1, leftFlags, 4 );
  okCancelSizer->Add( new wxButton( this, wxID_CANCEL, wxT("&Cancel") ), 1, rightFlags, 4 );
  topSizer->Add(okCancelSizer, 1, wxALIGN_CENTER, 0);
  topSizer->AddSpacer(10);

  SetAutoLayout(true);
  SetSizer(topSizer);

  //topSizer->Fit(this);
  topSizer->SetSizeHints(this);
}

}
