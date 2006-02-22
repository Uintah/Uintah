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
 * PortIcon.cc
 *
 */



//#include <wx/region.h>
//#include <wx/dc.h>

#include <wx/dcbuffer.h>
#include <wx/gdicmn.h> // color database

#include <CCA/Components/Builder/PortIcon.h>
#include <CCA/Components/Builder/ComponentIcon.h>
#include <CCA/Components/Builder/NetworkCanvas.h>

#include <string>

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(PortIcon, wxWindow)
  EVT_LEFT_DOWN(PortIcon::OnLeftDown)
  EVT_LEFT_UP(PortIcon::OnLeftUp)
  EVT_RIGHT_UP(PortIcon::OnRightClick) // show compatible components menu
// EVT_MIDDLE_DOWN(PortIcon::OnMouseDown)
  EVT_MOTION(PortIcon::OnMouseMove)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(PortIcon, wxWindow)

PortIcon::PortIcon(ComponentIcon* parent, wxWindowID id, Builder::PortType pt, const std::string& name) : parent(parent), type(pt), name(name), connecting(false), ID_MENU_POPUP(BuilderWindow::GetNextID())
{
  Init();
  Create(parent, id, wxT(name));

}

PortIcon::~PortIcon()
{
}

bool PortIcon::Create(wxWindow *parent, wxWindowID id, const wxString &name)
{
  if (! wxWindow::Create(parent, id, wxDefaultPosition, wxSize(PORT_WIDTH, PORT_HEIGHT), wxNO_BORDER, name)) {
    return false;
  }

  //need database of port types/colours
  if (type == Builder::Uses) {
    pColour = wxColour(wxTheColourDatabase->Find("FIREBRICK"));
    hColour = wxColour(wxTheColourDatabase->Find("GREEN"));
  } else {
    pColour = wxColour(wxTheColourDatabase->Find("SLATE BLUE"));
    hColour = wxColour(wxTheColourDatabase->Find("RED"));
  }
  SetBackgroundColour(pColour);
  //hRect = wxRect(, , HIGHLIGHT_WIDTH, PORT_HEIGHT);

  SetToolTip(name);

  return true;
}

void PortIcon::OnLeftDown(wxMouseEvent& event)
{
  if (type == Builder::Uses) {
    connecting = parent->GetCanvas()->ShowPossibleConnections(this);
  }
}

void PortIcon::OnLeftUp(wxMouseEvent& event)
{
  parent->GetCanvas()->ClearPossibleConnections();
}

void PortIcon::OnMouseMove(wxMouseEvent& event)
{
  if (connecting) {
    NetworkCanvas *canvas = parent->GetCanvas();
    wxPoint p = event.GetPosition();
    wxPoint pp = wxGetMousePosition();
    canvas->CalcUnscrolledPosition(pp.x, pp.y, &pp.x, &pp.y);
    std::cerr << "PortIcon::OnMouseMove(..): (" << p.x << ", " << p.y << ")" << std::endl
	      << "Canvas position: (" << pp.x << ", " << pp.y << ")" << std::endl;
    // figure out which connection we're over and change to highlight colour
  }
}

void PortIcon::OnRightClick(wxMouseEvent& event)
{
  // show component menu w/ compatible ports
  wxMenu *m = new wxMenu();
  m->Append(wxID_ANY, wxT("Port Icon Menu Item"));

  // fill with compatible component types -> handle bridging???
  PopupMenu(m, event.GetPosition());
}

// void PortIcon::OnDraw(wxDC& dc)
// {
// }

///////////////////////////////////////////////////////////////////////////
// protected constructor and member functions

PortIcon::PortIcon() : ID_MENU_POPUP(BuilderWindow::GetNextID())
{
  Init();
}

void PortIcon::Init()
{
}

}
