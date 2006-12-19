/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
#include <wx/gdicmn.h>

#include <CCA/Components/GUIBuilder/PortIcon.h>
#include <CCA/Components/GUIBuilder/BuilderWindow.h>
#include <CCA/Components/GUIBuilder/ComponentIcon.h>
#include <CCA/Components/GUIBuilder/NetworkCanvas.h>

#include <string>

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(PortIcon, wxWindow)
  EVT_PAINT(PortIcon::OnPaint)
  EVT_LEFT_DOWN(PortIcon::OnLeftDown)
  EVT_LEFT_UP(PortIcon::OnLeftUp)
  EVT_RIGHT_UP(PortIcon::OnRightClick) // show compatible components menu
  //EVT_MIDDLE_DOWN(PortIcon::OnMouseDown)
  EVT_MOTION(PortIcon::OnMouseMove)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(PortIcon, wxWindow)

PortIcon::PortIcon(const sci::cca::GUIBuilder::pointer& bc,
                   ComponentIcon* parent,
                   wxWindowID id,
                   GUIBuilder::PortType pt,
                   const std::string& name,
                   const std::string& componentModel,
                   const std::string& sidlType)
  : builder(bc), parent(parent), portType(pt),
    name(name), componentModel(componentModel), sidlType(sidlType),
    ID_MENU_POPUP(BuilderWindow::GetNextID())
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
  void* c = builder->getPortColor(sidlType);
  pColor = wxColor(*((wxColor*) c));

  //need database of port types/colours
  if (portType == GUIBuilder::Uses) {
    hColor = wxTheColourDatabase->Find("GREEN");
  } else {
    hColor = wxTheColourDatabase->Find("RED");
  }
  SetBackgroundColour(pColor);
  SetToolTip(wxT(sidlType + " " + this->name));

  return true;
}

void PortIcon::OnLeftDown(wxMouseEvent& event)
{
  if (portType == GUIBuilder::Uses) {
    parent->GetCanvas()->ShowPossibleConnections(this);
  }
}

void PortIcon::OnLeftUp(wxMouseEvent& event)
{
  // canvas draw connection
  parent->GetCanvas()->Connect(this);
}

void PortIcon::OnMouseMove(wxMouseEvent& WXUNUSED(event))
{
  NetworkCanvas *canvas = parent->GetCanvas();
  wxPoint mp;
  canvas->GetUnscrolledMousePosition(mp);
  canvas->HighlightConnection(mp);
}

void PortIcon::OnRightClick(wxMouseEvent& event)
{
  // show component menu w/ compatible ports
  wxMenu *m = new wxMenu();
  m->Append(wxID_ANY, wxT("Port Icon Menu Item"));

  // fill with compatible component types -> handle bridging???
  PopupMenu(m, event.GetPosition());
}

void PortIcon::OnPaint(wxPaintEvent& event)
{
  wxPaintDC dc(this);

  wxRect windowRect(wxPoint(0, 0), GetClientSize());
  if (IsExposed(windowRect)) {
    wxPen* pen = wxThePenList->FindOrCreatePen(pColor, 1, wxSOLID);
    wxBrush* brush = wxTheBrushList->FindOrCreateBrush(pColor, wxSOLID);
    dc.SetPen(*pen);
    dc.SetBrush(*brush);

    dc.DrawRectangle(windowRect);
  }
  wxRect hRect;
  if (portType == GUIBuilder::Uses) {
    hRect = wxRect(0, 0, HIGHLIGHT_WIDTH, PORT_HEIGHT);
  } else {
    hRect = wxRect(PORT_WIDTH - HIGHLIGHT_WIDTH, 0, HIGHLIGHT_WIDTH, PORT_HEIGHT);
  }

  if (IsExposed(hRect)) {
    wxPen* pen = wxThePenList->FindOrCreatePen(hColor, 1, wxSOLID);
    wxBrush* brush = wxTheBrushList->FindOrCreateBrush(hColor, wxSOLID);
    dc.SetPen(*pen);
    dc.SetBrush(*brush);

    dc.DrawRectangle(hRect);
  }
}


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
