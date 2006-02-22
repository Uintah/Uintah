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

#include <wx/dcbuffer.h>
#include <wx/scrolwin.h>

#include <CCA/Components/Builder/NetworkCanvas.h>
#include <CCA/Components/Builder/BuilderWindow.h>
#include <CCA/Components/Builder/ComponentIcon.h>
#include <CCA/Components/Builder/PortIcon.h>
#include <CCA/Components/Builder/Connection.h>

#include <iostream>

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(NetworkCanvas, wxScrolledWindow)
  EVT_PAINT(NetworkCanvas::OnPaint)
  EVT_ERASE_BACKGROUND(NetworkCanvas::OnEraseBackground)
  EVT_LEFT_DOWN(NetworkCanvas::OnLeftDown)
  EVT_LEFT_UP(NetworkCanvas::OnLeftUp)
  EVT_RIGHT_UP(NetworkCanvas::OnRightClick) // show popup menu
  EVT_MOTION(NetworkCanvas::OnMouseMove)
// EVT_MIDDLE_DOWN(NetworkCanvas::OnLeftDown) // ignore middle clicks on canvas for now...
  EVT_SCROLLWIN(NetworkCanvas::OnScroll)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(NetworkCanvas, wxScrolledWindow)

NetworkCanvas::NetworkCanvas(const sci::cca::BuilderComponent::pointer& bc, BuilderWindow* bw, wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size) : builder(bc), ID_MENU_POPUP(BuilderWindow::GetNextID()), builderWindow(bw)
{
  Init();
  Create(parent, id, pos, size);
}

NetworkCanvas::~NetworkCanvas()
{
}

bool NetworkCanvas::Create(wxWindow *parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
  if (!wxScrolledWindow::Create(parent, id, pos, size, style)) {
    return false;
  }

  handCursor = new wxCursor(wxCURSOR_HAND);
  arrowCursor = new wxCursor(wxCURSOR_ARROW);

  SetBackgroundStyle(wxBG_STYLE_COLOUR);
  SetBackgroundColour(BuilderWindow::BACKGROUND_COLOUR);
  SetVirtualSize(DEFAULT_VWIDTH, DEFAULT_VHEIGHT);
  SetScrollRate(DEFAULT_SCROLLX, DEFAULT_SCROLLY);
  SetCursor(*arrowCursor);

  return true;
}


void NetworkCanvas::OnPaint(wxPaintEvent& event)
{
//   wxScrolledWindow::OnPaint(event);

  wxBufferedPaintDC dc(this, wxBUFFER_VIRTUAL_AREA);
  // Shifts the device origin so we don't have to worry
  // about the current scroll position ourselves.
  PrepareDC(dc);

  PaintBackground(dc);
  OnDraw(dc);
}

void NetworkCanvas::OnDraw(wxDC& dc)
{
  DrawConnections(dc);
}

// Paint the background - from wxWidgets book
void NetworkCanvas::PaintBackground(wxDC& dc)
{
  dc.Clear();
  wxColour backgroundColour = GetBackgroundColour();
  if (! backgroundColour.Ok()) {
    backgroundColour =
      wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE);
  }

  dc.SetBrush(wxBrush(backgroundColour));
  dc.SetPen(wxPen(backgroundColour, 1));

  wxRect windowRect = GetClientRect();
  dc.DrawRectangle(windowRect);
}

void NetworkCanvas::DrawConnections(wxDC& dc)
{
  //std::cerr << "NetworkCanvas::DrawConnections(..)" << std::endl;
  for (unsigned int i = 0; i < possibleConnections.size(); i++) {
    possibleConnections[i]->OnDraw(dc);
  }

  for (std::vector<Connection*>::iterator it = connections.begin(); it != connections.end(); it++) {
  }
}

bool NetworkCanvas::ShowPossibleConnections(PortIcon* port)
{
std::cerr << "NetworkCanvas::ShowPossibleConnections(..): " << port->GetPortName() << std::endl;
  ComponentIcon* pCI = port->GetParent();
  Builder::PortType type = port->GetPortType();
  for (ComponentMap::iterator iter = components.begin(); iter != components.end(); iter++) {
    ComponentIcon* ci_ = iter->second;
    SSIDL::array1<std::string> portArray;
    builder->getCompatiblePortList(pCI->GetComponentInstance(), port->GetPortName(), ci_->GetComponentInstance(), portArray);

    for (unsigned int j = 0; j < portArray.size(); j++) {
      Connection *con;
      if (type == Builder::Uses) {
        PortIcon *pUses = pCI->GetPortIcon(port->GetPortName());
        if (pUses == 0) {
          std::cerr << "Error: could not locate port " << port->GetPortName() << std::endl;
          continue;
        }
        PortIcon *pProvides = ci_->GetPortIcon(portArray[j]);
        if (pProvides == 0) {
          std::cerr << "Error: could not locate port " << portArray[j] << std::endl;
          continue;
        }
        con = new Connection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), this);
      } else {
        PortIcon *pUses = ci_->GetPortIcon(portArray[j]);
        if (pUses == 0) {
          std::cerr << "Error: could not locate port " << portArray[j] << std::endl;
          return false;
        }
        PortIcon *pProvides = pCI->GetPortIcon(port->GetPortName());
        if (pProvides == 0) {
          std::cerr << "Error: could not locate port " << port->GetPortName() << std::endl;
          return false;
        }
        con = new Connection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), true);
      }
      //con->show();
      possibleConnections.push_back(con);
      Refresh();
      //std::cerr<<portList[j]<<std::endl;
    }
  }
  if (possibleConnections.size() > 0) {
    return true;
  }
  return false;
}

void NetworkCanvas::ClearPossibleConnections()
{
  possibleConnections.clear();
  Refresh();
}

// Connection* NetworkCanvas::FindConnectionIntersection(wxPoint p)
// {
//   // compute bounding rectangle for connections, possible connections
// }

void NetworkCanvas::OnLeftDown(wxMouseEvent& event)
{
//   if (movingIcon || connectingIcon) {
//     return;
//   }
  if (movingIcon) {
    wxClientDC dc(this);
    DoPrepareDC(dc);
    movingStart = event.GetLogicalPosition(dc);

  wxPoint p = event.GetPosition();
  std::cerr << "NetworkCanvas::OnLeftDown(..) event pos=(" << p.x << ", " << p.y << ")"
            << std::endl
	    << "logical event pos=(" << movingStart.x << ", " << movingStart.y << ")" << std::endl;
    //CaptureMouse();
  }

}

void NetworkCanvas::OnLeftUp(wxMouseEvent& event)
{
  if (movingIcon) {
    movingIcon->Show(false);
    wxClientDC dc(this);
    DoPrepareDC(dc);
    wxPoint p = event.GetLogicalPosition(dc);
    CalcScrolledPosition(p.x, p.y, &p.x, &p.y);
    movingIcon->Move(p.x, p.y);
    movingIcon->Show(true);
    builderWindow->RedrawMiniCanvas();
    SetMovingIcon(0);
    //ReleaseMouse();
  }
}

void NetworkCanvas::OnMouseMove(wxMouseEvent& event)
{
  // see wxMouseEvent::LeftIsDown and wxMouseEvent::Dragging
  if (event.LeftIsDown() && event.Dragging() && movingIcon) {

    wxClientDC dc(this);
    DoPrepareDC(dc);
    wxPoint p = event.GetLogicalPosition(dc);

  wxPoint pp = event.GetPosition();
  std::cerr << "NetworkCanvas::OnMouseMove(..) event pos=(" << pp.x << ", " << pp.y << ")"
            << std::endl
	    << "logical event pos=(" << movingStart.x << ", " << movingStart.y << ")" << std::endl;
    //SetCursor(*handCursor);

    // set tolerance of about 2 pixels

    movingIcon->Show(false);

    int dx = 0, dy = 0, mi_x_ = 0, mi_y_ = 0;
    CalcUnscrolledPosition(movingIcon->GetPosition().x, movingIcon->GetPosition().y, &mi_x_, &mi_y_);
    int newX = mi_x_ + p.x - movingStart.x;
    int newY = mi_y_ + p.y - movingStart.y;
    wxPoint topLeft(newX, newY);

    // adjust for canvas boundaries
    if (topLeft.x < 0) {
      //newX -= topLeft.x; // 0?
      newX = 0;
      dx -= 1;
    }

    if (topLeft.y < 0) {
      //newY -= topLeft.y; // 0?
      newY = 0;
      dy -= 1;
    }

    int cw = GetVirtualSize().GetWidth();
    int mw = movingIcon->GetVirtualSize().GetWidth();

    if (topLeft.x > cw - mw) {
      newX -= topLeft.x - (cw - mw);
      dx = 1;
    }

    int ch = GetVirtualSize().GetHeight();
    int mh = movingIcon->GetVirtualSize().GetHeight();

    if (topLeft.y > ch - mh) {
      newY -= topLeft.y - (ch - mh);
      dy = 1;
    }

    movingStart = p;


    CalcScrolledPosition(newX, newY, &newX, &newY);
    movingIcon->Move(newX, newY);
    //WarpPointer(newX, newY); // move mouse pointer
    movingIcon->Show(true);
    // reset moving icon connections
    Refresh();
    builderWindow->RedrawMiniCanvas();

    wxRect windowRect = GetClientRect();
    if (! windowRect.Inside(newX + mw, newY + mh)) {
      int xu = 0, yu = 0;
      GetScrollPixelsPerUnit(&xu, &yu);
      Scroll(newX/xu, newY/yu);
    }

    //SetCursor(*arrowCursor);
  } else {
    wxScrolledWindow();
  }
}

void NetworkCanvas::OnRightClick(wxMouseEvent& event)
{
  wxMenu *m = new wxMenu();
  m->Append(wxID_ANY, wxT("Canvas Menu Item"));

  PopupMenu(m, event.GetPosition());
}

void NetworkCanvas::OnScroll(wxScrollWinEvent& event)
{
  wxScrolledWindow::OnScroll(event);
  builderWindow->RedrawMiniCanvas();
}

void NetworkCanvas::AddIcon(sci::cca::ComponentID::pointer& compID)
{
  const int x = 10, y = 10;

  // cache existing component icon positions, if any
  std::vector<wxRect> rects;
  GetComponentRects(rects);

  ComponentIcon* ci = new ComponentIcon(builder, BuilderWindow::GetNextID(), this, compID, x, y);
  components[compID->getInstanceName()] = ci;

  wxPoint origin(x, y);
  CalcUnscrolledPosition(origin.x, origin.y, &origin.x, &origin.y); // get logical coordinates
  wxSize size = ci->GetSize();
  const int ci_h = size.GetHeight();
  const int ci_w = size.GetWidth();

  wxSize displ = size + wxSize(x, y);

  int w_ = 0;
  int h_ = 0;
  GetVirtualSize(&w_, &h_); // want the size of the scrollable window area
  const int max_row = h_/displ.GetHeight();
  const int max_col = w_/displ.GetWidth();

  if (rects.empty()) {
    ci->Move(x, y);
    ci->Show(true);
    builderWindow->RedrawMiniCanvas();
    Refresh();
    return;
  }

  for (int icol = 0; icol < max_col; icol++) {
    for (int irow = 0; irow < max_row; irow++) {

      wxRect candidateRect(origin.x + (displ.GetWidth() * icol), origin.y + (displ.GetHeight() * irow), ci_w, ci_h);

      // check with all the viewable modules - can new icon be placed?
      // searching for intersections with existing icons
      bool intersects = false;

      for (std::vector<wxRect>::iterator iter = rects.begin(); iter != rects.end(); iter++) {
        intersects |= candidateRect.Intersects(*iter);
      }

      if (!intersects) {
	int x_ = 0, y_ = 0;
	CalcScrolledPosition(candidateRect.GetPosition().x, candidateRect.GetPosition().y, &x_, &y_);
        ci->Move(x_, y_);
        ci->Show(true);
        Refresh();

	wxRect windowRect = GetClientRect();
	if (! windowRect.Inside(x_, y_)) {
	  int xu = 0, yu = 0;
	  GetScrollPixelsPerUnit(&xu, &yu);
	  Scroll(x_/xu, y_/yu);
	}
	builderWindow->RedrawMiniCanvas();
	Refresh();
        return;
      }
    }
  }

  ci->Move(max_row, max_col);
  ci->Show(true);
  builderWindow->RedrawMiniCanvas();
  Refresh();
}

wxRect NetworkCanvas::GetClientRect()
{
  wxRect windowRect(wxPoint(0, 0), GetClientSize());
  // We need to shift the client rectangle to take into account
  // scrolling, converting device to logical coordinates
  CalcUnscrolledPosition(windowRect.x, windowRect.y, &windowRect.x, &windowRect.y);
  return windowRect;
}

wxPoint NetworkCanvas::GetIconPosition(ComponentIcon* ci)
{
  if (!ci) { return wxPoint(0, 0); }

  for (ComponentMap::iterator iter = components.begin(); iter != components.end(); iter++) {
    if (iter->first == ci->GetComponentInstanceName()) {
      // calc. component icon position rel. to parent...
      wxPoint p = iter->second->GetPosition();
      CalcUnscrolledPosition(p.x, p.y, &p.x, &p.y);
      return p;
    }
  }

  return wxPoint(0, 0);
}

wxRect NetworkCanvas::GetIconRect(ComponentIcon* ci)
{
  return wxRect(0, 0, 0, 0);
}

// compute and cache component icon rectangles
void NetworkCanvas::GetComponentRects(std::vector<wxRect>& rv)
{
  for (ComponentMap::iterator iter = components.begin(); iter != components.end(); iter++) {
    ComponentIcon* ci_ = iter->second;
    wxPoint p = ci_->GetPosition();
    CalcUnscrolledPosition(p.x, p.y, &p.x, &p.y);
    rv.push_back(wxRect(p, ci_->GetSize()));
  }
}

///////////////////////////////////////////////////////////////////////////
// protected constructor

NetworkCanvas::NetworkCanvas() : ID_MENU_POPUP(BuilderWindow::GetNextID())
{
  Init();
}

void NetworkCanvas::Init()
{
  movingIcon = 0;
  connectingIcon = 0;
}

///////////////////////////////////////////////////////////////////////////
// private member functions

}
