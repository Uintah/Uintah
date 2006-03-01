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

  builderWindow->RedrawMiniCanvas();
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
        con = new Connection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), true);
	possibleConnections.push_back(con);
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
	possibleConnections.push_back(con);
      }
    }
  }
  if (possibleConnections.empty()) {
    return false;
  }
  Refresh();
  return true;
}

void NetworkCanvas::ClearPossibleConnections()
{
  for (unsigned int i = 0; i < possibleConnections.size(); i++) {
    Connection *c = possibleConnections[i];
    delete c;
  }
  possibleConnections.clear();
  Refresh();
}

void NetworkCanvas::HighlightConnection(const wxPoint& point)
{
  for (unsigned int i = 0; i < possibleConnections.size(); i++) {
    // use return value for???
    bool ret = possibleConnections[i]->IsMouseOver(point);
    if (ret == true) {
      possibleConnections[i]->HighlightConnection();
    }
  }

  for (unsigned int i = 0; i < connections.size(); i++) {
  }
  Refresh();
}

void NetworkCanvas::Clear()
{
  SSIDL::array1<sci::cca::ComponentID::pointer> cids;

  ComponentMap::iterator iter = components.begin();
  while (iter != components.end()) {
    ComponentIcon *ci = iter->second;
    components.erase(iter);
    iter = components.begin();

    //removeAllConnections(module);
    cids.push_back(ci->GetComponentInstance());

    ci->Show(false);
    delete ci;
  }
  int ret = builder->destroyInstances(cids, 0);
  // show warning if not all component instances were destroyed
}

void NetworkCanvas::OnLeftDown(wxMouseEvent& event)
{
}

void NetworkCanvas::OnLeftUp(wxMouseEvent& event)
{
}

void NetworkCanvas::OnMouseMove(wxMouseEvent& event)
{
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

  wxClientDC dc(this);
  DoPrepareDC(dc);

  // cache existing component icon positions, if any
  std::vector<wxRect> rects;
  GetComponentRects(rects);

  ComponentIcon* ci = new ComponentIcon(builder, BuilderWindow::GetNextID(), this, compID, x, y);
  components[compID->getInstanceName()] = ci;

  if (rects.empty()) {
    ci->Move(x, y);
    ci->Show(true);
    //builderWindow->RedrawMiniCanvas();
    Refresh();
    return;
  }

  wxPoint origin(x, y);
  CalcUnscrolledPosition(origin.x, origin.y, &origin.x, &origin.y);
// std::cerr << "NetworkCanvas::AddIcon(..) " << ci->GetComponentInstanceName() << " origin = (" << origin.x << ", " << origin.y << ")" << std::endl;

  wxSize size = ci->GetSize();
  const int ci_h = size.GetHeight();
  const int ci_w = size.GetWidth();

  wxSize displ = size + wxSize(x, y);
  int w_ = 0;
  int h_ = 0;
  GetVirtualSize(&w_, &h_); // want the size of the scrollable window area
  const int max_row = h_/displ.GetHeight();
  const int max_col = w_/displ.GetWidth();

  for (int icol = 0; icol < max_col; icol++) {
    for (int irow = 0; irow < max_row; irow++) {
      wxRect candidateRect(origin.x + (displ.GetWidth() * icol), origin.y + (displ.GetHeight() * irow), ci_w, ci_h);

// std::cerr << "\tcandidate rect: ("
//           << origin.x + (displ.GetWidth() * icol)
//           << ", "
//           << origin.y + (displ.GetHeight() * irow)
//           << ", "
//           << ci_w
//           << ", "
//           << ci_h
// 	  << ")" << std::endl;
      // check with all the viewable modules - can new icon be placed?
      // searching for intersections with existing icons
      bool intersects = false;

      for (std::vector<wxRect>::iterator iter = rects.begin(); iter != rects.end(); iter++) {
        intersects |= candidateRect.Intersects(*iter);
      }

      if (! intersects) {
	int x_ = 0, y_ = 0;
	CalcScrolledPosition(candidateRect.GetPosition().x, candidateRect.GetPosition().y, &x_, &y_);

// std::cerr << "\tmove to: ("
//           << x_
//           << ", "
//           << y_
// 	  << ")" << std::endl;

        ci->Move(x_, y_);
        ci->Show(true);

	wxRect windowRect = GetClientRect();
	if (! windowRect.Inside(x_, y_)) {
	  int xu = 0, yu = 0;
	  GetScrollPixelsPerUnit(&xu, &yu);
	  Scroll(x_/xu, y_/yu);
	}
	//builderWindow->RedrawMiniCanvas();
	Refresh();
        return;
      }
    }
  }

  ci->Move(max_row, max_col);
  ci->Show(true);
  //builderWindow->RedrawMiniCanvas();
  Refresh();
}

void NetworkCanvas::GetScrolledPosition(const wxPoint& p, wxPoint& position)
{
  wxClientDC dc(this);
  DoPrepareDC(dc);
  CalcScrolledPosition(p.x, p.y, &position.x, &position.y);
  wxRect windowRect = GetClientRect();
  if (! windowRect.Inside(position.x, position.y)) {
    int xu = 0, yu = 0;
    GetScrollPixelsPerUnit(&xu, &yu);
    Scroll(position.x/xu, position.y/yu);
  }
}

void NetworkCanvas::GetUnscrolledPosition(const wxPoint& p, wxPoint& position)
{
  wxClientDC dc(this);
  DoPrepareDC(dc);
  CalcUnscrolledPosition(p.x, p.y, &position.x, &position.y);
}

void NetworkCanvas::GetUnscrolledMousePosition(wxPoint& position)
{
  // DoPrepareDC must be called before CalcUnscrolledPosition
  wxClientDC dc(this);
  DoPrepareDC(dc);
  wxPoint mp = wxGetMousePosition();
  wxPoint stc = ScreenToClient(mp);
  CalcUnscrolledPosition(stc.x, stc.y, &position.x, &position.y);
}

ComponentIcon* NetworkCanvas::FindIconAtPointer(wxPoint& position)
{
  wxPoint wp;
  wxWindow* w = wxFindWindowAtPointer(wp);
  ComponentIcon* ci = dynamic_cast<ComponentIcon*>(w);
  if (ci) {
    GetUnscrolledPosition(wp, position);
    return ci;
  }
  return 0;
}

PortIcon* NetworkCanvas::FindPortIconAtPointer(wxPoint& position)
{
  wxPoint wp;
  wxWindow* w = wxFindWindowAtPointer(wp);
  PortIcon* pi = dynamic_cast<PortIcon*>(w);
  if (pi) {
    GetUnscrolledPosition(wp, position);
    return pi;
  }
  return 0;
}

// wxRect NetworkCanvas::GetIconRect(ComponentIcon* ci)
// {
//   return wxRect(0, 0, 0, 0);
// }

// compute and cache component icon rectangles
void NetworkCanvas::GetComponentRects(std::vector<wxRect>& rv)
{
  wxClientDC dc(this);
  DoPrepareDC(dc);
  for (ComponentMap::iterator iter = components.begin(); iter != components.end(); iter++) {
    ComponentIcon* ci_ = iter->second;
    wxPoint p = ci_->GetPosition();
    CalcUnscrolledPosition(p.x, p.y, &p.x, &p.y);
    rv.push_back(wxRect(p, ci_->GetSize()));
  }
}

///////////////////////////////////////////////////////////////////////////
// protected constructor and member functions

NetworkCanvas::NetworkCanvas() : ID_MENU_POPUP(BuilderWindow::GetNextID())
{
  Init();
}

void NetworkCanvas::Init()
{
//   movingIcon = 0;
//   connectingIcon = 0;
}

// called from within paint event handler only!
wxRect NetworkCanvas::GetClientRect()
{
  // We need to shift the client rectangle to take into account
  // scrolling, converting device to logical coordinates.
  //
  // Device context should have been prepared for scrolling in paint event handler.
   wxRect windowRect(wxPoint(0, 0), GetClientSize());
   CalcUnscrolledPosition(windowRect.x, windowRect.y, &windowRect.x, &windowRect.y);
   //std::cerr << "NetworkCanvas::GetClientRect() rect=(" << windowRect.x << ", " << windowRect.y << ", " << windowRect.width << ", " << windowRect.height <<  ")" << std::endl;
  return windowRect;
}

///////////////////////////////////////////////////////////////////////////
// private member functions

}
