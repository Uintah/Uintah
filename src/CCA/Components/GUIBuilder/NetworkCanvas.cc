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
 *  NetworkCanvas.cc:
 *
 *  Written by (NetworkCanvasView.cc):
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *  Modified by:
 *   Keming Zhang
 *   March 2002
 *  Ported to wxWidgets:
 *   Ayla Khan
 *   January 2006
 *
 *  Modifications:
 *   The wxWidgets toolkit does not provide canvas classes,
 *   so QCanvasView's functionality is approximated in this class.
 */

#include <wx/dcbuffer.h>
#include <wx/scrolwin.h>
#include <wx/event.h>
#include <wx/menu.h>

#include <sci_metacomponents.h>

#include <CCA/Components/GUIBuilder/NetworkCanvas.h>
#include <CCA/Components/GUIBuilder/BuilderWindow.h>
#include <CCA/Components/GUIBuilder/ComponentIcon.h>
#include <CCA/Components/GUIBuilder/PortIcon.h>
#include <CCA/Components/GUIBuilder/Connection.h>
#include <CCA/Components/GUIBuilder/BridgeConnection.h>

#include <iostream>

#ifndef DEBUG
#  define DEBUG 1
#endif

namespace GUIBuilder {

using namespace SCIRun;
typedef BuilderWindow::MenuMap MenuMap;

BEGIN_EVENT_TABLE(NetworkCanvas, wxScrolledWindow)
  EVT_PAINT(NetworkCanvas::OnPaint)
  EVT_ERASE_BACKGROUND(NetworkCanvas::OnEraseBackground)
  EVT_LEFT_DOWN(NetworkCanvas::OnLeftDown)
  EVT_LEFT_UP(NetworkCanvas::OnLeftUp)
  EVT_RIGHT_UP(NetworkCanvas::OnRightClick) // show popup menu
  EVT_MOTION(NetworkCanvas::OnMouseMove)
  EVT_MIDDLE_UP(NetworkCanvas::OnMiddleClick)
  EVT_SCROLLWIN(NetworkCanvas::OnScroll)
  EVT_MENU(ID_MENU_CLEAR, NetworkCanvas::OnClear)
  EVT_MENU(ID_MENU_DISCONNECT, NetworkCanvas::OnDisconnect)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(NetworkCanvas, wxScrolledWindow)

NetworkCanvas::NetworkCanvas(const sci::cca::GUIBuilder::pointer& bc,
                             BuilderWindow* bw,
                             wxWindow* parent,
                             wxWindowID id,
                             const wxPoint& pos,
                             const wxSize& size)
  : builder(bc), builderWindow(bw), movingIcon(0), selectedConnection(0)
{
  Init();
  Create(parent, id, pos, size);
}

NetworkCanvas::~NetworkCanvas()
{
}

bool NetworkCanvas::Create(wxWindow *parent,
                           wxWindowID id,
                           const wxPoint& pos,
                           const wxSize& size,
                           long style)
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

  SetMenus();

  return true;
}

///////////////////////////////////////////////////////////////////////////
// event handlers

void NetworkCanvas::OnLeftDown(wxMouseEvent& event)
{
//   wxPoint p = event.GetPosition();
//   wxPoint pp;
//   GetUnscrolledPosition(p, pp);
//   wxPoint mp;
//   GetUnscrolledMousePosition(mp);
// #if DEBUG
//   std::cerr << "NetworkCanvas::OnLeftDown(..):" << std::endl
//             << "\t event position=(" << p.x << ", " << p.y << ")" << std::endl
//             << "\t unscrolled event position=(" << pp.x << ", " << pp.y << ")" << std::endl
//             << "\t unscrolled mouse position=(" << mp.x << ", " << mp.y << ")" << std::endl
//             << std::endl;
//   std::cerr << "NetworkCanvas::OnLeftDown(..)" << std::endl;
//   if (movingIcon) {
//     std::cerr << "\tmoving icon: " << movingIcon->GetComponentInstanceName() << std::endl;
//   }
// #endif
}

void NetworkCanvas::OnLeftUp(wxMouseEvent& event)
{
// #if DEBUG
//   std::cerr << "NetworkCanvas::OnLeftUp(..)" << std::endl;
// #endif
//   if (movingIcon) {
// #if DEBUG
//     std::cerr << "\tmoving icon: " << movingIcon->GetComponentInstanceName() << std::endl;
// #endif
//     movingIcon->OnLeftUp(event);
//     movingIcon = 0;
//   }
}

void NetworkCanvas::OnMouseMove(wxMouseEvent& event)
{
//   if (movingIcon) {
// #if DEBUG
//     std::cerr << "NetworkCanvas::OnMouseMove(..)" << std::endl;
//     std::cerr << "\tmoving icon: " << movingIcon->GetComponentInstanceName() << std::endl;
// #endif
//     movingIcon->OnMouseMove(event);
//     //wxPoint p = event.GetPosition();
//     //WarpPointer(p.x, p.y);
//   }
}

void NetworkCanvas::OnRightClick(wxMouseEvent& event)
{
  wxPoint mp;
  GetUnscrolledMousePosition(mp);
  selectedConnection = 0;
  for (ConnectionMap::iterator iter = connections.begin(); iter != connections.end(); iter++) {
    if (iter->second->IsMouseOver(mp)) {
      selectedConnection = iter->second;
      PopupMenu(connectionPopupMenu, mp);
      return;
    }
  }

  PopupMenu(popupMenu, event.GetPosition());
}

void NetworkCanvas::OnMiddleClick(wxMouseEvent& event)
{
  wxPoint mp;
  GetUnscrolledMousePosition(mp);
  for (ConnectionMap::iterator iter = connections.begin(); iter != connections.end(); iter++) {
    if (iter->second->IsMouseOver(mp)) {
      Disconnect(iter);
      return;
    }
  }
}

void NetworkCanvas::OnScroll(wxScrollWinEvent& event)
{
  wxScrolledWindow::OnScroll(event);
  builderWindow->RedrawMiniCanvas();
}

void NetworkCanvas::OnClear(wxCommandEvent& event)
{
  Clear();
  builderWindow->RedrawMiniCanvas();
}

void NetworkCanvas::OnDisconnect(wxCommandEvent& event)
{
  if (selectedConnection) {
      ConnectionMap::iterator lb = connections.lower_bound(selectedConnection->GetUsesPortIcon());
      ConnectionMap::iterator ub = connections.upper_bound(selectedConnection->GetUsesPortIcon());

      ConnectionMap::iterator cIter = lb;
      while (cIter != ub) {
        Connection *c = cIter->second;
        if (selectedConnection->GetConnectionID() == c->GetConnectionID()) {
          Disconnect(cIter);
          return;
        }
        cIter++;
      }
  }
}

void NetworkCanvas::OnPaint(wxPaintEvent& event)
{
  wxBufferedPaintDC dc(this);
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

///////////////////////////////////////////////////////////////////////////
// canvas functions

// Paint the background - from wxWidgets book
void NetworkCanvas::PaintBackground(wxDC& dc)
{
  dc.Clear();
  wxColor backgroundColour = GetBackgroundColour();
  if (! backgroundColour.Ok()) {
    backgroundColour =
      wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE);
  }

  dc.SetBrush(wxBrush(backgroundColour));
  dc.SetPen(wxPen(backgroundColour, 1));

  wxRect windowRect = GetClientRect();
  dc.DrawRectangle(windowRect);
}

void NetworkCanvas::Connect(PortIcon* usesPortIcon)
{
  ConnectionMap::iterator lb = possibleConnections.lower_bound(usesPortIcon);
  ConnectionMap::iterator ub = possibleConnections.upper_bound(usesPortIcon);
  ConnectionMap::iterator iter = lb;
  while (iter != ub) {
    Connection* c = iter->second;
    PortIcon* p = c->GetProvidesPortIcon();

#ifdef BUILD_BRIDGE
    // check if possible connection was a BridgeConnection
    BridgeConnection* bc = dynamic_cast<BridgeConnection*>(c);
    if (bc && bc->IsHighlighted()) {
      sci::cca::ConnectionID::pointer connID1, connID2;
      sci::cca::ComponentID::pointer bCID =
        builder->generateBridge(usesPortIcon->GetParent()->GetComponentInstance(),
                                usesPortIcon->GetPortName(),
                                p->GetParent()->GetComponentInstance(),
                                p->GetPortName(),
                                connID1,
                                connID2);
      // placement?
      ComponentIcon* bCI = AddIcon(bCID);
      // Bridge connections:
      // [user component] -con1- [bridge component] -con2- [provider component]
      BridgeConnection *con1 = new BridgeConnection(usesPortIcon, bCI->GetPortIcon(connID1->getProviderPortName()), connID1);
      connections.insert(std::make_pair(usesPortIcon, (Connection*) con1));

      PortIcon* pi2 = bCI->GetPortIcon(connID2->getUserPortName());
      BridgeConnection *con2 = new BridgeConnection(pi2, p, connID2);
      connections.insert(std::make_pair(pi2, (Connection*) con2));
      ClearPossibleConnections();
      return;
    }
#endif

    if (c->IsHighlighted()) {
      sci::cca::ConnectionID::pointer connID =
        builder->connect(usesPortIcon->GetParent()->GetComponentInstance(),
                         usesPortIcon->GetPortName(),
                         p->GetParent()->GetComponentInstance(),
                         p->GetPortName());
      if (connID.isNull()) {
        builderWindow->DisplayErrorMessage("Connection failed.");
      } else {
        Connection *con = new Connection(usesPortIcon, p, connID);
        connections.insert(std::make_pair(usesPortIcon, con));
      }
    }
    iter++;
  }
  ClearPossibleConnections();
}

bool NetworkCanvas::ShowPossibleConnections(PortIcon* port)
{
  ComponentIcon* pCI = port->GetParent();
  GUIBuilder::PortType type = port->GetPortType();
  for (ComponentMap::iterator iter = components.begin(); iter != components.end(); iter++) {
    ComponentIcon* ci_ = iter->second;

    SSIDL::array1<std::string> portArray;
    builder->getCompatiblePortList(pCI->GetComponentInstance(),
                                   port->GetPortName(),
                                   ci_->GetComponentInstance(),
                                   portArray);

    for (unsigned int j = 0; j < portArray.size(); j++) {
      Connection *con;
      if (type == GUIBuilder::Uses) {
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
        possibleConnections.insert(std::make_pair(pUses, con));
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
        possibleConnections.insert(std::make_pair(pUses, con));
      }
    }
  }

// TODO: is there a better way to figure out if bridging is available or not?
#ifdef BUILD_BRIDGE
  for (ComponentMap::iterator iter = components.begin(); iter != components.end(); iter++) {
    ComponentIcon* ci_ = iter->second;

    SSIDL::array1<std::string> portArray;
    builder->getBridgeablePortList(pCI->GetComponentInstance(),
                                   port->GetPortName(),
                                   ci_->GetComponentInstance(),
                                   portArray);

    for (unsigned int j = 0; j < portArray.size(); j++) {
      Connection *con;
      if (type == GUIBuilder::Uses) {
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
        con = new BridgeConnection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), true);
        possibleConnections.insert(std::make_pair(pUses, con));
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
        con = new BridgeConnection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), true);
        possibleConnections.insert(std::make_pair(pUses, con));
      }
    }
  }
#endif

  if (possibleConnections.empty()) {
    return false;
  }
  Refresh();
  return true;
}

void NetworkCanvas::HighlightConnection(const wxPoint& point)
{
  Connection *hc = 0;
  for (ConnectionMap::iterator iter = possibleConnections.begin(); iter != possibleConnections.end(); iter++) {
    if (iter->second->IsHighlighted()) {
      iter->second->Unhighlight();
    }
    if (iter->second->IsMouseOver(point) && hc == 0) {
      hc = iter->second;
    }
  }
  if (hc) {
    hc->Highlight();
  }
  Refresh();
}

void NetworkCanvas::Clear()
{
  SSIDL::array1<sci::cca::ComponentID::pointer> cids;

  //ClearPossibleConnections();
  ClearConnections();

  ComponentMap::iterator iter = components.begin();
  while (iter != components.end()) {
    ComponentIcon *ci = iter->second;
    components.erase(iter);
    iter = components.begin();

    cids.push_back(ci->GetComponentInstance());

    ci->Show(false);
    delete ci;
  }
  int destroyCount = cids.size();
  int ret = builder->destroyInstances(cids, 0);
  // show warning if not all component instances were destroyed
  if (ret != destroyCount) {
    // get error message?
    builderWindow->DisplayErrorMessage("Not all component instances were destroyed by the framework.");
  }
}

void NetworkCanvas::ClearPossibleConnections()
{
  ConnectionMap::iterator iter = possibleConnections.begin();
  while (iter != possibleConnections.end()) {
    Connection *c = iter->second;
    possibleConnections.erase(iter);
    iter = possibleConnections.begin();
    delete c;
  }
  possibleConnections.clear();
  Refresh();
}

void NetworkCanvas::ClearConnections()
{
  ConnectionMap::iterator iter = connections.begin();
  while (iter != connections.end()) {
    Disconnect(iter);
    iter = connections.begin();
  }
  connections.clear();
  Refresh();
}

ComponentIcon*
NetworkCanvas::AddIcon(sci::cca::ComponentID::pointer& compID)
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
    Refresh();
    return ci;
  }

  wxPoint origin(x, y);
  CalcUnscrolledPosition(origin.x, origin.y, &origin.x, &origin.y);

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

      // check with all the viewable modules - can new icon be placed?
      // searching for intersections with existing icons
      bool intersects = false;

      for (std::vector<wxRect>::iterator iter = rects.begin(); iter != rects.end(); iter++) {
        intersects |= candidateRect.Intersects(*iter);
      }

      if (! intersects) {
        int x_ = 0, y_ = 0;
        CalcScrolledPosition(candidateRect.GetPosition().x, candidateRect.GetPosition().y, &x_, &y_);

        ci->Move(x_, y_);
        ci->Show(true);

        wxRect windowRect = GetClientRect();
        if (! windowRect.Inside(x_, y_)) {
          int xu = 0, yu = 0;
          GetScrollPixelsPerUnit(&xu, &yu);
          Scroll(x_/xu, y_/yu);
        }
        Refresh();
        return ci;
      }
    }
  }

  ci->Move(max_row, max_col);
  ci->Show(true);
  Refresh();
  return ci;
}

ComponentIcon* NetworkCanvas::GetIcon(const std::string& instanceName)
{
  ComponentMap::iterator iter = components.find(instanceName);
  if (iter != components.end()) {
    return iter->second;
  }

  return 0;
}

void NetworkCanvas::DeleteIcon(const std::string& instanceName)
{
  ComponentMap::iterator iter = components.find(instanceName);
  if (iter != components.end()) {
    ComponentIcon *ci = iter->second;
    sci::cca::ComponentID::pointer cid = ci->GetComponentInstance();
    // disconnect
    PortList upl = ci->GetUsesPorts();
    for (PortList::const_iterator plIter = upl.begin(); plIter != upl.end(); plIter++) {
      ConnectionMap::iterator lb = connections.lower_bound(*plIter);
      ConnectionMap::iterator ub = connections.upper_bound(*plIter);

      ConnectionMap::iterator cIter = lb;
      while (cIter != ub) {
        Connection *c = cIter->second;
        builder->disconnect(c->GetConnectionID(), 0);
        delete c;
        ConnectionMap::iterator tmp = cIter;
        connections.erase(tmp);
        cIter++;
      }
    }

    PortList ppl = ci->GetProvidesPorts();
    for (PortList::const_iterator plIter = ppl.begin(); plIter != ppl.end(); plIter++) {
      ConnectionMap::iterator cIter = connections.begin();
      while (cIter != connections.end()) {
        Connection* c = cIter->second;
        if (c->GetProvidesPortIcon() == *plIter) {
          builder->disconnect(c->GetConnectionID(), 0);
          delete c;
          connections.erase(cIter);
          cIter = connections.begin();
        } else {
          cIter++;
        }
      }
    }
    ci->Show(false);
    delete ci;

   // destroy instance
    builder->destroyInstance(cid, 0);
    components.erase(iter);
  }
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

void NetworkCanvas::GetConnections(std::vector<Connection*>& cv)
{
  for (ConnectionMap::iterator iter = connections.begin(); iter != connections.end(); iter++) {
    cv.push_back(iter->second);
  }
}

///////////////////////////////////////////////////////////////////////////
// protected constructor and member functions

NetworkCanvas::NetworkCanvas()
{
  Init();
}

void NetworkCanvas::Init()
{
}

void NetworkCanvas::SetMenus()
{
  popupMenu = new wxMenu();
  componentMenu = new wxMenu(wxT(""), wxMENU_TEAROFF);
  clearMenuItem = new wxMenuItem(popupMenu, ID_MENU_CLEAR, wxT("&Clear"), wxT("Clear All"));

  popupMenu->Append(clearMenuItem);
  popupMenu->AppendSeparator();
  //popupMenu->Append(ID_MENU_COMPONENTS, wxT("Components submenu"), componentMenu);

  connectionPopupMenu = new wxMenu();
  deleteConnection = new wxMenuItem(connectionPopupMenu, ID_MENU_DISCONNECT, wxT("Disconnect"), wxT("Break this connection"));
  connectionPopupMenu->Append(deleteConnection);
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
  return windowRect;
}

void NetworkCanvas::Disconnect(const ConnectionMap::iterator& iter)
{
  Connection* c = iter->second;
  if (c) {
    builder->disconnect(c->GetConnectionID(), 0);
    delete c;
  }
  connections.erase(iter);

  builderWindow->RedrawMiniCanvas();
  Refresh();
}

void NetworkCanvas::DrawConnections(wxDC& dc)
{
  ConnectionMap::iterator iter;
  for (iter = possibleConnections.begin(); iter != possibleConnections.end(); iter++) {
    iter->second->OnDraw(dc);
  }

  for (iter = connections.begin(); iter != connections.end(); iter++) {
    iter->second->OnDraw(dc);
  }
}

///////////////////////////////////////////////////////////////////////////
// private member functions

}
