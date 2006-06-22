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
 *  ComponentIcon.cc:
 *
 *  Written by (Module.cc):
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 */


//#include <wx/region.h>
//#include <wx/dc.h>

#include <wx/dcbuffer.h>
#include <wx/gdicmn.h> // color database
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/gbsizer.h>
#include <wx/gauge.h>
#include <wx/string.h>

#include <CCA/Components/GUIBuilder/ComponentIcon.h>
#include <CCA/Components/GUIBuilder/PortIcon.h>
#include <CCA/Components/GUIBuilder/NetworkCanvas.h>

#include <string>

#ifndef DEBUG
#  define DEBUG 1
#endif

namespace GUIBuilder {

using namespace SCIRun;

BEGIN_EVENT_TABLE(ComponentIcon, wxPanel)
  EVT_LEFT_DOWN(ComponentIcon::OnLeftDown)
  EVT_LEFT_UP(ComponentIcon::OnLeftUp)
  EVT_RIGHT_UP(ComponentIcon::OnRightClick) // show popup menu
  EVT_MOTION(ComponentIcon::OnMouseMove)
  EVT_MENU(ID_MENU_GO, ComponentIcon::OnGo)
  EVT_MENU(ID_MENU_DELETE, ComponentIcon::OnDelete)
  EVT_BUTTON(ID_BUTTON_UI, ComponentIcon::OnUI)
  END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(ComponentIcon, wxPanel)

ComponentIcon::ComponentIcon(const sci::cca::GUIBuilder::pointer& bc,
                             wxWindowID winid,
                             NetworkCanvas* parent,
                             const sci::cca::ComponentID::pointer& compID,
                             int x, int y)
    : canvas(parent), hasUIPort(false), hasGoPort(false), isMoving(false), cid(compID), builder(bc)
{

  Init();
  Create(parent, winid, wxPoint(x, y));
}

ComponentIcon::~ComponentIcon()
{
  if (hasGoPort) {
    builder->disconnectGoPort(goPortName);
  }

  if (hasUIPort) {
    builder->disconnectUIPort(uiPortName);
  }

  PortList::iterator iter;
  for (iter = providesPorts.begin(); iter != providesPorts.end(); iter++) {
    (*iter)->Destroy();
  }

  for (iter = usesPorts.begin(); iter != usesPorts.end(); iter++) {
    (*iter)->Destroy();
  }
}

bool ComponentIcon::Create(wxWindow* parent, wxWindowID winid, const wxPoint& pos, const wxSize& size, long style)
{
  if (! wxPanel::Create(parent, winid, pos, size, style)) {
    return false;
  }

  SetFont(wxFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, "", wxFONTENCODING_SYSTEM));

  SetLayout();
  SetExtraStyle(wxWS_EX_BLOCK_EVENTS);

  popupMenu->Append(ID_MENU_DELETE, wxT("&Delete"), wxT("Delete this component icon."));
  //wxSize cs = GetClientSize();
  //wxSize s = GetSize();
  //borderSize = s - cs;
  return true;
}

///////////////////////////////////////////////////////////////////////////
// event handlers

void ComponentIcon::OnLeftDown(wxMouseEvent& event)
{
  canvas->GetUnscrolledMousePosition(movingStart);

#if DEBUG
  std::cerr << "ComponentIcon::OnLeftDown(..) pos=(" << movingStart.x << ", " << movingStart.y << ") "
            << std::endl;
#endif

  isMoving = true;
  canvas->SetMovingIcon(this);
}

void ComponentIcon::OnLeftUp(wxMouseEvent& event)
{
  isMoving = false;
  canvas->SetMovingIcon(0);
  //ReleaseMouse();
  event.StopPropagation();
}

void ComponentIcon::OnMouseMove(wxMouseEvent& event)
{
  if (event.LeftIsDown() && event.Dragging() && isMoving) {
    CaptureMouse();
    //Show(false);
    wxPoint p;
    canvas->GetUnscrolledPosition(event.GetPosition(), p);
    wxPoint mp;
    canvas->GetUnscrolledMousePosition(mp);

    wxPoint pp;
    GetCanvasPosition(pp);
    //canvas->FindIconAtPointer(pp);

    int dx = 0, dy = 0;
    //int newX = pp.x + p.x - movingStart.x;
    //int newY = pp.y + p.y - movingStart.y;
    int newX = mp.x + pp.x + p.x - movingStart.x;
    int newY = mp.y + pp.y + p.y - movingStart.y;
    wxPoint topLeft;
    canvas->GetUnscrolledPosition(wxPoint(newX, newY), topLeft);

#if DEBUG
    std::cerr << "ComponentIcon::OnMouseMove(..) "
      //<< "event pos=(" << p.x << ", " << p.y << ")"
              << std::endl
              << "\tmouse canvas pos=(" << mp.x << ", " << mp.y << ")"
              << std::endl
              << "\ticon pos=(" << pp.x << ", " << pp.y << ")"
              << std::endl
              << "\ttop left pos=(" << topLeft.x << ", " << topLeft.y << ")"
              << std::endl;
#endif

    //     // test
    //     wxPoint p;
    //     p = mp;

    // adjust for canvas boundaries
    if (topLeft.x < 0) {
      newX -= topLeft.x;
      if (p.x < 0) {
        mp.x -= p.x;
        p.x = 0;
        WarpPointer(mp.x, mp.y); // move mouse pointer
      }
      dx -= 1;
    }

    if (topLeft.y < 0) {
      newY -= topLeft.y;
      if (p.y < 0) {
        mp.y -= p.y;
        p.y = 0;
        WarpPointer(mp.x, mp.y); // move mouse pointer
      }
      dy -= 1;
    }

    int cw = canvas->GetVirtualSize().GetWidth();
    int mw = GetSize().GetWidth();

    if (topLeft.x > cw - mw) {
      newX -= topLeft.x - (cw - mw);
      if (p.x > cw) {
        mp.x -= (p.x - cw);
        p.x = cw - mw;
        WarpPointer(mp.x, mp.y);
      }
      dx = 1;
    }

    int ch = canvas->GetVirtualSize().GetHeight();
    int mh = GetSize().GetHeight();

    if (topLeft.y > ch - mh) {
      newY -= topLeft.y - (ch - mh);
      if (p.y > ch) {
        mp.y -= (p.y - ch);
        p.y = ch;
        WarpPointer(mp.x, mp.y);
      }
      dy = 1;
    }

    movingStart = p;
    wxPoint np;
    canvas->GetScrolledPosition(wxPoint(newX, newY), np);

    //     std::cerr << "\tmove to scrolled (" << np.x << ", " << np.y << ") or unscrolled (" << newX << ", " << newY << ")" << std::endl;
    //Move(np.x, np.y);
    Move(newX, newY);
    //Show(true);
    ReleaseMouse();
    canvas->Refresh();


    //     CalcScrolledPosition(newX, newY, &newX, &newY);
    //     movingIcon->Move(newX, newY);
    //     movingIcon->Show(true);
    //     // reset moving icon connections
    //     Refresh();
    //     builderWindow->RedrawMiniCanvas();

    //     wxRect windowRect = GetClientRect();
    //     if (! windowRect.Inside(newX + mw, newY + mh)) {
    //       int xu = 0, yu = 0;
    //       GetScrollPixelsPerUnit(&xu, &yu);
    //       Scroll(newX/xu, newY/yu);
    //     }
  }

  event.StopPropagation();
}

void ComponentIcon::OnRightClick(wxMouseEvent& event)
{
  PopupMenu(popupMenu, event.GetPosition());
}

void ComponentIcon::OnGo(wxCommandEvent& event)
{
  int status = builder->go(goPortName);
}

void ComponentIcon::OnDelete(wxCommandEvent& event)
{
  canvas->DeleteIcon(cid->getInstanceName());
}

void ComponentIcon::OnUI(wxCommandEvent& event)
{
  int status = builder->ui(uiPortName);
}

void ComponentIcon::GetCanvasPosition(wxPoint& p)
{
  canvas->GetUnscrolledPosition(this->GetPosition(), p);
}

PortIcon* ComponentIcon::GetPortIcon(const std::string& portName)
{
  PortList::iterator iter;
  for (iter = providesPorts.begin(); iter != providesPorts.end(); iter++) {
    if ((*iter)->GetPortName() == portName) {
      return *iter;
    }
  }

  for (iter = usesPorts.begin(); iter != usesPorts.end(); iter++) {
    if ((*iter)->GetPortName() == portName) {
      return *iter;
    }
  }
  return 0;
}


///////////////////////////////////////////////////////////////////////////
// protected constructor and member functions

ComponentIcon::ComponentIcon()
{
  Init();
}

void ComponentIcon::Init()
{
}

void ComponentIcon::SetLayout()
{
  popupMenu = new wxMenu();
  SetPortIcons();

  const wxSize UI_SIZE(30, 30);
  uiButton = new wxButton(this, ID_BUTTON_UI, wxT("UI"), wxDefaultPosition, UI_SIZE, wxDOUBLE_BORDER|wxRAISED_BORDER);
  gridBagSizer->Add(uiButton, wxGBPosition(0, 1), /* wxDefaultSpan */ wxGBSpan(2, 1),
                    wxFIXED_MINSIZE|wxALIGN_CENTER, BORDER_SIZE);
  if (! hasUIPort) {
    uiButton->Enable(false);
  }

  label = new wxStaticText(this, wxID_ANY, cid->getInstanceName(), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT);
  gridBagSizer->Add(label, wxGBPosition(0, 2), wxGBSpan(1, 2), wxALL|wxALIGN_CENTER, BORDER_SIZE);

  timeLabel = new wxStaticText(this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER);
  gridBagSizer->Add(timeLabel, wxGBPosition(1, 2), wxDefaultSpan, wxALL|wxALIGN_CENTER_HORIZONTAL, BORDER_SIZE);

  const int PROG_LEN = 50;
  const wxSize PROG_SIZE(PROG_LEN, 15);
  progressGauge = new wxGauge(this, ID_PROGRESS, PROG_LEN, wxDefaultPosition, PROG_SIZE,
                              wxDOUBLE_BORDER|wxSUNKEN_BORDER|wxGA_HORIZONTAL|wxGA_SMOOTH);
  gridBagSizer->Add(progressGauge, wxGBPosition(1, 3), wxDefaultSpan, wxALL|wxALIGN_LEFT, BORDER_SIZE);

  const wxSize STATUS_SIZE(15, 15);
  msgButton = new wxButton(this, ID_BUTTON_STATUS, wxT(""), wxDefaultPosition,
                           STATUS_SIZE, wxDOUBLE_BORDER|wxRAISED_BORDER);
  gridBagSizer->Add(msgButton, wxGBPosition(1, 4), wxDefaultSpan, wxFIXED_MINSIZE|wxALIGN_LEFT, BORDER_SIZE);

  // hide until implemented
  msgButton->Enable(false);
  msgButton->Show(false);

  SetSizerAndFit(gridBagSizer);
}

void ComponentIcon::SetPortIcons()
{
  gridBagSizer = new wxGridBagSizer(GAP_SIZE, GAP_SIZE);

  gridBagSizer->AddGrowableCol(PROV_PORT_COL);
  gridBagSizer->AddGrowableCol(USES_PORT_COL);

  SSIDL::array1<std::string> providedPorts;
  builder->getProvidedPortNames(cid, providedPorts);

  if (providedPorts.size() == 0) {
    // make room at edges
    gridBagSizer->Add(PortIcon::PORT_WIDTH, PortIcon::PORT_HEIGHT, wxGBPosition(0, PROV_PORT_COL),
                      wxDefaultSpan, wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
  } else {
    for (unsigned int i = 0, j = 0; i < providedPorts.size(); i++) {
 std::cerr << "ComponentIcon::SetPortIcons(): provided port=" << providedPorts[i] << std::endl;
      if (providedPorts[i].rfind("ui") != std::string::npos) {
        if (builder->connectUIPort(cid->getInstanceName(), providedPorts[i], cid, uiPortName)) {
          hasUIPort = true;
        }
      } else if (providedPorts[i].rfind("go") != std::string::npos) {
        if (builder->connectGoPort(cid->getInstanceName(), providedPorts[i], cid, goPortName)) {
          hasGoPort = true;
          popupMenu->Append(ID_MENU_GO, wxT("&Go"), wxT("CCA go port"));
        }
      } else {
        PortIcon *pi = new PortIcon(builder, this, wxID_ANY, GUIBuilder::Provides, providedPorts[i]);
        //ports[providedPorts[i]] = pi;
        providesPorts.push_back(pi);
        gridBagSizer->Add(pi, wxGBPosition(j++, 0), wxDefaultSpan,
                          wxFIXED_MINSIZE|wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
      }
    }
  }

  SSIDL::array1<std::string> usedPorts;
  builder->getUsedPortNames(cid, usedPorts);

  if (usedPorts.size() == 0) {
    // make room at edges
    gridBagSizer->Add(PortIcon::PORT_WIDTH, PortIcon::PORT_HEIGHT, wxGBPosition(0, USES_PORT_COL),
                      wxDefaultSpan, wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
  } else {
    for (unsigned int i = 0; i < usedPorts.size(); i++) {
      if (usedPorts[i].rfind("ui") != std::string::npos) {
        // show warning
        continue;
      } else if (usedPorts[i].rfind("go") != std::string::npos) {
        // show warning
        continue;
      } else {
        PortIcon *pi = new PortIcon(builder, this, wxID_ANY, GUIBuilder::Uses, usedPorts[i]);
        //ports[usedPorts[i]] = pi;
        usesPorts.push_back(pi);
        gridBagSizer->Add(pi, wxGBPosition(i, USES_PORT_COL),
                          wxDefaultSpan, wxFIXED_MINSIZE|wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
      }
    }
  }
}

}

