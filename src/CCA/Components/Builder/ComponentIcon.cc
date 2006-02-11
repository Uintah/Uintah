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


//#include <wx/region.h>
//#include <wx/dc.h>

#include <wx/dcbuffer.h>
#include <wx/gdicmn.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/gbsizer.h>
#include <wx/gauge.h>
#include <wx/gdicmn.h> // color database

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
    parent->GetCanvas()->ShowPossibleConnections(this);
  }
}

void PortIcon::OnLeftUp(wxMouseEvent& event)
{
  parent->GetCanvas()->ClearPossibleConnections();
}

void PortIcon::OnMouseMove(wxMouseEvent& event)
{
  std::cerr << "PortIcon::OnMouseMove(..)" << std::endl;
  // connect
  connecting = true;
  // figure out which connection we're over and change to highlight colour

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


BEGIN_EVENT_TABLE(ComponentIcon, wxPanel)
  //EVT_PAINT(ComponentIcon::OnPaint)
  //EVT_ERASE_BACKGROUND(ComponentIcon::OnEraseBackground)
  EVT_LEFT_DOWN(ComponentIcon::OnLeftDown)
  EVT_LEFT_UP(ComponentIcon::OnLeftUp)
  //EVT_RIGHT_DOWN(ComponentIcon::OnMouseDown)
  //EVT_MIDDLE_DOWN(ComponentIcon::OnMouseDown)
  EVT_MOTION(ComponentIcon::OnMouseMove)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(ComponentIcon, wxPanel)

ComponentIcon::ComponentIcon(const sci::cca::BuilderComponent::pointer& bc, wxWindowID winid, NetworkCanvas* parent, const sci::cca::ComponentID::pointer& compID, int x, int y)
  : /* dragMode(TEST_DRAG_NONE), */ canvas(parent), cid(compID), builder(bc), hasUIPort(false), hasGoPort(false), isSciPort(false), ID_MENU_POPUP(BuilderWindow::GetNextID()), ID_BUTTON_UI(BuilderWindow::GetNextID()), ID_BUTTON_STATUS(BuilderWindow::GetNextID()), ID_PROGRESS(BuilderWindow::GetNextID())
{

  Init();
  Create(parent, winid, wxPoint(x, y));
  //Show(true);
}

ComponentIcon::~ComponentIcon()
{
}

bool ComponentIcon::Create(wxWindow* parent, wxWindowID winid, const wxPoint& pos, const wxSize& size, long style)
{
  if (! wxPanel::Create(parent, winid, pos, size, style)) {
    return false;
  }

  SetFont(wxFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, "", wxFONTENCODING_SYSTEM));

// test
//SetOwnBackgroundColour(wxTheColourDatabase->Find("GOLDENROD"));

  const int GAP_SIZE = 1;
  const int BORDER_SIZE = 4;
  const int PORT_BORDER_SIZE = 10;
  gridBagSizer = new wxGridBagSizer(GAP_SIZE, GAP_SIZE);

  gridBagSizer->AddGrowableCol(0);
  gridBagSizer->AddGrowableCol(5);

  SSIDL::array1<std::string> providedPorts;
  builder->getProvidedPortNames(cid, providedPorts);

  if (providedPorts.size() == 0) {
    gridBagSizer->Add(PortIcon::PORT_WIDTH, PortIcon::PORT_HEIGHT, wxGBPosition(0, 0), wxDefaultSpan, wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
  } else {
    for (unsigned int i = 0; i < providedPorts.size(); i++) {
      if (providedPorts[i] == "ui") {
	hasUIPort = true;
      } else if (providedPorts[i] == "sci.ui") {
	hasUIPort = true;
	isSciPort = true;
      } else if (providedPorts[i] == "go") {
	hasGoPort = true;
      } else if (providedPorts[i] == "sci.go") {
	hasGoPort = true;
	isSciPort = true;
      } else {
	PortIcon *pi = new PortIcon(this, wxID_ANY, Builder::Provides, providedPorts[i]);
	ports[providedPorts[i]] = pi;
	gridBagSizer->Add(pi, wxGBPosition(i, 0), wxDefaultSpan,
			  wxFIXED_MINSIZE|wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
      }
    }
  }

  SSIDL::array1<std::string> usedPorts;
  builder->getUsedPortNames(cid, usedPorts);

  if (usedPorts.size() == 0) {
    gridBagSizer->Add(PortIcon::PORT_WIDTH, PortIcon::PORT_HEIGHT, wxGBPosition(0, 5), wxDefaultSpan, wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
  } else {
    for (unsigned int i = 0; i < usedPorts.size(); i++) {
      //int id = BuilderWindow::GetNextID();
      PortIcon *pi = new PortIcon(this, wxID_ANY, Builder::Uses, usedPorts[i]);
      ports[usedPorts[i]] = pi;
      gridBagSizer->Add(pi, wxGBPosition(i, 5), wxDefaultSpan, wxFIXED_MINSIZE|wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL, PORT_BORDER_SIZE);
    }
  }

  const wxSize UI_SIZE(30, 30);
  uiButton = new wxButton(this, ID_BUTTON_UI, wxT("UI"), wxDefaultPosition, UI_SIZE, wxDOUBLE_BORDER|wxRAISED_BORDER);
  gridBagSizer->Add(uiButton, wxGBPosition(0, 1), /* wxDefaultSpan */ wxGBSpan(2, 1), wxFIXED_MINSIZE|wxALIGN_CENTER, BORDER_SIZE);
  if (! hasUIPort) {
    uiButton->Enable(false);
  }

  label = new wxStaticText(this, wxID_ANY, cid->getInstanceName(), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT);
  gridBagSizer->Add(label, wxGBPosition(0, 2), wxGBSpan(1, 2), wxALL|wxALIGN_CENTER, BORDER_SIZE);

  timeLabel = new wxStaticText(this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER);
  gridBagSizer->Add(timeLabel, wxGBPosition(1, 2), wxDefaultSpan, wxALL|wxALIGN_CENTER_HORIZONTAL, BORDER_SIZE);

  const int PROG_LEN = 50;
  const wxSize PROG_SIZE(PROG_LEN, 15);
  progressGauge = new wxGauge(this, ID_PROGRESS, PROG_LEN, wxDefaultPosition, PROG_SIZE,  wxDOUBLE_BORDER|wxSUNKEN_BORDER|wxGA_HORIZONTAL|wxGA_SMOOTH);
  gridBagSizer->Add(progressGauge, wxGBPosition(1, 3), wxDefaultSpan, wxALL|wxALIGN_LEFT, BORDER_SIZE);

  const wxSize STATUS_SIZE(15, 15);
  msgButton = new wxButton(this, ID_BUTTON_STATUS, wxT(""), wxDefaultPosition, STATUS_SIZE, wxDOUBLE_BORDER|wxRAISED_BORDER);
  gridBagSizer->Add(msgButton, wxGBPosition(1, 4), wxDefaultSpan, wxFIXED_MINSIZE|wxALIGN_LEFT, BORDER_SIZE);

  SetSizerAndFit(gridBagSizer);

  SetExtraStyle(wxWS_EX_BLOCK_EVENTS);

  wxSize cs = GetClientSize();
  wxSize s = GetSize();
  borderSize = s - cs;

  return true;
}

void ComponentIcon::OnPaint(wxPaintEvent& event)
{
std::cerr << "ComponentIcon::OnPaint()" << std::endl;

// paint border etc. here


//   wxBufferedPaintDC dc(this);
//   wxColour backgroundColour = GetBackgroundColour();
//   if (!backgroundColour.Ok())
//     backgroundColour =
//       wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE);

//   dc.SetBrush(wxBrush(backgroundColour));
//   dc.SetPen(wxPen(backgroundColour, 1));

//   wxRect windowRect(wxPoint(0, 0), GetClientSize());
//   dc.DrawRectangle(windowRect);

//   wxRect test(GetPosition(), borderSize);
//   dc.DrawRectangle(test);

//   wxBrush* b = wxTheBrushList->FindOrCreateBrush(wxSYS_COLOUR_3DHILIGHT, wxSOLID);
//   wxPen* p = wxThePenList->FindOrCreatePen(wxSYS_COLOUR_3DHILIGHT, 1, wxSOLID);
//   wxBrush* b = wxTheBrushList->FindOrCreateBrush("MAGENTA", wxSOLID);
//   wxPen* p = wxThePenList->FindOrCreatePen("MAGENTA", 1, wxSOLID);
//   dc.SetBrush(*b);
//   dc.SetPen(*p);
//   dc.DrawRectangle(0, 0, 4, GetSize().GetHeight());

//   b = wxTheBrushList->FindOrCreateBrush("BLUE", wxSOLID);
//   p = wxThePenList->FindOrCreatePen("BLUE", 1, wxSOLID);
//   dc.SetBrush(*b);
//   dc.SetPen(*p);
//   dc.DrawRectangle(0, 0, GetSize().GetWidth(), 4);

//   b = wxTheBrushList->FindOrCreateBrush("GREEN", wxSOLID);
//   p = wxThePenList->FindOrCreatePen("GREEN", 1, wxSOLID);
//   dc.SetBrush(*b);
//   dc.SetPen(*p);
//   dc.DrawRectangle(GetSize().GetWidth()-4, 0, 4, GetSize().GetHeight());

//   b = wxTheBrushList->FindOrCreateBrush("YELLOW", wxSOLID);
//   p = wxThePenList->FindOrCreatePen("YELLOW", 1, wxSOLID);
//   dc.SetBrush(*b);
//   dc.SetPen(*p);
//   dc.DrawRectangle(0, GetSize().GetHeight()-4, GetSize().GetWidth(), 4);
}

void ComponentIcon::OnLeftDown(wxMouseEvent& event)
{
  wxPoint p = event.GetPosition();
  wxClientDC dc;
  wxPoint lp = event.GetLogicalPosition(dc);
  std::cerr << "ComponentIcon::OnLeftDown(..) event pos=(" << p.x << ", " << p.y << ")"
            << std::endl
	    << "logical event pos=(" << lp.x << ", " << lp.y << ")" << std::endl;

  // normally, don't propagate mouse move events; in this case, we want the network canvas (parent) to handle this
  canvas->SetMovingIcon(this);
  CaptureMouse();
  canvas->GetEventHandler()->ProcessEvent(event);
}

void ComponentIcon::OnLeftUp(wxMouseEvent& event)
{
  canvas->GetEventHandler()->ProcessEvent(event);
  ReleaseMouse();
}

void ComponentIcon::OnMouseMove(wxMouseEvent& event)
{
  // normally, don't propagate mouse move events; in this case, we want the network canvas (parent) to handle this
  //canvas->SetMovingIcon(this);

  wxPoint p = event.GetPosition();
  wxClientDC dc;
  wxPoint lp = event.GetLogicalPosition(dc);
  std::cerr << "ComponentIcon::OnMouseMove(..) event pos=(" << p.x << ", " << p.y << ")"
            << std::endl
	    << "logical event pos=(" << lp.x << ", " << lp.y << ")" << std::endl;

  canvas->GetEventHandler()->ProcessEvent(event);
}

wxPoint ComponentIcon::GetCanvasPosition()
{
  return canvas->GetIconPosition(this);
}


///////////////////////////////////////////////////////////////////////////
// protected constructor and member functions

ComponentIcon::ComponentIcon() : ID_MENU_POPUP(BuilderWindow::GetNextID()), ID_BUTTON_UI(BuilderWindow::GetNextID()), ID_BUTTON_STATUS(BuilderWindow::GetNextID()), ID_PROGRESS(BuilderWindow::GetNextID())
{
  Init();
}

void ComponentIcon::Init()
{
}



}
