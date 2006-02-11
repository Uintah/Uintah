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

#include <CCA/Components/Builder/Connection.h>
#include <CCA/Components/Builder/ComponentIcon.h>
#include <CCA/Components/Builder/NetworkCanvas.h>
//#include <iostream>

#include <wx/wx.h>
#include <wx/window.h>
#include <wx/gdicmn.h>
#include <wx/dc.h>

namespace GUIBuilder {

BEGIN_EVENT_TABLE(Connection, wxEvtHandler)
  EVT_LEFT_DOWN(Connection::OnLeftDown)
  EVT_LEFT_UP(Connection::OnLeftUp)
  EVT_RIGHT_UP(Connection::OnRightClick)
END_EVENT_TABLE()

Connection::Connection(PortIcon* pU, PortIcon* pP, const sci::cca::ConnectionID::pointer& connID, bool possibleConnection) : wxEvtHandler(), NUM_POINTS(12), NUM_DRAW_POINTS(6), pUses(pU), pProvides(pP), possibleConnection(possibleConnection), connectionID(connID)
{
  points = new wxPoint[NUM_POINTS];
  ResetPoints();
  // set colour
  if (possibleConnection) {
    colour = wxColour(wxTheColourDatabase->Find("BLACK"));
  }
  hColour = wxColour(wxTheColourDatabase->Find("WHITE"));
}

Connection::~Connection()
{
std::cerr << "Connection::~Connection()" << std::endl;
  delete [] points;
}

void Connection::ResetPoints()
{
  wxPoint u = pUses->GetPosition();
  u.y +=  u.y/2; // connect at vertical halfway point on port
  wxPoint up = pUses->GetParent()->GetCanvasPosition() + u; // position rel. to component icon parent

  wxPoint p = pProvides->GetPosition();
  p.y += p.y/2;
  wxPoint pp = pProvides->GetParent()->GetCanvasPosition() + p; // position rel. to component icon parent

  wxRect ru(pUses->GetParent()->GetCanvasPosition(), pUses->GetParent()->GetSize());
  wxRect rp(pProvides->GetParent()->GetCanvasPosition(), pProvides->GetParent()->GetSize());

  int t = PortIcon::PORT_WIDTH;
  int h = PortIcon::PORT_HEIGHT;
  int mid;

  if ( (up.x + h) < (pp.x - h) ) {
    mid = (up.y + pp.y) / 2;
    int x_mid = (up.x + pp.x) / 2;

    points[0] = wxPoint(up.x,     up.y - t);
    points[1] = wxPoint(up.x + 1, up.y - t);
    points[2] = wxPoint(up.x + 2, up.y - t);

    if (up.y <= mid) {
      points[3] = wxPoint(x_mid + t, points[2].y);
    } else {
      points[3] = wxPoint(x_mid - t, points[2].y);
    }
    points[4] = wxPoint(points[3].x, pp.y - t);
    points[5] = wxPoint(pp.x,    points[4].y);
    points[6] = wxPoint(pp.x,    pp.y + t);

    if (up.y <= mid) {
      points[7] = wxPoint(x_mid - t, points[6].y);
    } else {
      points[7] = wxPoint(x_mid + t, points[6].y);
    }
    points[8] =  wxPoint(points[7].x,  up.y + t);
    points[9] =  wxPoint(up.x + 2, points[8].y);
    points[10] = wxPoint(up.x + 1, points[8].y);
    points[11] = wxPoint(up.x,     points[8].y);
  } else {
    int adj = 2 * t;

    if (ru.GetTop() > rp.GetBottom() + adj) {
      mid = (ru.GetTop() + rp.GetBottom()) / 2;
    } else if (rp.GetTop() > ru.GetBottom() + adj) {
      mid = (ru.GetBottom() + rp.GetTop()) / 2;
    } else if (ru.GetTop() < rp.GetTop()) {
      mid = ru.GetTop() - adj;
    } else {
      mid = rp.GetTop() - adj;
    }

    points[0] = wxPoint(up.x, up.y  - t);
    if (up.y < mid) {
      points[1] = wxPoint(up.x + h + t, points[0].y);
    } else {
      points[1] = wxPoint(up.x + h - t, points[0].y);
    }

    if (up.x + h < pp.x - h) {
      points[2] = wxPoint(points[1].x, mid - t);
    } else {
      points[2] = wxPoint(points[1].x, mid + t);
    }

    if (pp.y > mid) {
      points[3] = wxPoint(pp.x - h + t, points[2].y);
    } else {
      points[3] = wxPoint(pp.x - h - t, points[2].y);
    }
    points[4] = wxPoint(points[3].x, pp.y - t);
    points[5] = wxPoint(pp.x,    points[4].y);
    points[6] = wxPoint(points[5].x, pp.y + t);

    if (pp.y > mid) {
      points[7] = wxPoint(pp.x - h - t, points[6].y);
    } else {
      points[7] = wxPoint(pp.x - h + t, points[6].y);
    }

    if (up.x + h < pp.x - h) {
      points[8] = wxPoint(points[7].x, mid + t);
    } else {
      points[8] = wxPoint(points[7].x, mid - t);
    }

    if (up.y < mid) {
      points[9] = wxPoint(up.x + h - t, points[8].y);
    } else {
      points[9] = wxPoint(up.x + h + t, points[8].y);
    }
    points[10] = wxPoint(points[9].x, up.y + t);
    points[11] = wxPoint(up.x,    points[10].y);
  }
}

void Connection::OnDraw(wxDC& dc)
{
  ResetPoints();
  wxPoint drawPoints[NUM_DRAW_POINTS];
  if (setConnection(drawPoints, NUM_DRAW_POINTS)) {
    if (possibleConnection) {
      dc.SetPen(wxPen(colour, 2, wxSOLID));
    } else {
      dc.SetPen(wxPen(colour, 4, wxSOLID));
    }
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    dc.DrawLines(NUM_DRAW_POINTS, drawPoints, 0, 0);
  }
}

void Connection::OnLeftDown(wxMouseEvent& event)
{
  // connecting
  std::cerr << "Connection::OnLeftDown(..)" << std::endl;
}

void Connection::OnLeftUp(wxMouseEvent& event)
{
  std::cerr << "Connection::OnLeftUp(..)" << std::endl;
}

void Connection::OnRightClick(wxMouseEvent& event)
{
  wxMenu *m = new wxMenu();
  m->Append(wxID_ANY, wxT("Connection Menu Item"));

  NetworkCanvas* canvas = pUses->GetParent()->GetCanvas();
  canvas->PopupMenu(m, event.GetPosition());
}


///////////////////////////////////////////////////////////////////////////
// protected member functions

// void Connection::drawConnection(const wxPoint[] points, wxPoint[] drawPoints)
// {
//   for (unsigned int i = 0; i < NUM_POINTS; i++) {
//     drawPoints[i] = (points[i] + points[NUM_POINTS - 1 - i]) / 2;
//   }
// }

bool Connection::setConnection(wxPoint drawPoints[], int arrayLen)
{
  if (arrayLen != NUM_DRAW_POINTS) {
    return false;
  }
  for (int i = 0; i < NUM_DRAW_POINTS; i++) {
    drawPoints[i] = (points[i] + points[NUM_POINTS - 1 - i]);
    drawPoints[i].x /= 2;
    drawPoints[i].y /= 2;
  }
  return true;
}

// void Connection::drawPoints(const wxPoint[] points)
// {
// }


}
