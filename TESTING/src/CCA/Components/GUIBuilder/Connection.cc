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
 *  NetworkCanvas.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Scientific Computing and Imaging
 *   April 2002
 *  Ported to wxWidgets:
 *   Ayla Khan
 *   January 2006
 *
 */

#include <CCA/Components/GUIBuilder/Connection.h>
#include <CCA/Components/GUIBuilder/ComponentIcon.h>
#include <CCA/Components/GUIBuilder/PortIcon.h>
#include <CCA/Components/GUIBuilder/NetworkCanvas.h>

#include <wx/wx.h>
#include <wx/window.h>
#include <wx/gdicmn.h>
#include <wx/dc.h>

#ifndef DEBUG
#  define DEBUG 0
#endif


namespace GUIBuilder {

const int Connection::NUM_POINTS(12);
const int Connection::NUM_DRAW_POINTS(6);

Connection::Connection(PortIcon* pU, PortIcon* pP, const sci::cca::ConnectionID::pointer& connID, bool possibleConnection) : pUses(pU), pProvides(pP), possibleConnection(possibleConnection), highlight(false), connectionID(connID)
{
  points = new wxPoint[NUM_POINTS];
  drawPoints = new wxPoint[NUM_DRAW_POINTS];

  ResetPoints();
  setConnection();
  // set color
  if (possibleConnection) {
    color = wxTheColourDatabase->Find("BLACK");
  } else {
    color = pUses->GetPortColor();
  }
  hColor = wxTheColourDatabase->Find("WHITE");
}

Connection::~Connection()
{
  delete [] points;
  delete [] drawPoints;
}

void Connection::ResetPoints()
{
  wxPoint u = pUses->GetPosition();
  u.y +=  u.y/2; // connect at vertical halfway point on port

  wxPoint usesIconPos;
  pUses->GetParent()->GetCanvasPosition(usesIconPos);
  wxPoint up = usesIconPos + u; // position rel. to component icon parent

  wxPoint p = pProvides->GetPosition();
  p.y += p.y/2;

  wxPoint providesIconPos;
  pProvides->GetParent()->GetCanvasPosition(providesIconPos);
  wxPoint pp = providesIconPos + p; // position rel. to component icon parent

  wxRect ru(usesIconPos, pUses->GetParent()->GetSize());
  wxRect rp(providesIconPos, pProvides->GetParent()->GetSize());

  int t = PortIcon::PORT_WIDTH;
  int h = PortIcon::PORT_HEIGHT;
  int mid;

#if DEBUG
std::cerr << "Uses icon pos=("
	  << usesIconPos.x
	  << ", "
	  << usesIconPos.y
          << ") Uses port pos=("
	  << up.x
	  << ", "
	  << up.y
	  << ") Provides icon pos=("
	  << providesIconPos.x
	  << ", "
	  << providesIconPos.y
          << ") Provides port pos=("
	  << pp.x
	  << ", "
	  << pp.y
	  << ")"
          << std::endl;
#endif

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
  setConnection();
  if (highlight) {
    wxPen* pen = wxThePenList->FindOrCreatePen(hColor, 2, wxSOLID);
    dc.SetPen(*pen);
  } else {
    wxPen* pen = wxThePenList->FindOrCreatePen(color, 2, wxSOLID);
    dc.SetPen(*pen);
  }
  dc.SetBrush(*wxTRANSPARENT_BRUSH);
  dc.DrawLines(NUM_DRAW_POINTS, drawPoints, 0, 0);
}

bool Connection::IsMouseOver(const wxPoint& position)
{
  const int tolerance = 2;

  for (int i = 0; i < NUM_DRAW_POINTS; i += 2) {
    wxRect r;
    wxRect pr;

    wxPoint pTopLeft, pBottomRight;
//     if (position.x - tolerance <= 0) {
//       pTopLeft.x = 0;
//     } else {
//       pTopLeft.x = position.x - tolerance;
//     }
//     if (position.y - tolerance <= 0) {
//       pTopLeft.y = 0;
//     } else {
//       pTopLeft.y = position.y - tolerance;
//     }
    pTopLeft.x = position.x - tolerance;
    pTopLeft.y = position.y - tolerance;

    pBottomRight.x = position.x + tolerance;
    pBottomRight.y = position.y + tolerance;

    pr = wxRect(pTopLeft, pBottomRight);

#if DEBUG
std::cerr << "Connection::IsMouseOver point top left ("
	  << pTopLeft.x << ", " << pTopLeft.y << ") point bottom right ("
	  << pBottomRight.x << ", " << pBottomRight.y << ")"
          << std::endl;
#endif

    if (drawPoints[i].x > drawPoints[i+1].x) {
      wxPoint topLeft(drawPoints[i+1].x - tolerance, drawPoints[i+1].y - tolerance);
      wxPoint bottomRight(drawPoints[i].x + tolerance, drawPoints[i].y + tolerance);
#if DEBUG
      std::cerr << "Connection::IsMouseOver (if) y_i = y_i+1, x_i > x_i+1" << std::endl
		<< "\tposition=(" << position.x << ", " << position.y << ") " << std::endl
		<< "\tpoint " << i << "=(" << drawPoints[i].x << ", " << drawPoints[i].y <<  ") " << std::endl
		<< "\tpoint " << i+1 << "=(" << drawPoints[i+1].x << ", " << drawPoints[i+1].y << ")" << std::endl
		<< "\ttopLeft=(" << topLeft.x << ", " << topLeft.y <<  ") " << std::endl
		<< "\tbottomRight=(" << bottomRight.x << ", " << bottomRight.y << ")"
		<< std::endl;
#endif
      r = wxRect(topLeft, bottomRight);
    } else {
      wxPoint topLeft(drawPoints[i].x - tolerance, drawPoints[i].y - tolerance);
      wxPoint bottomRight(drawPoints[i+1].x + tolerance, drawPoints[i+1].y + tolerance);
#if DEBUG
      std::cerr << "Connection::IsMouseOver (else) y_i = y_i+1, x_i <= x_i+1" << std::endl
		<< "\tposition=(" << position.x << ", " << position.y << ") " << std::endl
		<< "\tpoint " << i << "=(" << drawPoints[i].x << ", " << drawPoints[i].y <<  ") " << std::endl
		<< "\tpoint " << i+1 << "=(" << drawPoints[i+1].x << ", " << drawPoints[i+1].y << ")" << std::endl
		<< "\ttopLeft=(" << topLeft.x << ", " << topLeft.y <<  ") " << std::endl
		<< "\tbottomRight=(" << bottomRight.x << ", " << bottomRight.y << ")"
		<< std::endl;
#endif
      r = wxRect(topLeft, bottomRight);
    }
    if (r.Intersects(pr)) {
#if DEBUG
      std::cerr << "Connection::IsMouseOver(..): mouse over!" << std::endl;
#endif
      return true;
    }
  }
  return false;
}

void Connection::GetDrawingPoints(wxPoint **pa, const int size)
{
  for (int i = 0; i < size && i < NUM_DRAW_POINTS; i++) {
    (*pa)[i] = drawPoints[i];
  }
}

///////////////////////////////////////////////////////////////////////////
// protected member functions

void Connection::setConnection()
{
  for (int i = 0; i < NUM_DRAW_POINTS; i++) {
    drawPoints[i] = (points[i] + points[NUM_POINTS - 1 - i]);
    drawPoints[i].x /= 2;
    drawPoints[i].y /= 2;
#if 0
#if DEBUG
std::cerr << "Connection::setConnection y_i = y_i+1, x_i <= x_i+1" << std::endl
          << "\tpoint " << i << "=(" << points[i].x << ", " << points[i].y <<  ") " << std::endl
          << "\tpoint " << NUM_POINTS - 1 - i << "=(" << points[NUM_POINTS - 1 - i].x << ", " << points[NUM_POINTS - 1 - i].y <<  ") " << std::endl
          << "\tdraw point " << i << "=(" << drawPoints[i].x << ", " << drawPoints[i].y <<  ") "
          << std::endl;
#endif
#endif
  }
}

}
