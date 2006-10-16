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

#include <wx/dcbuffer.h>
#include <wx/scrolwin.h>
#include <wx/string.h>
#include <wx/event.h>

#include <CCA/Components/GUIBuilder/MiniCanvas.h>
#include <CCA/Components/GUIBuilder/BuilderWindow.h>
#include <CCA/Components/GUIBuilder/Connection.h>
#include <CCA/Components/GUIBuilder/NetworkCanvas.h>

#include <iostream>
#include <cmath>

namespace GUIBuilder {

//using namespace SCIRun;

BEGIN_EVENT_TABLE(MiniCanvas, wxScrolledWindow)
  EVT_PAINT(MiniCanvas::OnPaint)
  EVT_ERASE_BACKGROUND(MiniCanvas::OnEraseBackground)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(MiniCanvas, wxScrolledWindow)

  MiniCanvas::MiniCanvas(wxWindow* parent, NetworkCanvas* canvas, wxWindowID id, const wxPoint& pos, const wxSize& size) : canvas(canvas)
{
  Init();
  Create(parent, id, pos, size);
}

MiniCanvas::~MiniCanvas()
{
  delete goldenrodPen;
  delete lightGreyPen;
  delete lightGreyBrush;
}

bool MiniCanvas::Create(wxWindow *parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
  if (!wxScrolledWindow::Create(parent, id, pos, size, style)) {
    return false;
  }

  SetBackgroundStyle(wxBG_STYLE_COLOUR);
  SetBackgroundColour(BuilderWindow::BACKGROUND_COLOUR);
  //SetScrollRate(DEFAULT_SCROLLX, DEFAULT_SCROLLY);

  vBoxColor = wxTheColourDatabase->Find("GOLDENROD");
  goldenrodPen = new wxPen(vBoxColor, 1, wxSOLID);

  iRectColor = wxTheColourDatabase->Find("LIGHT GREY");
  lightGreyPen = new wxPen(iRectColor, 1, wxSOLID);
  lightGreyBrush = new wxBrush(iRectColor, wxSOLID);

  SetCursor(wxCursor(wxCURSOR_ARROW));
  return true;
}

void MiniCanvas::OnDraw(wxDC& dc)
{
  wxSize cs = canvas->GetVirtualSize();
  wxSize ms = GetSize();
  double scaleH = double( cs.GetWidth() ) / ms.GetWidth();
  double scaleV = double( cs.GetHeight() ) / ms.GetHeight();

  iRects.clear();
  canvas->GetComponentRects(iRects);
  wxRect canvasRect = canvas->GetClientRect();

  std::vector<Connection*> conns;
  canvas->GetConnections(conns);

  scaleRect(canvasRect, scaleV, scaleH);
  dc.SetPen(*goldenrodPen);
  dc.DrawRectangle(canvasRect.x, canvasRect.y, canvasRect.width, canvasRect.height);

  dc.SetBrush(*lightGreyBrush);
  dc.SetPen(*wxBLACK_PEN);
  for (std::vector<wxRect>::iterator rectIter = iRects.begin(); rectIter != iRects.end(); rectIter++) {
    scaleRect(*rectIter, scaleV, scaleH);
    dc.DrawRectangle(rectIter->x, rectIter->y, rectIter->width, rectIter->height);
  }

  dc.SetBrush(*wxTRANSPARENT_BRUSH);
  dc.SetPen(*lightGreyPen);

  const int NUM_POINTS = Connection::GetDrawingPointsSize();
  wxPoint *pointsArray = new wxPoint[NUM_POINTS];
  for (std::vector<Connection*>::iterator connIter = conns.begin(); connIter != conns.end(); connIter++) {
    (*connIter)->GetDrawingPoints(&pointsArray, NUM_POINTS);
    scalePoints(&pointsArray, NUM_POINTS, scaleV, scaleH);
    dc.DrawLines(NUM_POINTS, pointsArray, 0, 0);
  }
  delete [] pointsArray;
}

void MiniCanvas::OnPaint(wxPaintEvent& event)
{
  wxBufferedPaintDC dc(this, wxBUFFER_VIRTUAL_AREA);
  // Shifts the device origin so we don't have to worry
  // about the current scroll position ourselves.
  PrepareDC(dc);

  PaintBackground(dc);
  OnDraw(dc);
}

void MiniCanvas::PaintBackground(wxDC& dc)
{
  dc.Clear();
  wxColor backgroundColour = GetBackgroundColour();
  if (! backgroundColour.Ok()) {
    backgroundColour =
      wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE);
  }

  dc.SetBrush(wxBrush(backgroundColour));
  dc.SetPen(wxPen(backgroundColour, 1));

  wxRect windowRect(wxPoint(0, 0), GetClientSize());

  CalcUnscrolledPosition(windowRect.x, windowRect.y, &windowRect.x, &windowRect.y);
  dc.DrawRectangle(windowRect);
}

///////////////////////////////////////////////////////////////////////////
// protected member functions

void MiniCanvas::Init()
{
}

///////////////////////////////////////////////////////////////////////////
// private member functions

void MiniCanvas::scaleRect(wxRect& rect, const double scaleV, const double scaleH)
{
  rect.x = (int) ceil(rect.x / scaleH);
  rect.y = (int) ceil(rect.y / scaleV);
  rect.width = (int) ceil(rect.width / scaleH);
  rect.height = (int) ceil(rect.height / scaleV);
}

void MiniCanvas::scalePoints(wxPoint **points, const int size,
			     const double scaleV, const double scaleH)
{
  for (int i = 0; i < size; i++) {
    (*points)[i].x = (int) ceil((*points)[i].x / scaleH);
    (*points)[i].y = (int) ceil((*points)[i].y / scaleV);
  }
}

}
