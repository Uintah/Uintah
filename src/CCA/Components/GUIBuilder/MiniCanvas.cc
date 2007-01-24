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
#include <wx/string.h>
#include <wx/event.h>

#include <CCA/Components/GUIBuilder/MiniCanvas.h>
#include <CCA/Components/GUIBuilder/BuilderWindow.h>
#include <CCA/Components/GUIBuilder/Connection.h>
#include <CCA/Components/GUIBuilder/NetworkCanvas.h>

#include <iostream>
#include <cmath>

namespace GUIBuilder {

BEGIN_EVENT_TABLE(MiniCanvas, wxScrolledWindow)
  EVT_PAINT(MiniCanvas::OnPaint)
  EVT_LEFT_DOWN(MiniCanvas::OnLeftDown)
  EVT_LEFT_UP(MiniCanvas::OnLeftUp)
  EVT_MOTION(MiniCanvas::OnMouseMove)
  EVT_ERASE_BACKGROUND(MiniCanvas::OnEraseBackground)
END_EVENT_TABLE()

IMPLEMENT_DYNAMIC_CLASS(MiniCanvas, wxScrolledWindow)

MiniCanvas::MiniCanvas(wxWindow* parent, BuilderWindow* bw, NetworkCanvas* canvas,
                       wxWindowID id, const wxPoint& pos, const wxSize& size)
  : builderWindow(bw), canvas(canvas), insideViewport(false)
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
  double scaleH, scaleV;
  getScale(scaleH, scaleV);

  iRects.clear();
  canvas->GetComponentRects(iRects);
  viewportRect = canvas->GetClientRect();

  std::vector<Connection*> conns;
  canvas->GetConnections(conns);

  scaleRect(viewportRect, scaleH, scaleV);
  dc.SetPen(*goldenrodPen);
  dc.DrawRectangle(viewportRect.x, viewportRect.y, viewportRect.width, viewportRect.height);

  dc.SetBrush(*lightGreyBrush);
  dc.SetPen(*wxBLACK_PEN);
  for (std::vector<wxRect>::iterator rectIter = iRects.begin(); rectIter != iRects.end(); rectIter++) {
    scaleRect(*rectIter, scaleH, scaleV);
    dc.DrawRectangle(rectIter->x, rectIter->y, rectIter->width, rectIter->height);
  }

  dc.SetBrush(*wxTRANSPARENT_BRUSH);
  dc.SetPen(*lightGreyPen);

  const int NUM_POINTS = Connection::GetDrawingPointsSize();
  wxPoint *pointsArray = new wxPoint[NUM_POINTS];
  for (std::vector<Connection*>::iterator connIter = conns.begin(); connIter != conns.end(); connIter++) {
    (*connIter)->GetDrawingPoints(&pointsArray, NUM_POINTS);
    scalePoints(&pointsArray, NUM_POINTS, scaleH, scaleV);
    dc.DrawLines(NUM_POINTS, pointsArray, 0, 0);
  }
  delete [] pointsArray;
}

void MiniCanvas::OnLeftDown(wxMouseEvent& event)
{
  wxClientDC dc(this);
  DoPrepareDC(dc);
  wxPoint ep = event.GetLogicalPosition(dc);
  if (viewportRect.Inside(ep.x, ep.y)) {
    insideViewport = true;
  }
}

void MiniCanvas::OnLeftUp(wxMouseEvent& event)
{
  wxClientDC dc(this);
  DoPrepareDC(dc);
  wxPoint ep = event.GetLogicalPosition(dc);
  insideViewport = false;
  scrollCanvas(ep);
}

void MiniCanvas::OnMouseMove(wxMouseEvent& event)
{
  wxClientDC dc(this);
  DoPrepareDC(dc);
  wxPoint ep = event.GetLogicalPosition(dc);
  builderWindow->DisplayMousePosition(ep);
  if (insideViewport && event.Dragging()) {
    scrollCanvas(ep);
  }
  Refresh();
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

void MiniCanvas::getScale(double& scaleH, double& scaleV)
{
  wxSize cs = canvas->GetVirtualSize();
  wxSize ms = GetSize();
  scaleH = double( cs.GetWidth() ) / ms.GetWidth();
  scaleV = double( cs.GetHeight() ) / ms.GetHeight();
}

void MiniCanvas::scrollCanvas(const wxPoint& point)
{
  double scaleH, scaleV;
  getScale(scaleH, scaleV);
  //SetTargetWindow(canvas);
  canvas->Scroll((point.x * scaleH) / NetworkCanvas::DEFAULT_SCROLLX, (point.y * scaleV) / NetworkCanvas::DEFAULT_SCROLLY);
}

///////////////////////////////////////////////////////////////////////////
// utility functions

void scaleRect(wxRect& rect, const double scaleH, const double scaleV)
{
  rect.x = (int) ceil(rect.x / scaleH);
  rect.y = (int) ceil(rect.y / scaleV);
  rect.width = (int) ceil(rect.width / scaleH);
  rect.height = (int) ceil(rect.height / scaleV);
}

void scalePoints(wxPoint **points, const int size, const double scaleH, const double scaleV)
{
  for (int i = 0; i < size; i++) {
    (*points)[i].x = (int) ceil((*points)[i].x / scaleH);
    (*points)[i].y = (int) ceil((*points)[i].y / scaleV);
  }
}

}
