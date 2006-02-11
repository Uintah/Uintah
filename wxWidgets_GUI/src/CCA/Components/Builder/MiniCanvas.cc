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
#include <wx/event.h>

#include <CCA/Components/Builder/MiniCanvas.h>
#include <CCA/Components/Builder/BuilderWindow.h>
#include <CCA/Components/Builder/Connection.h>
#include <CCA/Components/Builder/NetworkCanvas.h>

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
}

void MiniCanvas::Init()
{
}

bool MiniCanvas::Create(wxWindow *parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
  if (!wxScrolledWindow::Create(parent, id, pos, size, style)) {
    return false;
  }

  SetBackgroundStyle(wxBG_STYLE_COLOUR);
  SetBackgroundColour(BuilderWindow::BACKGROUND_COLOUR);
  //SetScrollRate(DEFAULT_SCROLLX, DEFAULT_SCROLLY);

  SetCursor(wxCursor(wxCURSOR_ARROW));
  return true;
}

void MiniCanvas::SetCanvasShapes()
{
std::cerr << "MiniCanvas::SetCanvasShapes()" << std::endl;

  Refresh();
}

void MiniCanvas::OnDraw(wxDC& dc)
{
  wxSize cs = canvas->GetVirtualSize();
  wxSize ms = GetSize();
  double scaleH = double( cs.GetWidth() ) / ms.GetWidth();
  double scaleV = double( cs.GetHeight() ) / ms.GetHeight();

  iRects.clear();
  canvas->GetComponentRects(iRects);

  // get connections lines

  wxRect canvasRect = canvas->GetClientRect();
  scaleRect(canvasRect, scaleV, scaleH);
  dc.SetPen(wxPen(*wxBLACK, 1, wxSOLID));
  dc.DrawRectangle(canvasRect.x, canvasRect.y, canvasRect.width, canvasRect.height);
  //dc.DrawRectangle(ceil(canvasRect.x / scaleH), ceil(canvasRect.y / scaleV), ceil(canvasRect.width / scaleH), ceil(canvasRect.height / scaleV));

  dc.SetBrush(*wxWHITE);
  for (std::vector<wxRect>::iterator it = iRects.begin(); it != iRects.end(); it++) {
    scaleRect(*it, scaleV, scaleH);
    dc.DrawRectangle(it->x, it->y, it->width, it->height);
  }
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
  wxColour backgroundColour = GetBackgroundColour();
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
// private member functions

void MiniCanvas::scaleRect(wxRect& rect, double scaleV, double scaleH)
{
  rect.x = (int) ceil(rect.x / scaleH);
  rect.y = (int) ceil(rect.y / scaleV);
  rect.width = (int) ceil(rect.width / scaleH);
  rect.height = (int) ceil(rect.height / scaleV);
}

}
