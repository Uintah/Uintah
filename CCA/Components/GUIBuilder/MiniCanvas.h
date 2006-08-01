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


#ifndef CCA_Components_GUIBuilder_MiniCanvas_h
#define CCA_Components_GUIBuilder_MiniCanvas_h

#include <map>
#include <string>

class wxRect;
class wxPoint;
class wxScrolledWindow;
class wxPen;
class wxBrush;

namespace GUIBuilder {

class BuilderWindow;
class NetworkCanvas;
class Connection;

// Scrolled window???

class MiniCanvas : public wxScrolledWindow {
public:
  MiniCanvas(wxWindow* parent, NetworkCanvas* canvas, wxWindowID id, const wxPoint& pos, const wxSize& size);
  virtual ~MiniCanvas();

  bool Create(wxWindow *parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style = wxHSCROLL|wxVSCROLL|wxSUNKEN_BORDER|wxRETAINED);

  void OnPaint(wxPaintEvent& event);
  void PaintBackground(wxDC& dc);
  void OnDraw(wxDC& dc);

protected:
  MiniCanvas() { Init(); }
  void Init();
  void OnEraseBackground(wxEraseEvent& event) {}

private:
  void scaleRect(wxRect& rect, const double scaleV, const double scaleH);
  void scalePoints(wxPoint **points, const int size, const double scaleV, const double scaleH);

  NetworkCanvas *canvas;
  std::vector<wxRect> iRects;
  std::vector<Connection*> conns;

  wxColor vBoxColor;
  wxColor iRectColor;
  wxPen* goldenrodPen;
  wxPen* lightGreyPen;
  wxBrush* lightGreyBrush;

  DECLARE_EVENT_TABLE()
  DECLARE_DYNAMIC_CLASS(MiniCanvas)
};

}

#endif
