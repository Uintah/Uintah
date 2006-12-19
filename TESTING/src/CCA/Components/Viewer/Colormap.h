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
 *  Colormap.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *  Ported to wxWidgets:
 *
 */

#ifndef Viewer_Colormap_h
#define Viewer_Colormap_h

#include <sci_wx.h>
#include <wx/gdicmn.h>

namespace Viewer {

class Colormap : public wxPanel {
public:
  Colormap(wxWindow *parent, const wxString& type = "Gray", double min = 0.0, double max = 1.0);

  void setValues(double min, double max);
  void setType(const wxString& type);
  wxColor getColor(double value);

  int height();

protected:
  void OnPaint(wxPaintEvent& event);

private:
  wxString type;
  double minVal, maxVal;
  int borderY;

  DECLARE_EVENT_TABLE()
};

}

#endif


