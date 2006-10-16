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
 *  MainWindow.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 */

#ifndef SCIRun_Viewer_MainWindow_h
#define SCIRun_Viewer_MainWindow_h

#include <sci_wx.h>
#include <Core/CCA/spec/cca_sidl.h>

namespace Viewer {

class Colormap;
class ViewerWindow;

class MainWindow: public wxFrame {
public:
  enum {
    ID_CHECKBOX_MESH = wxID_HIGHEST,
    ID_CHECKBOX_COORDS = ID_CHECKBOX_MESH + 1,
  };
  MainWindow(wxWindow *parent,
             const char *name,
             const SSIDL::array1<double> nodes1d,
             const SSIDL::array1<int> triangles,
             const SSIDL::array1<double> solution);
  virtual ~MainWindow();

private:
  const static int X = 200;
  const static int Y = 200;
  const static int WIDTH = 500;
  const static int HEIGHT = 500;

  Colormap* cmap;
  ViewerWindow* viewer;
  wxCheckBox* meshCheckBox;
  wxCheckBox* coordsCheckBox;
};

}

#endif



