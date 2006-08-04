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
 *  ViewerWindow.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef Viewer_ViewWinodw_h
#define Viewer_ViewWinodw_h

#include <qapplication.h>
#include <qpushbutton.h>
#include <qmainwindow.h>
#include <qmessagebox.h>
#include <qapplication.h>
#include <qframe.h>
#include <qpainter.h>
#include <qcolor.h>
#include <qscrollview.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include <Core/CCA/spec/cca_sidl.h>

#include "vector2d.h"
#include "Colormap.h"

class ViewerWindow:public QFrame
{
  Q_OBJECT
public:
  ViewerWindow( QWidget *parent,  Colormap *cmap, const std::vector<double> nodes1d, 
		const std::vector<int> triangles, 
		const std::vector<double> solution);
  int height();
  int width();
  QPoint toViewport(vector2d v);
  vector2d toField(QPoint p);
  double fieldWidth();
  double fieldHeight();
public slots:
  void refresh(const QString &type);
  void toggleMesh();
  void toggleCoordinates();

protected:
  void	paintEvent(QPaintEvent*e);
  void	ViewerWindow::mousePressEvent(QMouseEvent* e);
  bool nodeInTriangle(vector2d vp, vector2d v1, vector2d v2, vector2d v3);
  int w, h;
  int borderX, borderY;
  Colormap *cmap;
  double minx, miny, maxx, maxy;
  SSIDL::array1<double> nodes1d;
  SSIDL::array1<int> triangles;
  SSIDL::array1<double> solution; 
  void setBackground(const QString &type);
  bool showMesh;
  bool showCoordinates;

};


#endif
