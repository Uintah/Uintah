/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
