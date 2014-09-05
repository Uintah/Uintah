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
 *  MeshWindow.h:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <qapplication.h>
#include <qframe.h>
#include <qpainter.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "vector2d.h"
#include "Delaunay.h"


class MeshWindow:public QFrame
{
 public:
  MeshWindow( QWidget *parent, Delaunay *mesh);
  int height();
  int width();
  int w, h;
  int meshToWindowX(double x);
  int meshToWindowY(double y);
  double windowToMeshX(int x);
  double windowToMeshY(int y);
 protected:
  void	paintEvent(QPaintEvent*e);
  void	MeshWindow::mousePressEvent(QMouseEvent* e);
  Delaunay *mesh;
  int border;
};

