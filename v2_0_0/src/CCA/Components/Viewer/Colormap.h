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
 *  Colormap.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 */

#ifndef Viewer_Colormap_h
#define Viewer_Colormap_h

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
#include "vector2d.h"

class Colormap:public QFrame
{
public:
  Colormap( QWidget *parent, const QString &type="Gray", double min=0.0, double max=1.0);
  void setValues(double min, double max);
  int height();
  QColor getColor(double value);
  void setType(const QString& type);
protected:
  void	paintEvent(QPaintEvent*e);
  double minVal, maxVal;
  QString type;
  int borderY;
};

#endif


