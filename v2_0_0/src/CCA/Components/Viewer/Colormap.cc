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
 *  Colormap.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

 
#include <qsizepolicy.h>
#include <iostream>

#include "Colormap.h"
using namespace std;

Colormap::Colormap( QWidget *parent, const QString &type, double min, double max) 
  : QFrame( parent )
{
  setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding, 30,100)); 
  setType(type);
  setValues(min,max);
  setFrameStyle(Panel|Sunken);
  setLineWidth(2);
  borderY=12;
}

int Colormap::height()
{
  return QFrame::height()-borderY*2;
}
void Colormap::setType(const QString &type)
{
  this->type=type;
  update();
}

void Colormap::setValues(double min, double max)
{
  minVal=min;
  maxVal=max;
}

QColor Colormap::getColor(double value)
{
  double val=(value-minVal)/(maxVal-minVal);
  if(type=="Color"){
    double r,g,b;
    if(val<=0.5){
      r=0;
      g=val*2;
      b=1-g;
    }
    else{
      b=0;
      g=(1-val)*2;
      r=1-g;
    }
    return QColor(int(r*255),int(g*255),int(b*255));
  }
  else if(type=="Gray"){
    int g=int(val*255);
    if(g>255) g=255;
    if(g<0) g=0;
    return QColor(g,g,g);
  }
  else{
    cerr<<"unkown colormap type"<<endl;
    return QColor(0,0,0);
  }
}

void Colormap::paintEvent(QPaintEvent* )
{
  QPainter p(this);
  for(int y=0; y<=height(); y++){
    double val=double(height()-y)/height();
    p.setPen(getColor(minVal+(maxVal-minVal)*val));
    p.drawLine(0,borderY+y,width(),borderY+y);
  }

  //display values
  
  int ny=5;

  p.setPen(QColor(255,0,255));
  //  p.setBrush(white);
  for(int iy=0; iy<=ny; iy++){
    double value=minVal+iy*(maxVal-minVal)/ny;
    int y=borderY+(ny-iy)*height()/ny;
    char s[10];
    if( fabs(value)>0.01 && fabs(value)<100){
	sprintf(s,"%6.4lf",value);
    }
    else{
      sprintf(s,"%7.2lf",value);
    }
    sprintf(s,"%7.2lf",value);
    
    int w=12*6;
    int h=14;
    QRect r(width()/2-w/2,y-h/2, w, h);
    
    p.drawText(r,Qt::AlignCenter, s);
  }
}  

