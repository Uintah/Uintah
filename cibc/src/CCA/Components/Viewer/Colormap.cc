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

