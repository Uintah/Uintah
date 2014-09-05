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
 *  MeshWindow.cc:
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
#include "MeshWindow.h"
#include <Core/CCA/spec/cca_sidl.h>

MeshWindow::MeshWindow( QWidget *parent, Delaunay *mesh)
  : QFrame( parent)
{
  w=h=500;
  border=20;
  setGeometry(QRect(200,200,w,h));
  this->mesh=mesh;
}

//define a smaller window height
int MeshWindow::height()
{
  return QFrame::height()-border*2;
}

//define a smaller window width
int MeshWindow::width()
{
  return QFrame::width()-border*2;
}


//plot the triangles and/or cirlcles
void MeshWindow::paintEvent(QPaintEvent* e)
{
  
  QFrame::paintEvent(e);
  QPainter p(this);
	
  int r=5;
  std::vector<vector2d> pts=mesh->getNodes();
  std::vector<Triangle> tri=mesh->getTriangles();
  std::vector<Circle> circles=mesh->getCircles();
  std::vector<Boundary> boundaries=mesh->getBoundaries();

  if(pts.empty()) return;

  p.setBrush(NoBrush);
  for(unsigned i=0;i<circles.size();i++){
    //check if the triangle does not contain the bounding vertices
    //if not, plot it.
    if(true){
      //(tri[i].index[0]>=4 && tri[i].index[1]>=4 && tri[i].index[0]>=4){
      
      //int x=meshToWindowX(circles[i].center.x);		
      //int y=meshToWindowY(circles[i].center.y);		
      //int rx=int(circles[i].radius*width()/mesh->width());
      //int ry=int(circles[i].radius*height()/mesh->height());
      
      
      int x1=meshToWindowX(pts[tri[i].index[0]].x); 
      int y1=meshToWindowY(pts[tri[i].index[0]].y);
      int x2=meshToWindowX(pts[tri[i].index[1]].x);
      int y2=meshToWindowY(pts[tri[i].index[1]].y);
      int x3=meshToWindowX(pts[tri[i].index[2]].x);
      int y3=meshToWindowY(pts[tri[i].index[2]].y);
      p.setPen(red);

      p.drawLine(x1, y1, x2, y2);
      p.drawLine(x2, y2, x3, y3);
      p.drawLine(x3, y3, x1, y1);
    }
  }

  p.setBrush(black);
  for(unsigned i=0;i<pts.size();i++){
    int x=meshToWindowX(pts[i].x);
    int y=meshToWindowY(pts[i].y);
    p.drawEllipse(x-r, y-r, r+r, r+r);
    char s[5];
    sprintf(s,"%d",i-4);
    p.drawText(x+r,y+r,s);
  }

  //plot boundaries
  p.setBrush(green);
  for(unsigned b=0; b<boundaries.size(); b++){
    for(unsigned i=0;i<boundaries[b].index.size();i++){
      int x=meshToWindowX(pts[boundaries[b].index[i]].x);
      int y=meshToWindowY(pts[boundaries[b].index[i]].y);
      p.drawEllipse(x-r, y-r, r+r, r+r);
      
    }
  }
  
}


//left button to add one node
void MeshWindow::mousePressEvent( QMouseEvent * e)
{
  /*	QPoint p=e->pos();
	if(e->button()==LeftButton){
	  //vector2d v=vector2d(windowToMeshX(p.x()), windowToMeshY(p.y()));
	  //mesh->addNode(v);
	  mesh->triangulation();
	  update();
	}
	else if(e->button()==RightButton){
	  
	}
  */
}

int MeshWindow::meshToWindowX(double x)
{
  return border+int( (x-mesh->minX())*width()/mesh->width());
}

int MeshWindow::meshToWindowY(double y)
{
  return border+height()-int( (y-mesh->minY())*height()/mesh->height());
}

double MeshWindow::windowToMeshX(int x)
{
  return mesh->minX()+mesh->width()*(x-border)/width();
}

double MeshWindow::windowToMeshY(int y)
{
  return mesh->minY()+mesh->height()*(height()-(y-border))/height();
}






