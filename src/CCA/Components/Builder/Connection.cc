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

#include "Connection.h"

#include <iostream>
using namespace std;

Connection::Connection(Module *pU, const std::string &portname1, Module *pP, 
		       const std::string &portname2,
		       const sci::cca::ConnectionID::pointer &connID,QCanvasView *cview)
  : QCanvasPolygon(cview->canvas())

{
	this->portname1=portname1;
	this->portname2=portname2;
	this->connID=connID;
	pUse=pU;
	pProvide=pP;
	cv=cview;
	resetPoints();
	setDefault();	
}

bool Connection::isConnectedTo(Module *who)
{
    return who==pUse || who==pProvide;
}


void Connection::resetPoints()
{
  QPoint P=pUse->usePortPoint(portname1)+QPoint(cv->childX(pUse),cv->childY(pUse));
  QPoint R=pProvide->providePortPoint(portname2)+QPoint(cv->childX(pProvide),cv->childY(pProvide));
  QRect rUse(cv->childX(pUse),cv->childY(pUse),pUse->width(),pUse->height() );
  QRect rProvide(cv->childX(pProvide),cv->childY(pProvide),pProvide->width(),pProvide->height() );

  int t=4;
  int h=10; //may vary with different ports
  
  int mid;
  
  if (P.x()+h<R.x()-h) {
    mid=(P.y()+R.y())/2;		
    
    QPointArray pa(NUM_POINTS);
    int xm=(P.x()+R.x())/2;

    pa[0]=QPoint(P.x(),P.y()-t);
    pa[1]=QPoint(P.x()+1,P.y()-t);
    pa[2]=QPoint(P.x()+2,P.y()-t);

    if(P.y()<=mid) pa[3]=QPoint(xm+t, pa[2].y());
    else           pa[3]=QPoint(xm-t, pa[2].y());
     
    pa[4]=QPoint(pa[3].x(),R.y()-t);
 
    pa[5]=QPoint(R.x(),pa[4].y());
    
    pa[6]=QPoint(R.x(), R.y()+t);

    if(P.y()<=mid) pa[7]=QPoint(xm-t,pa[6].y());
    else	   pa[7]=QPoint(xm+t,pa[6].y());
	
    pa[8]=QPoint(pa[7].x(),P.y()+t);
    
    pa[9]=QPoint(P.x()+2,pa[8].y());
    pa[10]=QPoint(P.x()+1,pa[8].y());
    pa[11]=QPoint(P.x(),pa[8].y());
 
    setPoints(pa);

  } else{
    if (rUse.top()>rProvide.bottom()+2*t) {
      mid=(rUse.top()+rProvide.bottom())/2;		
    } else if (rProvide.top()>rUse.bottom()+2*t) {
      mid=(rUse.bottom()+rProvide.top())/2;		
    } else {
      mid=rUse.top()<rProvide.top()?rUse.top():rProvide.top();		
      mid-=2*t;
    }
    QPointArray pa(NUM_POINTS);
    
    pa[0]=QPoint(P.x(),P.y()-t);
    if(P.y()<mid) pa[1]=QPoint(P.x()+h+t,pa[0].y());
    else          pa[1]=QPoint(P.x()+h-t,pa[0].y());
    
    if(P.x()+h<R.x()-h) pa[2]=QPoint(pa[1].x(),mid-t);
    else                pa[2]=QPoint(pa[1].x(),mid+t);
    
    if(R.y()>mid) pa[3]=QPoint(R.x()-h+t,pa[2].y());
    else	pa[3]=QPoint(R.x()-h-t,pa[2].y());

    pa[4]=QPoint(pa[3].x(),R.y()-t);
    
    pa[5]=QPoint(R.x(),pa[4].y());
    
    pa[6]=QPoint(pa[5].x(),R.y()+t);
    
    if(R.y()>mid) pa[7]=QPoint(R.x()-h-t,pa[6].y());
    else	pa[7]=QPoint(R.x()-h+t,pa[6].y());
    
    if(P.x()+h<R.x()-h)	pa[8]=QPoint(pa[7].x(),mid+t);
    else	              pa[8]=QPoint(pa[7].x(),mid-t);
    
    if(P.y()<mid) pa[9]=QPoint(P.x()+h-t,pa[8].y());
    else          pa[9]=QPoint(P.x()+h+t,pa[8].y());
    
    pa[10]=QPoint(pa[9].x(),P.y()+t);
    
    pa[11]=QPoint(P.x(),pa[10].y());

    setPoints(pa);
  }
}

void Connection::drawShape(QPainter& p)
{
    QPointArray par(NUM_DRAW_POINTS);
    Connection::drawConnection(p, points(), par);

    p.setPen( QPen(color, 4) );
    p.setBrush(Qt::blue);
    p.drawPolyline(par);
}

sci::cca::ConnectionID::pointer Connection::getConnectionID()
{
  return connID;
}

void Connection:: setDefault()
{
  color=yellow;
}

void Connection:: highlight()
{
  color=red;
}

Module * Connection::getUsesModule()
{
  return pUse;
}

Module * Connection::getProvidesModule()
{
  return pProvide;
}

std::string Connection::getUsesPortName()
{
  return portname1;    
}

std::string Connection::getProvidesPortName()
{
  return portname2;
}

std::string Connection::getConnectionType()
{
  return "Connection";
}

MiniConnection::MiniConnection(QCanvasView *cv, QPointArray pa,
				double scaleX, double scaleY)
  : QCanvasPolygon(cv->canvas())
{
    scalePoints(&pa, scaleX, scaleY);
}

void MiniConnection::scalePoints(QPointArray *pa, double scaleX, double scaleY)
{
    for (unsigned int i = 0; i < Connection::NUM_POINTS; i++) {
	pa->setPoint(i, int( (*pa)[i].x() / scaleX ), int( (*pa)[i].y() / scaleY ));
    }
    this->setPoints(*pa);
}

void MiniConnection::drawShape(QPainter & p)
{
    QPointArray par(Connection::NUM_DRAW_POINTS);
    Connection::drawConnection(p, points(), par);

    p.setPen( QPen(Qt::yellow, 0) );
    p.setBrush(Qt::black);
    p.drawPolyline(par);
}
