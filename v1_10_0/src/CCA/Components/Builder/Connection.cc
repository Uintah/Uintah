#include "Connection.h"

#include <iostream>
using namespace std;

Connection::Connection(Module *pU, const std::string &portname1, Module *pP, 
		       const std::string &portname2,
		       const sci::cca::ConnectionID::pointer &connID,QCanvasView *cview)
  :QCanvasPolygon(cview->canvas())

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
  int h=10; //may varie with different ports
  
  int mid;
  
  if(P.y()+h<R.y()-h){
    mid=(P.x()+R.x())/2;		
    
    QPointArray pa(12);
    int ym=(P.y()+R.y())/2;

    pa[0]=QPoint(P.x()-t,P.y());
    pa[1]=QPoint(P.x()-t,P.y()+1);
    pa[2]=QPoint(P.x()-t,P.y()+2);

    if(P.x()<=mid) pa[3]=QPoint(pa[2].x(),ym+t);
    else           pa[3]=QPoint(pa[2].x(),ym-t);
     
    pa[4]=QPoint(R.x()-t,pa[3].y());
 
    pa[5]=QPoint(pa[4].x(),R.y());
    
    pa[6]=QPoint(R.x()+t,R.y());

    if(P.x()<=mid) pa[7]=QPoint(pa[6].x(), ym-t);
    else	   pa[7]=QPoint(pa[6].x(), ym+t);
	
    pa[8]=QPoint(P.x()+t,pa[7].y());
    
    pa[9]=QPoint(pa[8].x(),P.y()+2);
    pa[10]=QPoint(pa[8].x(),P.y()+1);
    pa[11]=QPoint(pa[8].x(),P.y());
 
    setPoints(pa);

  }
  else{
    if(rUse.left()>rProvide.right()+2*t){
      mid=(rUse.left()+rProvide.right())/2;		
    }	
    else if(rProvide.left()>rUse.right()+2*t){
      mid=(rUse.right()+rProvide.left())/2;		
    }
    else{
      mid=rUse.left()<rProvide.left()?rUse.left():rProvide.left();		
      mid-=2*t;
    }
QPointArray pa(12);

	pa[0]=QPoint(P.x()-t,P.y());
	if(P.x()<mid) pa[1]=QPoint(pa[0].x(),P.y()+h+t);
	else          pa[1]=QPoint(pa[0].x(),P.y()+h-t);
 	
	if(P.y()+h<R.y()-h) pa[2]=QPoint(mid-t,pa[1].y());
	else                pa[2]=QPoint(mid+t,pa[1].y());

	if(R.x()>mid) pa[3]=QPoint(pa[2].x(),R.y()-h+t);
	else					pa[3]=QPoint(pa[2].x(),R.y()-h-t);

	pa[4]=QPoint(R.x()-t,pa[3].y());

	pa[5]=QPoint(pa[4].x(),R.y());

	pa[6]=QPoint(R.x()+t, pa[5].y());
	
	if(R.x()>mid) pa[7]=QPoint(pa[6].x(), R.y()-h-t);
	else					pa[7]=QPoint(pa[6].x(), R.y()-h+t);
	
	if(P.y()+h<R.y()-h)	pa[8]=QPoint(mid+t,pa[7].y());
	else	              pa[8]=QPoint(mid-t,pa[7].y());
	
	if(P.x()<mid) pa[9]=QPoint(pa[8].x(),P.y()+h-t);
	else          pa[9]=QPoint(pa[8].x(),P.y()+h+t);

	pa[10]=QPoint(P.x()+t,pa[9].y());

	pa[11]=QPoint(pa[10].x(),P.y());

	setPoints(pa);
  }

/*	QPointArray pa(6);
	int h=5; //may varies with different ports
	pa[0]=P;
	pa[1]=QPoint(P.x(),P.y()+h);
	pa[2]=QPoint(mid,P.y()+h);
	pa[3]=QPoint(mid,R.y()-h);
	pa[4]=QPoint(R.x(),R.y()-h);
	pa[5]=R;
*/
  /*
	QPointArray pa(12);

	pa[0]=QPoint(P.x()-t,P.y());
	if(P.x()<mid) pa[1]=QPoint(pa[0].x(),P.y()+h+t);
	else          pa[1]=QPoint(pa[0].x(),P.y()+h-t);
 	
	if(P.y()+h<R.y()-h) pa[2]=QPoint(mid-t,pa[1].y());
	else                pa[2]=QPoint(mid+t,pa[1].y());

	if(R.x()>mid) pa[3]=QPoint(pa[2].x(),R.y()-h+t);
	else					pa[3]=QPoint(pa[2].x(),R.y()-h-t);

	pa[4]=QPoint(R.x()-t,pa[3].y());

	pa[5]=QPoint(pa[4].x(),R.y());

	pa[6]=QPoint(R.x()+t, pa[5].y());
	
	if(R.x()>mid) pa[7]=QPoint(pa[6].x(), R.y()-h-t);
	else					pa[7]=QPoint(pa[6].x(), R.y()-h+t);
	
	if(P.y()+h<R.y()-h)	pa[8]=QPoint(mid+t,pa[7].y());
	else	              pa[8]=QPoint(mid-t,pa[7].y());
	
	if(P.x()<mid) pa[9]=QPoint(pa[8].x(),P.y()+h-t);
	else          pa[9]=QPoint(pa[8].x(),P.y()+h+t);

	pa[10]=QPoint(P.x()+t,pa[9].y());

	pa[11]=QPoint(pa[10].x(),P.y());

	setPoints(pa);
  */

}

void Connection::drawShape ( QPainter & p)
{
	QPointArray par(6);
	for(int i=0;i<6;i++)	par[i]=(points()[i]+points()[11-i])/2;

	p.setPen(QPen(color,4));
	p.setBrush(blue);
	p.drawPolyline(par);
	//p.drawPolygon(points());
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



