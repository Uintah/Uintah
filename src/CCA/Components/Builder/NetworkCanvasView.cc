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
 *  NetworkCanvasView.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *  Modified by:
 *   Keming Zhang
 *   March 2002
 */

#include <CCA/Components/Builder/NetworkCanvasView.h>
#include <qwmatrix.h>
#include <Core/CCA/spec/cca_sidl.h>

//using namespace SCIRun;

NetworkCanvasView::NetworkCanvasView(QCanvas* canvas, QWidget* parent)
				     
  : QCanvasView(canvas, parent)
{
	moving=connecting=0;
}

NetworkCanvasView::~NetworkCanvasView()
{
}

void NetworkCanvasView::contentsMousePressEvent(QMouseEvent* e)
{
	if(moving || connecting) return;
	//IMPORTANT NOTES: e->pos() returns the mouse point in the canvas coordinates	
	QPoint p = contentsToViewport(e->pos());
	QWidget *who=childAt(p);
	
        if(e->button()==Qt::RightButton){
          QCanvasItemList lst=canvas()->collisions(e->pos());
          cerr<<"lst.size="<<lst.size()<<endl;
          if(lst.size()>0) removeConnection(lst[0]);
        }

	else if(e->button()==Qt::MidButton){
		  for(std::vector<Module*>::iterator it=modules.begin(); it!=modules.end(); it++) {
		     if( (QWidget*)(*it)==who ){
			QPoint localpos=e->pos()-QPoint(childX(who), childY(who));
		        cerr<<"local point="<<localpos.x()<<" "<<localpos.y()<<endl;	
			if((*it)->clickedPort(localpos, porttype, portnum)){
				connecting= *it;
				return;
			}
		     }
		}
	}
        else if(e->button()==Qt::LeftButton){
		  for(std::vector<Module*>::iterator it=modules.begin(); it!=modules.end(); it++) {
		     if( (QWidget*)(*it)==who ){
             		   moving = *it;
               		 moving_start = p;
           	          return;
		     }	
		}
	    }	
}

void NetworkCanvasView::contentsMouseReleaseEvent(QMouseEvent* e)
{
        //IMPORTANT NOTES: e->pos() returns the mouse point in the canvas coordinates
        //cerr<<"MousePress e->pos()="<<e->pos().x()<<endl;
        QPoint p = contentsToViewport(e->pos());
        QWidget *who=childAt(p);
  if(connecting)
  for(std::vector<Module*>::iterator it=modules.begin(); it!=modules.end(); it++) {
     if( (QWidget*)(*it)==who ){
             if( (QWidget*)(*it)==who ){
                        QPoint localpos=e->pos()-QPoint(childX(who), childY(who));
                        cerr<<"local point="<<localpos.x()<<" "<<localpos.y()<<endl;
                       	int portnum1=0;
			Module::PortType porttype1=Module::USES;
			 if((*it)->clickedPort(localpos, porttype1, portnum1)){
				if(connecting!=(*it) && porttype!=porttype1){ 
					if(porttype==Module::USES)
						addConnection(connecting, portnum, *it, portnum1); 
		                	else
						addConnection(*it, portnum1, connecting, portnum); 

			 cerr<<"Connection added"<<endl;
				}
                        }
		   break;		
             }
     }
  }
  connecting=0;
  moving = 0;
}



void NetworkCanvasView::contentsMouseMoveEvent(QMouseEvent* e)
{
  if ( moving ) {
		int dx=0;
		int dy=0;
		QPoint p = contentsToViewport(e->pos());
		//newX, newY are in canvas coordinates
		int newX=childX(moving) + p.x() - moving_start.x();
		int newY=childY(moving) + p.y() - moving_start.y();
		QPoint pLeftTop = contentsToViewport(QPoint(newX, newY));

		QPoint mouse=e->globalPos();
		if(pLeftTop.x()<0){
	 		newX-=pLeftTop.x();
			if(p.x()<0){
				mouse.setX(mouse.x()-p.x());
				p.setX(0);
			  QCursor::setPos(mouse);	
			}
			dx=-1;
		}

		if(pLeftTop.y()<0){
	 		newY-=pLeftTop.y();		
			if(p.y()<0){
				mouse.setY(mouse.y()-p.y());
				p.setY(0);
			  QCursor::setPos(mouse);	
			}
			dy=-1;
		}

		int cw=contentsRect().width();
		int mw=moving->frameSize().width();
		if(pLeftTop.x()>cw-mw){
	 		newX-=pLeftTop.x()-(cw-mw);
			if(p.x()>cw){
				mouse.setX(mouse.x()-(p.x()-cw) );
				p.setX(cw-mw);
			  QCursor::setPos(mouse);	
			}
			dx=1;
		}

		int ch=contentsRect().height();
		int mh=moving->frameSize().height();
		if(pLeftTop.y()>ch-mh){
	 		newY-=pLeftTop.y()-(ch-mh);
			if(p.y()>ch){
				mouse.setY(mouse.y()-(p.y()-ch) );
				p.setY(ch);
			  QCursor::setPos(mouse);	
			}
			dy=1;
		}

		//if(pLeftTop.x()<0 || pLeftTop.y()<0)
		//if(  ! canvas()->rect().contains( QRect(pLeftTop, moving->frameSize()), true) ) return;

		moving_start = p;
		moveChild(moving, newX, newY);

		if(dx || dy) scrollBy(dx*5,dy*5);



		for(std::vector<Connection*>::iterator ct=connections.begin(); ct!=connections.end(); ct++) {
    	if( (*ct)->isConnectedTo(moving) ) 	(*ct)->resetPoints();						
		}
    canvas()->update();
  }
}

void NetworkCanvasView::addModule(const char *name, gov::cca::ports::UIPort::pointer &uip,CIA::array1<std::string> & up, CIA::array1<std::string> &pp , const gov::cca::ComponentID::pointer &cid)
{
	
	Module *module=new Module(this,name,uip,up,pp, cid);
        addChild(module,20, 20);
	modules.push_back(module);
	module->show();		
}

void NetworkCanvasView::addConnection(Module *m1,int portnum1,  Module *m2, int portnum2)
{

gov::cca::ports::BuilderService::pointer bs = pidl_cast<gov::cca::ports::BuilderService::pointer>(services->getPort("cca.builderService"));
  if(bs.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
  }
  bs->connect(m1->cid, m1->usesPortName(portnum1), m2->cid, m2->providesPortName(portnum2));

  services->releasePort("cca.builderService");

	Connection *con=new Connection(m1,portnum1, m2,portnum2, this);


	con->show();
	connections.push_back(con);
	canvas()->update();

}

void NetworkCanvasView::removeConnection(QCanvasItem *c)
{
	std::vector<Connection *>::iterator iter=connections.begin();
	while(iter!=connections.end()){
	   if((QCanvasItem*) (*iter)==c){
		connections.erase(iter);
		//canvas()->removeItem(c);
                
		//canvas()->removeChild(c); //
                cerr<<"connection.size()="<<connections.size()<<endl;
                cerr<<"all item.size before del="<<canvas()->allItems().size()<<endl;
		delete c;
                cerr<<"all item.size after del="<<canvas()->allItems().size()<<endl;
	        canvas()->update();
                cerr<<"all item.size after del="<<canvas()->allItems().size()<<endl;

           }			
	}
}



void NetworkCanvasView::setServices(const gov::cca::Services::pointer &services)
{
	this->services=services;
}
