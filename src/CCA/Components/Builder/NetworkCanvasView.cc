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
#include <CCA/Components/Builder/BuilderWindow.h>
#include <qwmatrix.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <iostream>
#include <qscrollview.h>
#include <qevent.h>

NetworkCanvasView::NetworkCanvasView(BuilderWindow* p2BuilderWindow,
					QCanvas* canvas,
					QWidget* parent)
				     
  : QCanvasView(canvas, parent)
{
	moving = connecting = 0;
	highlightedConnection = 0;
	this->p2BuilderWindow = p2BuilderWindow;
	connect(horizontalScrollBar(), SIGNAL(sliderMoved(int)),
		p2BuilderWindow, SLOT(updateMiniView()));
	connect(verticalScrollBar(), SIGNAL(sliderMoved(int)),
		p2BuilderWindow, SLOT(updateMiniView()));
}

NetworkCanvasView::~NetworkCanvasView()
{
}

void NetworkCanvasView::contentsMousePressEvent(QMouseEvent* e)
{
    if (moving || connecting) {
	return;
    }
    // IMPORTANT NOTES: e->pos() returns the mouse point in the canvas coordinates	
    QPoint p = contentsToViewport(e->pos());
    QWidget *who = childAt(p);
	
    if (e->button() == Qt::RightButton) {
	QCanvasItemList lst = canvas()->collisions(e->pos());
	//	std::cerr << "Qt::RightButton (" << e->pos().x() << "," << e->pos().y() << "): lst.size=" << lst.size() << std::endl;
	if (lst.size() > 0) {
	    removeConnection(lst[0]);
	}
    } else if (e->button() == Qt::LeftButton) {
      //std::cerr << "Qt::LeftButton: pos=" << e->pos().x() << " " << e->pos().y() << std::endl;
      for (std::vector<Module*>::iterator it = modules.begin();
	   it != modules.end();
	   it++) {
	if ((QWidget*)(*it) == who) {
	  QPoint localpos = e->pos()-QPoint(childX(who), childY(who));
	  //std::cerr << "local point=" << localpos.x() << " " << localpos.y() << std::endl;	
	  if ((*it)->clickedPort(localpos, porttype, portname)) {
	    connecting = *it;
	    showPossibleConnections(connecting, portname, porttype);
	    return;
	  }
	}
      }
      for (std::vector<Module*>::iterator it = modules.begin();
	   it != modules.end();
	   it++) {
	if ((QWidget*)(*it) == who) {
	  moving = *it;
	  moving_start = p;
	  return;
	}
      }
    }
}

void NetworkCanvasView::contentsMouseReleaseEvent(QMouseEvent* /*e*/)
{
  //IMPORTANT NOTES: e->pos() returns the mouse point in the canvas coordinates
  //std::cerr<<"NetworkCanvasView::contentsMouseReleaseEvent e->pos()="<<e->pos().x()<<std::endl;

  if (connecting && highlightedConnection != 0) {
      if (highlightedConnection->getConnectionType() == "BridgeConnection") {
        /*Create an automatic bridge*/
        if (porttype == Module::USES) {
      	  addBridgeConnection(connecting, portname, highlightedConnection->getProvidesModule(),
	                    highlightedConnection->getProvidesPortName());  
	} else {
          addBridgeConnection(highlightedConnection->getUsesModule(),
		            highlightedConnection->getUsesPortName(), connecting, portname);
	}
      } else {	        
        if (porttype == Module::USES) {
	  addConnection(connecting, portname, highlightedConnection->getProvidesModule(),
		    highlightedConnection->getProvidesPortName());
	} else {
	  addConnection(highlightedConnection->getUsesModule(),
		      highlightedConnection->getUsesPortName(), connecting, portname); 
	}
      }
  }
  clearPossibleConnections();
  connecting=0;
  highlightedConnection=0;
  moving = 0;
}

void NetworkCanvasView::contentsMouseMoveEvent(QMouseEvent* e)
{
    if (moving) {
	int dx = 0;
	int dy = 0;
	QPoint p = contentsToViewport(e->pos());
	//newX, newY are in canvas coordinates
	int newX = childX(moving) + p.x() - moving_start.x();
	int newY = childY(moving) + p.y() - moving_start.y();
	QPoint pLeftTop = contentsToViewport( QPoint(newX, newY) );

	QPoint mouse = e->globalPos();
	if (pLeftTop.x() < 0) {
	    newX -= pLeftTop.x();
	    if (p.x() < 0) {
		mouse.setX(mouse.x() - p.x());
		p.setX(0);
		QCursor::setPos(mouse);	
	    }
	    dx =- 1;
	}
	if (pLeftTop.y() < 0) {
	    newY -= pLeftTop.y();
	    if (p.y() < 0) {
		mouse.setY(mouse.y() - p.y());
		p.setY(0);
		QCursor::setPos(mouse);	
	    }
	    dy =- 1;
	}
	int cw = contentsRect().width();
	int mw = moving->frameSize().width();
	if (pLeftTop.x() > cw - mw) {
	    newX -= pLeftTop.x() - (cw-mw);
	    if (p.x() > cw) {
		mouse.setX(mouse.x() - (p.x() - cw));
		p.setX(cw - mw);
		QCursor::setPos(mouse);
	    }
	    dx = 1;
	}
	int ch = contentsRect().height();
	int mh = moving->frameSize().height();
	if (pLeftTop.y() > ch - mh) {
	    newY -= pLeftTop.y() - (ch - mh);
	    if (p.y() > ch) {
		mouse.setY(mouse.y() - (p.y() - ch));
		p.setY(ch);
		QCursor::setPos(mouse);	
	    }
	    dy = 1;
	}
	//if (pLeftTop.x()<0 || pLeftTop.y()<0)
	//if (! canvas()->rect().contains(QRect(pLeftTop, moving->frameSize()), true)) return;
	moving_start = p;
	moveChild(moving, newX, newY);

	if (dx || dy) {
	    scrollBy(dx * 5,dy * 5);
	}
	for (std::vector<Connection*>::iterator ct = connections.begin();
		ct != connections.end(); ct++) {
	    if ((*ct)->isConnectedTo(moving)) {
		(*ct)->resetPoints();
	    }
	}
	canvas()->update();
    }
    if (connecting) {
	QCanvasItemList lst = canvas()->collisions(e->pos());
	if (lst.size()>0) {
	    highlightConnection(lst[0]);
	} else if (highlightedConnection!=0) {
	    highlightConnection(0);
	}
    }
    p2BuilderWindow->updateMiniView();
}

// TEK
// updates position of the new module w/in view
void NetworkCanvasView::addChild(Module* mod2add, int x , int y, bool reposition)
{
  std::vector<Module*> add_module = this->getModules();

  int buf = 20;
  QPoint stdOrigin(buf, buf);
  QSize stdSize(120,mod2add->height());
  QSize stdDisp = stdSize+QSize(buf,buf);
  int maxrow= height()/stdDisp.height();
  int maxcol= width()/stdDisp.width();
  
  if (!reposition){
    QPoint p = viewportToContents(QPoint(x,y));
    QScrollView::addChild(mod2add, p.x(), p.y());
    return;
  }
  for (int icol = 0; icol<maxcol; icol++){
    
    for (int irow = 0; irow<maxrow; irow++){

      QRect candidateRect=  QRect(stdOrigin.x()+stdDisp.width()*icol, 
				  stdOrigin.y()+stdDisp.height()*irow,
				  stdSize.width(), stdSize.height());
      
      // check with all the viewable modules - can new module be placed?
      // searching through all points of mod2add for conflicts

      bool intersects = false;

      for (unsigned int i = 0; i < add_module.size(); i++ ){
	
	QRect rect(add_module[i]->x(), add_module[i]->y(), 
		   add_module[i]->width(),add_module[i]->height());
	
	intersects |=candidateRect.intersects(rect);
      }
      if (!intersects){
	QPoint p = viewportToContents(candidateRect.topLeft());
	QScrollView::addChild(mod2add, p.x(), p.y());
	return;
      }
    }
  }
  //std::cerr<<"not candidate rect found!"<<std::endl;
  QPoint p = viewportToContents(QPoint(0,0));
  QScrollView::addChild(mod2add, p.x(), p.y());
}


Module* NetworkCanvasView::addModule(const std::string& name, int x, int y,
				  SSIDL::array1<std::string> &up,
				  SSIDL::array1<std::string> &pp ,
				  const sci::cca::ComponentID::pointer &cid,
				   bool reposition)
{
    Module *module = new Module(this, name, up, pp, services, cid);
    addChild(module, x, y, reposition);

    // std::string testString; -- not used

  connect(module, SIGNAL(destroyModule(Module *)), this, SLOT(removeModule(Module *)));
  modules.push_back(module);
  module->show();		
  // have to updateMiniView() after added to canvas
  p2BuilderWindow->updateMiniView();
  return module;
}

void NetworkCanvasView::removeModule(Module * module)
{
  removeAllConnections(module);
  for (unsigned int i = 0; i<modules.size(); i++){
    if (modules[i]==module){
      modules.erase(modules.begin()+i);
      break;
    }
  }
  module->hide();
  delete module;
  p2BuilderWindow->updateMiniView();
}

void NetworkCanvasView::addConnection(Module *m1,const std::string &portname1,  Module *m2, const std::string &portname2)
{

  sci::cca::ports::BuilderService::pointer bs = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if (bs.isNull()){
    std::cerr << "Fatal Error: Cannot find builder service\n";
    return;
  }
  sci::cca::ConnectionID::pointer connID = bs->connect(m1->cid, portname1, m2->cid, portname2);

  services->releasePort("cca.BuilderService");
  
  Connection *con = new Connection(m1,portname1, m2,portname2, connID,this);

  //std::string instanceName = m1->cid->getInstanceName();

  connections.push_back(con);
  con->show();
  //canvas()->update();
  // have to updateMiniView() after added to canvas
  p2BuilderWindow->updateMiniView();
}

void NetworkCanvasView::addBridgeConnection(Module *m1,const std::string &portname1,  Module *m2, const std::string &portname2)
{
  sci::cca::ports::BuilderService::pointer bs = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if (bs.isNull()){
    std::cerr << "Fatal Error: Cannot find builder service\n";
    return;
  }

  //Instantiate bridge component
  std::string instanceT = bs->generateBridge(m1->cid,portname1,m2->cid,portname2);
  if (instanceT==""){
    std::cerr << "Error: could not properly generate bridge... aborting connection...\n";
    return;
  } 
  std::string classT = "bridge:Bridge."+instanceT;
  Module* bm = p2BuilderWindow->instantiateComponent(instanceT,classT,instanceT);

  //Logically connect to and from bridge	
  SSIDL::array1<std::string> usesPorts = bs->getUsedPortNames(bm->cid);
  SSIDL::array1<std::string> providesPorts = bs->getProvidedPortNames(bm->cid);
  std::cerr<<"connect "<<m1->cid->getInstanceName()<<"->"<<portname1<<" to "<<bm->cid->getInstanceName()<<"->"<<providesPorts[0]<<"\n";
  sci::cca::ConnectionID::pointer connID1 = bs->connect(m1->cid, portname1, bm->cid, providesPorts[0]);
  std::cerr<<"connect "<<bm->cid->getInstanceName()<<"->"<<usesPorts[0]<<" to "<<m2->cid->getInstanceName()<<"->"<<portname2<<"\n";
  sci::cca::ConnectionID::pointer connID2 = bs->connect(bm->cid, usesPorts[0], m2->cid, portname2);

  //Graphically connect to and from bridge
  Connection *con1 = new Connection(m1,portname1, bm,"pport", connID1,this);
  con1->show();
  connections.push_back(con1);
  Connection *con2 = new Connection(bm,"uport", m2,portname2, connID2,this);
  con2->show();
  connections.push_back(con2);
  canvas()->update();

  services->releasePort("cca.BuilderService");
}

void NetworkCanvasView::removeConnection(QCanvasItem *c)
{
	for (std::vector<Connection *>::iterator iter = connections.begin();
		iter!=connections.end(); iter++){
	   if ((QCanvasItem*) (*iter) == c){
                
		//std::cerr<<"connection.size()="<<connections.size()<<std::endl;
		//std::cerr<<"all item.size before del="<<canvas()->allItems().size()<<std::endl;
		sci::cca::ports::BuilderService::pointer bs = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
		if (bs.isNull()){
		  std::cerr << "Fatal Error: Cannot find builder service\n";
		}
		bs->disconnect((*iter)->getConnectionID(),0);
		services->releasePort("cca.BuilderService");
		connections.erase(iter);

		delete c;
                //std::cerr<<"allitem.size after del="<<canvas()->allItems().size()<<std::endl;
	        canvas()->update();
		break;

           }			
	}
    p2BuilderWindow->updateMiniView();
}

void NetworkCanvasView::removeAllConnections(Module *module)
{
  bool needUpdate = false;
  for (int i = connections.size()-1; i>=0; i--){
    if (connections[i]->isConnectedTo(module)){
      sci::cca::ports::BuilderService::pointer bs = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
      if (bs.isNull()){
	std::cerr << "Fatal Error: Cannot find builder service\n";
      }
      bs->disconnect(connections[i]->getConnectionID(),0);
      services->releasePort("cca.BuilderService");
      delete connections[i];
      connections.erase(connections.begin()+i);

      needUpdate = true;
    }
  }
  if (needUpdate) {
    canvas()->update();
  }
  p2BuilderWindow->updateMiniView();
}

void NetworkCanvasView::setServices(const sci::cca::Services::pointer &services)
{
	this->services = services;
}


void NetworkCanvasView::showPossibleConnections(Module *m, const std::string &portname, Module::PortType porttype)
{
  sci::cca::ports::BuilderService::pointer bs = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if (bs.isNull()){
    std::cerr << "Fatal Error: Cannot find builder service\n";
  }

  //std::cerr<<"Possible Ports:"<<std::endl;
  for (unsigned int i = 0; i<modules.size(); i++){
      SSIDL::array1<std::string> portList = bs->getCompatiblePortList(m->cid,portname,modules[i]->cid);
      for (unsigned int j = 0; j<portList.size(); j++){
	Connection *con;
	if (porttype == Module::USES)
	   con = new Connection(m,portname, modules[i],portList[j], 
			      sci::cca::ConnectionID::pointer(0),this);      
	else			       
	   con = new Connection(modules[i],portList[j], m, portname, 
			     sci::cca::ConnectionID::pointer(0),this);
	con->show();
	possibleConns.push_back(con);
	canvas()->update();
	//std::cerr<<portList[j]<<std::endl;
      }
  }    

  //Show Bridgeable Ports
  for (unsigned int i = 0; i<modules.size(); i++){
    SSIDL::array1<std::string> portList = bs->getBridgablePortList(m->cid,portname,modules[i]->cid);
    for (unsigned int j = 0; j<portList.size(); j++){
      Connection *con;
      if (porttype == Module::USES)
	con = new BridgeConnection(m,portname, modules[i],portList[j],
			   sci::cca::ConnectionID::pointer(0),this);
      else
	con = new BridgeConnection(modules[i],portList[j], m, portname,
			   sci::cca::ConnectionID::pointer(0),this);
      con->show();
      possibleConns.push_back(con);
      canvas()->update();
      //std::cerr<<portList[j]<<std::endl;
    }
  }


  services->releasePort("cca.BuilderService");		
}

void NetworkCanvasView::clearPossibleConnections()
{
	for (unsigned int i = 0; i<possibleConns.size(); i++){
	   delete possibleConns[i];
        }
	possibleConns.erase(possibleConns.begin(),possibleConns.end());
	canvas()->update();
}  

void NetworkCanvasView::highlightConnection(QCanvasItem *c)
{
	//std::cerr<<"Highlight"<<std::endl;
  if (highlightedConnection!=0){
    highlightedConnection->setDefault();
    highlightedConnection->hide();
    canvas()->update();
    highlightedConnection->show();
    canvas()->update();
    highlightedConnection = 0;  
  }


	for (unsigned int i = 0; i<possibleConns.size();i++){
	  if ((QCanvasItem*) (possibleConns[i]) == c){
	     possibleConns[i]->highlight();
	     possibleConns[i]->hide();
	     canvas()->update();
	     possibleConns[i]->show();
	     canvas()->update();
	     highlightedConnection = possibleConns[i];
	     break;
	  }			
	}
}

std::vector<Module*> NetworkCanvasView::getModules()
{
  return modules;
}

std::vector<Connection*> NetworkCanvasView::getConnections()
{
  return connections;
}

void NetworkCanvasView::viewportResizeEvent(QResizeEvent* p2QResizeEvent)
{
  QScrollView::viewportResizeEvent(p2QResizeEvent);
  p2BuilderWindow->updateMiniView();  
}
