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
#include <SCIRun/CCA/CCAException.h>

#include <qwmatrix.h>
#include <qscrollview.h>
#include <qevent.h>
#include <qfont.h>
#include <qstring.h>
#include <qpopupmenu.h>

#include <iostream>

NetworkCanvasView::NetworkCanvasView(BuilderWindow* p2BuilderWindow,
                                     QCanvas* canvas, QWidget* parent,
                                     const sci::cca::Services::pointer &services)
                     
  : QCanvasView(canvas, parent), p2BuilderWindow(p2BuilderWindow), services(services)
{
    moving = connecting = 0;
    highlightedConnection = 0;
    displayFont = new QFont(p2BuilderWindow->font().family(), 8);

    connect(horizontalScrollBar(), SIGNAL( sliderMoved(int) ),
        p2BuilderWindow, SLOT( updateMiniView() ));
    connect(verticalScrollBar(), SIGNAL( sliderMoved(int) ),
        p2BuilderWindow, SLOT( updateMiniView() ));

    sci::cca::ports::ConnectionEventService::pointer ces =
        pidl_cast<sci::cca::ports::ConnectionEventService::pointer>(
            services->getPort("cca.ConnectionEventService")
        );
    if (ces.isNull()) {
        p2BuilderWindow->displayMsg("Error: Cannot find connection event service.");
    } else {
        sci::cca::ports::ConnectionEventListener::pointer listener(this);
        ces->addConnectionEventListener(sci::cca::ports::ALL, listener);
        services->releasePort("cca.ConnectionEventService");
    }
}

NetworkCanvasView::~NetworkCanvasView()
{
}

void NetworkCanvasView::contentsContextMenuEvent(QContextMenuEvent *e)
{
    QCanvasItemList lst = canvas()->collisions(e->pos());
    if (lst.size() > 0) {
        // check for Connection RTTI 
        removeConnection(lst[0]);
        e->accept();
    } else {
        p2BuilderWindow->componentContextMenu()->exec(QCursor::pos());
    }
}

void NetworkCanvasView::contentsMousePressEvent(QMouseEvent* e)
{
    if (moving || connecting) {
        return;
    }
    // IMPORTANT NOTES: e->pos() returns the mouse point in the canvas coordinates  
    QPoint p = contentsToViewport(e->pos());
    QWidget *who = childAt(p);
    
    // right mouse button events are being handled by context menu events 
    // middle mouse button is ignored for now
    if (e->button() == Qt::LeftButton) {
        //std::cerr << "Qt::LeftButton: pos=" << e->pos().x() << " " << e->pos().y() << std::endl;
        for (std::vector<Module*>::iterator it = modules.begin();
                it != modules.end(); it++) {
            if ((QWidget*)(*it) == who) {
                QPoint localpos =
                    e->pos() - QPoint(childX(who), childY(who));
                //std::cerr << "local point=" << localpos.x() << " " << localpos.y() << std::endl;    
                PortIcon *port;
                if ((port = (*it)->clickedPort(localpos))) {
                    portname = port->name();
                    porttype = port->type();
                    QCanvasText *t =
                        new QCanvasText(QString(port->typeName().c_str()), *displayFont, canvas());
                    t->setX(e->pos().x() + 4);
                    t->setY(e->pos().y() + 4);
                    t->setColor(Qt::white);
                    t->show();
                    connecting = *it;
                    showPossibleConnections(connecting, portname, porttype);
                    return;
                }
            }
        }
        for (std::vector<Module*>::iterator it = modules.begin();
                it != modules.end(); it++) {
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
            if (porttype == PortIcon::USES) {
                addBridgeConnection(connecting, portname,
                    highlightedConnection->providesPort()->module(),
                    highlightedConnection->providesPort()->name());  
            } else {
                addBridgeConnection(highlightedConnection->usesPort()->module(),
                    highlightedConnection->usesPort()->name(),
                    connecting, portname);
            }
        } else {          
            if (porttype == PortIcon::USES) {
                addConnection(connecting, portname,
                    highlightedConnection->providesPort()->module(),
                    highlightedConnection->providesPort()->name());
            } else { // provides
                addConnection(highlightedConnection->usesPort()->module(),
                    highlightedConnection->usesPort()->name(),
                    connecting, portname); 
            }
        }
    }
    QCanvasItemList l = canvas()->allItems();
    for (QCanvasItemList::Iterator it = l.begin(); it != l.end(); it++) {
        if ((*it)->rtti() == QCanvasItem::Rtti_Text) {
            QCanvasText *t = (QCanvasText *)(*it);
            t->hide();
            delete t;
        }
    }
    clearPossibleConnections();
    connecting = 0;
    highlightedConnection = 0;
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
            scrollBy(dx * 5, dy * 5);
        }
        for (std::vector<Connection*>::iterator ct = connections.begin();
            ct != connections.end(); ct++) {
            if ((*ct)->isConnectedTo(moving)) {
                (*ct)->resetPoints();
            }
        }
        canvas()->update();

        sci::cca::ports::ComponentEventService::pointer ces =
            pidl_cast<sci::cca::ports::ComponentEventService::pointer>(
                services->getPort("cca.ComponentEventService")
            );
        if (ces.isNull()) {
            p2BuilderWindow->displayMsg("Error: Cannot find component event service.");
        } else {
            ces->moveComponent(moving->componentID(), newX, newY);
            services->releasePort("cca.ComponentEventService");
        }
    }

    if (connecting) {
        QCanvasItemList lst = canvas()->collisions(e->pos());
        if (lst.size() > 0) {
            highlightConnection(lst[0]);
        } else if (highlightedConnection != 0) {
            highlightConnection(0);
        }
    }
    p2BuilderWindow->updateMiniView();
}

// TEK
// updates position of the new module w/in view
void NetworkCanvasView::addChild(Module* mod2add, int x , int y, bool reposition)
{
  std::vector<Module*> add_module = getModules();

  int buf = 20;
  QPoint stdOrigin(buf, buf);
  QSize stdSize(120, mod2add->height());
  QSize stdDisp = stdSize+QSize(buf, buf);
  int maxrow = height() / stdDisp.height();
  int maxcol = width() / stdDisp.width();
  
  if (!reposition){
    QPoint p = viewportToContents(QPoint(x, y));
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
           add_module[i]->width(), add_module[i]->height());
    
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
  QPoint p = viewportToContents(QPoint(0, 0));
  QScrollView::addChild(mod2add, p.x(), p.y());
}


Module* NetworkCanvasView::addModule(const std::string& name, int x, int y,
                  const sci::cca::ComponentID::pointer &cid,
                   bool reposition)
{
    Module *module = new Module(this, name, services, cid);
    addChild(module, x, y, reposition);

  connect(module, SIGNAL(destroyModule(Module *)), this, SLOT(removeModule(Module *)));
  modules.push_back(module);
  module->show();       
  // have to updateMiniView() after added to canvas
  p2BuilderWindow->updateMiniView();
  return module;
}

void NetworkCanvasView::removeModule(Module *module)
{
    removeAllConnections(module);
    for (unsigned int i = 0; i<modules.size(); i++){
        if (modules[i]==module){
            modules.erase(modules.begin()+i);
            break;
        }
    }
    module->hide();

    sci::cca::ports::BuilderService::pointer bs =
        pidl_cast<sci::cca::ports::BuilderService::pointer>(
            services->getPort("cca.BuilderService")
        );
    if (bs.isNull()) {
        p2BuilderWindow->displayMsg("Error: cannot find builder service.");
        return;
    }
    try {
        sci::cca::ComponentID::pointer cid = bs->getComponentID(module->name());
        if (cid == module->componentID()) {
            bs->destroyInstance(module->componentID(), 0);
        }
    }
    catch (const Exception& e) {
        p2BuilderWindow->displayMsg(e.message());
    }
    services->releasePort("cca.BuilderService");

    delete module;
    p2BuilderWindow->updateMiniView();
}

void NetworkCanvasView::addConnection(Module *m1, const std::string &portname1,  Module *m2, const std::string &portname2)
{
    try {
        sci::cca::ports::BuilderService::pointer bs =
            pidl_cast<sci::cca::ports::BuilderService::pointer>(
                services->getPort("cca.BuilderService")
            );
        if (bs.isNull()){
            p2BuilderWindow->displayMsg("Error: cannot find builder service.");
            return;
        }
            sci::cca::ConnectionID::pointer connID =
                bs->connect(m1->componentID(), portname1, m2->componentID(), portname2);
        services->releasePort("cca.BuilderService");
  
        PortIcon *pUses = m1->getPort(portname1, PortIcon::USES);
        if (pUses == 0) {
            std::cerr << "Error: could not locate port " << portname1 << std::endl;
            return;
        }
        PortIcon *pProvides = m2->getPort(portname2, PortIcon::PROVIDES);
        if (pProvides == 0) {
            std::cerr << "Error: could not locate port " << "pport" << std::endl;
            return;
        }
        Connection *con = new Connection(pUses, pProvides, connID, this);

        connections.push_back(con);
        con->show();
        p2BuilderWindow->updateMiniView();
    }
    catch (CCAException e) {
        p2BuilderWindow->displayMsg(e.message());
    }
}

void NetworkCanvasView::addBridgeConnection(Module *m1, const std::string &portname1,  Module *m2, const std::string &portname2)
{
    try {
        sci::cca::ports::BuilderService::pointer bs =
            pidl_cast<sci::cca::ports::BuilderService::pointer>(
                services->getPort("cca.BuilderService")
            );
        if (bs.isNull()) {
            p2BuilderWindow->displayMsg("Error: Cannot find builder service.");
            return;
        }

        // Instantiate bridge component
        std::string instanceT =
            bs->generateBridge(m1->componentID(), portname1, m2->componentID(), portname2);
        if (instanceT == "") {
            std::cerr << "Error: could not properly generate bridge... aborting connection..." << std::endl;
            return;
        } 
        std::string classT = "bridge:Bridge." + instanceT;
        Module* bm = p2BuilderWindow->instantiateBridgeComponent(instanceT, classT, instanceT);

    // Logically connect to and from bridge
        SSIDL::array1<std::string> usesPorts = bs->getUsedPortNames(bm->componentID());
        SSIDL::array1<std::string> providesPorts = bs->getProvidedPortNames(bm->componentID());
        std::cerr << "connect " << m1->componentID()->getInstanceName() << "->"
                  << portname1 << " to " << bm->componentID()->getInstanceName() << "->"
                  << providesPorts[0] << "\n";
        sci::cca::ConnectionID::pointer connID1 =
            bs->connect(m1->componentID(), portname1, bm->componentID(), providesPorts[0]);
        std::cerr << "connect " << bm->componentID()->getInstanceName() << "->"
                  << usesPorts[0] << " to " << m2->componentID()->getInstanceName() << "->"
                  << portname2 << "\n";
        sci::cca::ConnectionID::pointer connID2 =
            bs->connect(bm->componentID(), usesPorts[0], m2->componentID(), portname2);

        services->releasePort("cca.BuilderService");

  //Graphically connect to and from bridge
  //Connection *con1 = new Connection(m1, portname1, bm, "pport", connID1, this);
        PortIcon *pUses1 = m1->getPort(portname1, PortIcon::USES);
        if (pUses1 == 0) {
            std::cerr << "Error: could not locate port " << portname1 << std::endl;
            return;
        }
        PortIcon *pProvides1 = bm->getPort("pport", PortIcon::PROVIDES);
        if (pProvides1 == 0) {
            std::cerr << "Error: could not locate port " << "pport" << std::endl;
            return;
        }
        Connection *con1 = new Connection(pUses1, pProvides1, connID1, this);
        con1->show();
        connections.push_back(con1);

//Connection *con2 = new Connection(bm, "uport", m2, portname2, connID2, this);
        PortIcon *pUses2 = bm->getPort("uport", PortIcon::USES);
        if (pUses2 == 0) {
            std::cerr << "Error: could not locate port " << "uport" << std::endl;
            return;
        }
        PortIcon *pProvides2 = m2->getPort(portname2, PortIcon::PROVIDES);
        if (pProvides2 == 0) {
            std::cerr << "Error: could not locate port " << portname2 << std::endl;
            return;
        }
        Connection *con2 = new Connection(pUses2, pProvides2, connID2, this);
        con2->show();
        connections.push_back(con2);
        canvas()->update();
    }
    catch (CCAException e) {
        p2BuilderWindow->displayMsg(e.message());
    }
}

void NetworkCanvasView::removeConnection(QCanvasItem *c)
{
    for (std::vector<Connection *>::iterator iter = connections.begin();
            iter!=connections.end(); iter++){
        if ((QCanvasItem*) (*iter) == c) {
            //std::cerr<<"connection.size()="<<connections.size()<<std::endl;
            //std::cerr<<"all item.size before del="<<canvas()->allItems().size()<<std::endl;
            sci::cca::ports::BuilderService::pointer bs =
                pidl_cast<sci::cca::ports::BuilderService::pointer>(
                    services->getPort("cca.BuilderService")
                );
            if (bs.isNull()) {
                p2BuilderWindow->displayMsg("Error: Cannot find builder service.");
                return;
            }
            bs->disconnect((*iter)->getConnectionID(), 0);
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
    for (int i = connections.size() - 1; i >= 0; i--) {
        if (connections[i]->isConnectedTo(module)) {
            sci::cca::ports::BuilderService::pointer bs =
                pidl_cast<sci::cca::ports::BuilderService::pointer>(
                    services->getPort("cca.BuilderService")
                );
            if (bs.isNull()) {
                p2BuilderWindow->displayMsg("Error: Cannot find builder service.");
                return;
            }
            bs->disconnect(connections[i]->getConnectionID(), 0);
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

void
NetworkCanvasView::showPossibleConnections(Module *m,
                                           const std::string &portname,
                                           PortIcon::PortType porttype)
{
    sci::cca::ports::BuilderService::pointer bs =
        pidl_cast<sci::cca::ports::BuilderService::pointer>(
            services->getPort("cca.BuilderService")
        );
    if (bs.isNull()) {
        p2BuilderWindow->displayMsg("Error: cannot find builder service.");
        return;
    }

    //std::cerr<<"Possible Ports:"<<std::endl;
    for (unsigned int i = 0; i < modules.size(); i++) {
        SSIDL::array1<std::string> portList =
            bs->getCompatiblePortList(m->componentID(), portname, modules[i]->componentID());
        for (unsigned int j = 0; j < portList.size(); j++) {
            Connection *con;
            if (porttype == PortIcon::USES) {
              //con = new Connection(m, portname, modules[i], portList[j], sci::cca::ConnectionID::pointer(0), this);      
                PortIcon *pUses = m->getPort(portname, PortIcon::USES);
                if (pUses == 0) {
                    std::cerr << "Error: could not locate port " << portname << std::endl;
                    continue;
                }
                PortIcon *pProvides = modules[i]->getPort(portList[j], PortIcon::PROVIDES);
                if (pProvides == 0) {
                    std::cerr << "Error: could not locate port " << portList[j] << std::endl;
                    continue;
                }
                con = new Connection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), this);      
            } else {
              //con = new Connection(modules[i], portList[j], m, portname, sci::cca::ConnectionID::pointer(0), this);
                PortIcon *pUses = modules[i]->getPort(portList[j], PortIcon::USES);
                if (pUses == 0) {
                    std::cerr << "Error: could not locate port " << portList[j] << std::endl;
                    return;
                }
                PortIcon *pProvides = m->getPort(portname, PortIcon::PROVIDES);
                if (pProvides == 0) {
                    std::cerr << "Error: could not locate port " << portname << std::endl;
                    return;
                }
                con = new Connection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), this);
            }
            con->show();
            possibleConns.push_back(con);
            canvas()->update();
            //std::cerr<<portList[j]<<std::endl;
        }
    }

    // Show Bridgeable Ports
    for (unsigned int i = 0; i < modules.size(); i++) {
        SSIDL::array1<std::string> portList =
            bs->getBridgablePortList(m->componentID(), portname, modules[i]->componentID());
        for (unsigned int j = 0; j < portList.size(); j++) {
            Connection *con;
            if (porttype == PortIcon::USES) {
              //con = new BridgeConnection(m, portname, modules[i], portList[j], sci::cca::ConnectionID::pointer(0), this);
                PortIcon *pUses = m->getPort(portname, PortIcon::USES);
                if (pUses == 0) {
                    std::cerr << "Error: could not locate port " << portname << std::endl;
                    return;
                }
                PortIcon *pProvides = modules[i]->getPort(portList[j], PortIcon::PROVIDES);
                if (pProvides == 0) {
                    std::cerr << "Error: could not locate port " << portList[j] << std::endl;
                    return;
                }
                con = new BridgeConnection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), this);
            } else {
                //con = new BridgeConnection(modules[i], portList[j], m, portname, sci::cca::ConnectionID::pointer(0), this);
                PortIcon *pUses = modules[i]->getPort(portList[j], PortIcon::USES);
                if (pUses == 0) {
                    std::cerr << "Error: could not locate port " << portList[j] << std::endl;
                    return;
                }
                PortIcon *pProvides = m->getPort(portname, PortIcon::PROVIDES);
                if (pProvides == 0) {
                    std::cerr << "Error: could not locate port " << portname << std::endl;
                    return;
                }
                con = new BridgeConnection(pUses, pProvides, sci::cca::ConnectionID::pointer(0), this);
            }
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
    for (unsigned int i = 0; i < possibleConns.size(); i++) {
        delete possibleConns[i];
    }
    possibleConns.erase(possibleConns.begin(), possibleConns.end());
    canvas()->update();
}  

void NetworkCanvasView::highlightConnection(QCanvasItem *c)
{
    // std::cerr<<"Highlight"<<std::endl;
    if (highlightedConnection != 0) {
        highlightedConnection->setDefault();
        highlightedConnection->hide();
        canvas()->update();
        highlightedConnection->show();
        canvas()->update();
        highlightedConnection = 0;  
    }
    for (unsigned int i = 0; i < possibleConns.size(); i++) {
        if ((QCanvasItem*) (possibleConns[i]) == c) {
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


void NetworkCanvasView::connectionActivity(const sci::cca::ports::ConnectionEvent::pointer &e)
{
    std::cerr << "NetworkCanvasView::connectionActivity: got connection event " << e->getEventType() << std::endl;
}

