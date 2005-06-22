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
 *  Module.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 2002 
 *
 */

#include <sci_defs/tao_defs.h>
#include <CCA/Components/Builder/Module.h>
#include <CCA/Components/Builder/NetworkCanvasView.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/CCA/CCAException.h>

#include <qapplication.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qmessagebox.h>
#include <qpainter.h>
#include <qpoint.h>
#include <qevent.h>
#include <qwhatsthis.h>

#include <iostream>
#include <sstream>


void ModuleProgress::updateProgress(int p)
{
    // should check that progress is less than totalSteps()
    if (p > mod->progress->totalSteps()) {
        std::cerr << "ModuleProgress::updateProgress: progress > totalSteps" << std::endl;
    } else {
        mod->progress->setProgress(p);
        // Force a repaint rather than waiting for return
        // to the Qt main event loop: may cause flicker (see QWidget docs).
        mod->repaint();
    }
}

void ModuleProgress::updateProgress(int p, int totalSteps)
{
    mod->progress->setProgress(p, totalSteps);
    mod->repaint();
}

Module::Module(NetworkCanvasView *parent,
               const std::string &mName,
               const sci::cca::Services::pointer &services,
               const sci::cca::ComponentID::pointer &cid)
  : QFrame(parent, mName.c_str()), mName(mName), services(services),
    cid(cid)
{
    pd = PORT_DIST;
    pw = PORT_W;
    ph = PORT_H;
    viewWindow = parent;

    instanceName = cid->getInstanceName();
    menu = new QPopupMenu(this);
    makePorts();

#ifdef HAVE_TAO
    // TODO: find a better way to handle port detection for
    // Corba components
    std::string componentName = mName.substr(0, mName.find('.'));
    if ("Corba" == componentName || "Tao" == componentName) {
        updatePorts();
    }
#endif
}


void
Module::makePorts() {
    int top = 5;
    int w = 120;
    int h = 60;

    SSIDL::array1<std::string> up;
    SSIDL::array1<std::string> pp;
    ports.clear();

    // get pointer to the framework that instantiated the component with ComponentID cid
    // until port properties are implemented properly    
    // TODO: finish port properties
    ComponentID *compID = dynamic_cast<ComponentID*>(cid.getPointer());
    SCIRunFramework* fwk = compID->framework;
    if (!fwk) {
        std::cerr << "Error: could not get the framework!" << std::endl;
        // quit here?
    }

    sci::cca::ports::BuilderService::pointer bs =
        pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    if (bs.isNull()) {
        std::cerr << "Error: Cannot find builder service." << std::endl;
    } else {
        up = bs->getUsedPortNames(cid);
        pp = bs->getProvidedPortNames(cid);

        // KOSTA add to make modules bigger if they have a lot of ports
        int max_ports = (up.size() > pp.size()) ? up.size() : pp.size();
        if (max_ports > 3) {
            h = max_ports * (h / 3);
        }
        if (mName.substr(0, mName.find('.')) == "Vtk") {
            setPaletteBackgroundColor( QColor(255, 0, 0) );
        } else if (mName.substr(0, mName.find('.')) == "Corba") {
            setPaletteBackgroundColor( QColor(0, 255, 0) );
        }
        //Eof KOSTA

        std::string loaderName;
        int nNodes = 1;
        sci::cca::TypeMap::pointer properties =
            bs->getPortProperties(cid, "");
        if (!properties.isNull()) {
            loaderName = properties->getString("LOADER NAME", loaderName);
            nNodes = properties->getInt("np", nNodes);
        }

        std::ostringstream nameWithNodes;
        nameWithNodes << mName;
        if (nNodes > 1) {
            nameWithNodes << "(" << nNodes << ")";
        }
        if ( loaderName.empty() ) {
            menu->insertItem(nameWithNodes.str().c_str());
        } else {
            std::string name = nameWithNodes.str() + "@" + loaderName;
            menu->insertItem(name.c_str());
        }
        menu->insertSeparator();  

        progress = new QProgressBar(100, this);
        progress->reset();
        //progress->hide();

        ModuleProgress *mp = new ModuleProgress();
        mp->setModule(this);
        modProgress = ModuleProgress::pointer(mp);
        progPortName = instanceName + "_moduleProgress";
        sci::cca::TypeMap::pointer props = services->createTypeMap();
        services->addProvidesPort(modProgress, progPortName, "sci.cca.ports.Progress", props);

        QPalette pal = progress->palette();
        QColorGroup cg = pal.active();
        QColor barColor(0, 127, 0);
        cg.setColor(QColorGroup::Highlight, barColor);
        pal.setActive(cg);
        cg = pal.inactive();
        cg.setColor(QColorGroup::Highlight, barColor);
        pal.setInactive(cg);
        cg = pal.disabled();
        cg.setColor(QColorGroup::Highlight, barColor);
        pal.setDisabled(cg);
        progress->setPalette(pal);
        progress->setPercentageVisible(false);
        progress->setGeometry(
            QRect(top + 22, h - top - 20, w - top - 24 - top, 20));

        //statusButton = new QPushButton(this, "status");
        //statusButton->setGeometry(QRect(progress->width() + 24, h - top - 20, 15, 15));
        //if (statusButton->width() + top * 2 > w) {
        //    w = statusButton->width() + top * 2;
        //}

        hasGoPort = hasUIPort = hasComponentIcon = false;
        bool isSciPort = false;
        int connectable_ports = 0;
        for (unsigned int i = 0; i < pp.size(); i++) {
            if (pp[i] == "ui") {
                hasUIPort = true;
            } else if (pp[i] == "sci.ui") {
                hasUIPort = true;
                isSciPort = true;
            } else if (pp[i] == "go") {
                hasGoPort = true;
            } else if (pp[i] == "sci.go") {
                hasGoPort = true;
                isSciPort = true;
            } else if (pp[i] == "icon") {
                hasComponentIcon = true;
            } else {
                std::string model;
                std::string type;
                ComponentInstance *ci =
                    fwk->lookupComponent(instanceName);
                if (ci) {
                    PortInstance* pi = ci->getPortInstance(pp[i]);
                    if (pi) {
                        model = pi->getModel();
                        type = pi->getType();
                    }
                }
                ports.push_back(
                    port(connectable_ports++, model, type, pp[i],
                        PortIcon::PROVIDES));
            }
        }
        if (hasUIPort) {
            uiButton = new QPushButton("UI", this, "ui");
            uiButton->setGeometry(QRect(top, h - top - 20, 20, 20));
            connect(uiButton, SIGNAL(clicked()), this, SLOT(ui()));

            uiPortName = instanceName + " uiPort";
            try {
                services->registerUsesPort(uiPortName, "sci.cca.ports.UIPort", sci::cca::TypeMap::pointer(0));
            }
            catch (const Exception& e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }

            try {
                bs->connect(services->getComponentID(), uiPortName, cid, isSciPort ? "sci.ui" : "ui");
            }
            catch (const Exception& e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }
        } else {
            uiButton = 0;
        }

        if (hasGoPort) {
            menu->insertItem("Go", this, SLOT(go()) );
            goPortName = instanceName + " goPort";
            try {
                services->registerUsesPort(goPortName, "sci.cca.ports.GoPort",
                    sci::cca::TypeMap::pointer(0));
            }
            catch (const Exception& e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }

            try {
                bs->connect(services->getComponentID(), goPortName,
                    cid,  isSciPort ? "sci.go" : "go");
            }
            catch (const Exception& e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }
        }

        if (hasComponentIcon) {
            // progress bar etc.
            //std::string instanceName = cid->getInstanceName();
            iconName = instanceName + " icon";
            try {
                services->registerUsesPort(iconName,
                    "sci.cca.ports.ComponentIcon",
                        sci::cca::TypeMap::pointer(0));
            }
            catch (const Exception& e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }

            try {
                bs->connect(services->getComponentID(), iconName, cid, "icon");
            }
            catch (const Exception& e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }

            sci::cca::Port::pointer p = services->getPort(iconName);
            sci::cca::ports::ComponentIcon::pointer icon =
                pidl_cast<sci::cca::ports::ComponentIcon::pointer>(p);
            if (icon.isNull()) {
                std::cerr << "icon is not connected";
            } else {
                int totalSteps = icon->getProgressBar();
                if (totalSteps != 0) {
                    progress->setTotalSteps(totalSteps);
                }
                dName = icon->getDisplayName();
                mDesc = icon->getDescription();
                menu->insertItem("Description", this, SLOT(desc()) );
            }
        } else {
            dName = nameWithNodes.str();
        }

        nameRect = QRect( QPoint(top, top), (new QLabel(dName.c_str(), 0))->sizeHint() );
        nameRect.addCoords(2, -2, 6, 6); // 0, -2, 4, 4
        if (nameRect.width() + top * 2 > w) {
            w = nameRect.width() + top * 3;
        }
        // QRect uiRect(top, nameRect.bottom() + d, 20, 20);
        setGeometry(QRect(0, 0, w, h));
        setFrameStyle(Panel | Raised);
        setLineWidth(4);

        menu->insertItem("Destroy", this, SLOT( destroy() ));

        int positionCtr = 0;
        for (unsigned int i = 0; i < up.size(); i++) {
            std::string model;
            std::string type;
            ComponentInstance *ci = fwk->lookupComponent(instanceName);
            if (ci) {
                PortInstance* pi = ci->getPortInstance(up[i]);
                if (pi) {
                    model = pi->getModel();
                    type = pi->getType();
                    if (type == "sci.cca.ports.Progress") {
                        try {
                            bs->connect(cid, "progress", services->getComponentID(), progPortName);
                            progress->show();
                        }
                        catch (const Exception& e) {
                            viewWindow->p2BuilderWindow->displayMsg(e.message());
                        }
                        continue;
                    }
                }
            }
            ports.push_back(port(positionCtr++, model, type, up[i], PortIcon::USES));
        }

        services->releasePort("cca.BuilderService");
    }
    // fill this in
    //QWhatsThis::add(this, "Module\nUses Ports:\nProvides Ports:");
}

Module::~Module()
{
    try {
        if (hasUIPort) {
            services->unregisterUsesPort(uiPortName);
        }
    }
    catch (CCAException e) {
        viewWindow->p2BuilderWindow->displayMsg(e.message());
    }
    try {
        if (hasGoPort) {
            services->unregisterUsesPort(goPortName);
        }
    }
    catch (CCAException e) {
        viewWindow->p2BuilderWindow->displayMsg(e.message());
    }
    try {
        if (hasComponentIcon) {
            services->unregisterUsesPort(iconName);
        }
    }
    catch (CCAException e) {
        viewWindow->p2BuilderWindow->displayMsg(e.message());
    }
    try{
        services->removeProvidesPort(progPortName);
    }
    catch (CCAException e) {
        viewWindow->p2BuilderWindow->displayMsg(e.message());
    }
}

void Module::paintEvent(QPaintEvent *e)
{
  QFrame::paintEvent(e);
  QPainter p( this );
  p.setPen(black);
  p.setFont( QFont("Times", 10, QFont::Bold) );
  p.drawText(nameRect, AlignCenter, dName.c_str());

  for (std::vector<PortIcon*>::iterator it = ports.begin();
       it != ports.end(); it++) {
    (*it)->drawPort(p);
  }
  p.flush();
}

PortIcon* Module::getPort(const std::string &name, PortIcon::PortType type)
{
  // ui & go ports?
  for (std::vector<PortIcon*>::iterator it = ports.begin();
       it != ports.end(); it++) {
    if ((*it)->name() == name && (*it)->type() == type) {
      return (*it);
    }
  }
  return 0;
}

QPoint Module::usesPortPoint(int num)
{
  int x = width();
  int y = pd + (ph + pd) * num + ph / 2;
  return QPoint(x, y);
}

QPoint Module::providesPortPoint(int num)
{
  int x = 0;
  int y = pd + (ph + pd) * num + ph / 2;
  return QPoint(x, y);
}

QPoint Module::posInCanvas()
{
  return viewWindow->viewportToContents(pos());
}

void Module::mousePressEvent(QMouseEvent *e)
{
  if (e->button() != RightButton) {
    QFrame::mousePressEvent(e);
  } else {
    PortIcon *port;
    if ( (port = clickedPort( e->pos() )) ) {
      port->menu()->popup(mapToGlobal(e->pos()));
    } else {
      menu->popup(mapToGlobal(e->pos()));
    }
  }
}

PortIcon* Module::clickedPort(QPoint localpos)
{
  const int ex = 2;
  for (std::vector<PortIcon*>::iterator it = ports.begin();
       it != ports.end(); it++) {
    QRect r = (*it)->rect();
    r = QRect(r.x() - ex, r.y() - ex, r.width() + ex * 2, r.height() + ex * 2);
    if (r.contains(localpos)) {
      return *it;
    }
  }
  return 0;
}

PortIcon* Module::port(int portnum, const std::string& model,
                       const std::string& type, const std::string& name,
                       PortIcon::PortType porttype)
{
  if (porttype == PortIcon::PROVIDES) {
    QPoint  r = providesPortPoint(portnum);
    return new PortIcon(this, model, type, name, PortIcon::PROVIDES,
            QRect(r.x(), r.y() - ph / 2, pw, ph),
            portnum, services);
  } else { // uses  
    QPoint r = usesPortPoint(portnum);
    return new PortIcon(this, model, type, name, PortIcon::USES,
            QRect(r.x() - pw, r.y() - ph / 2, pw, ph),
            portnum, services);
  }
}

void Module::go()
{
    progress->reset();

    sci::cca::Port::pointer p = services->getPort(goPortName);
    sci::cca::ports::GoPort::pointer goPort =
        pidl_cast<sci::cca::ports::GoPort::pointer>(p);

    if (goPort.isNull()) {
        viewWindow->p2BuilderWindow->displayMsg("goPort is not connected, cannot bring up Go!");
    } else {
        // exception handling?

        int status = goPort->go();
        if (status == 0) {
            if (progress->progress() < progress->totalSteps()) {
                progress->setProgress(progress->totalSteps());
            }
        }
        services->releasePort(goPortName);
    }
}

void Module::stop()
{
  viewWindow->p2BuilderWindow->displayMsg("stop() not implemented");
}

void Module::destroy()
{
  emit destroyModule(this);
}

void Module::ui()
{
  sci::cca::Port::pointer p = services->getPort(uiPortName);
  sci::cca::ports::UIPort::pointer uiPort =
    pidl_cast<sci::cca::ports::UIPort::pointer>(p);
  if (uiPort.isNull()) {
    viewWindow->p2BuilderWindow->displayMsg("uiPort is not connected, cannot bring up UI!");
  } else {
    int status = uiPort->ui();
    std::cerr << "UI status=" << status << std::endl;
    services->releasePort(uiPortName);
  }
}

void Module::desc()
{
  QMessageBox::information(this, dName.c_str(), mDesc.c_str());
}

void
Module::updatePorts() {
    int top = 5;
    //int w = 120;
    int h = 60;

    SSIDL::array1<std::string> up;
    SSIDL::array1<std::string> pp;
    ports.clear();

    // get pointer to the framework that instantiated the component with ComponentID cid
    // until port properties are implemented properly    
    ComponentID *compID = dynamic_cast<ComponentID*>(cid.getPointer());
    SCIRunFramework* fwk = compID->framework;
    if (!fwk) {
        std::cerr << "Error: could not get the framework!" << std::endl;
    }

    sci::cca::ports::BuilderService::pointer bs =
    pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    if (bs.isNull()) {
        std::cerr << "Error: Cannot find builder service." << std::endl;
    } else {
        up = bs->getUsedPortNames(cid);
        pp = bs->getProvidedPortNames(cid);

        std::string loaderName;
        int nNodes = 1;
        sci::cca::TypeMap::pointer properties =
            bs->getPortProperties(cid, "");
        if (!properties.isNull()) {
            loaderName = properties->getString("LOADER NAME", loaderName);
            nNodes = properties->getInt("np", nNodes);
        }

        std::ostringstream nameWithNodes;
        nameWithNodes << mName;
        if (nNodes > 1) {
            nameWithNodes << "(" << nNodes << ")";
        }

        /*    if ( loaderName.empty() ) {
                menu->insertItem(nameWithNodes.str().c_str());
            } else {
                std::string name = nameWithNodes.str() + "@" + loaderName;
                menu->insertItem(name.c_str());
            }
            menu->insertSeparator();  
        */

        hasGoPort = hasUIPort = hasComponentIcon = false;
        bool isSciPort = false;
        int connectable_ports = 0;
        for (unsigned int i = 0; i < pp.size(); i++) {
            if (pp[i] == "ui") {
                hasUIPort = true;
            } else if (pp[i] == "sci.ui") {
                hasUIPort = true;
                isSciPort = true;
            } else if (pp[i] == "go") {
                hasGoPort = true;
            } else if (pp[i] == "sci.go") {
                hasGoPort = true;
                isSciPort = true;
            } else if (pp[i] == "icon") {
                hasComponentIcon = true;
            } else {
                std::string model;
                std::string type;

                ComponentInstance *ci =
                    fwk->lookupComponent(instanceName);
                if (ci) {
                    PortInstance* pi = ci->getPortInstance(pp[i]);
                    if (pi) {
                        model = pi->getModel();
                        type = pi->getType();
                    }
                }
                ports.push_back(
                    port(connectable_ports++, model, type, pp[i],
                        PortIcon::PROVIDES));
            }
        }

        if (hasUIPort) {
            uiButton = new QPushButton("UI", this, "ui");
            uiButton->setGeometry(QRect(top, h - top - 20, 20, 20));
            connect(uiButton, SIGNAL(clicked()), this, SLOT(ui()));

            std::string instanceName = cid->getInstanceName();
            uiPortName = instanceName + " uiPort";
            try {
                services->registerUsesPort(uiPortName, "sci.cca.ports.UIPort",
                    sci::cca::TypeMap::pointer(0));
            }
            catch (CCAException e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }  

            try {
                bs->connect(services->getComponentID(), uiPortName,
                    cid, isSciPort ? "sci.ui" : "ui");
            }
            catch (CCAException e) {
                viewWindow->p2BuilderWindow->displayMsg(e.message());
            }  
        }
        int positionCtr = 0;
        for (unsigned int i = 0; i < up.size(); i++) {
            std::string model;
            std::string type;
            ComponentInstance *ci = fwk->lookupComponent(cid->getInstanceName());
            if (ci) {
                PortInstance* pi = ci->getPortInstance(up[i]);
                if (pi) {
                    model = pi->getModel();
                    type = pi->getType();
                    if (type == "sci.cca.ports.Progress") {
                         continue;
                    }
                }
            }
            ports.push_back(port(positionCtr++, model, type, up[i], PortIcon::USES));
        }
        services->releasePort("cca.BuilderService");
    }
}
  
