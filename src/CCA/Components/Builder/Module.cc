/***************************************************************************
                          Module.cpp  -  description
                             -------------------
    begin                : Mon Mar 18 2002
    copyright            : (C) 2002 by kzhang
    email                : kzhang@rat.sci.utah.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#include "Module.h"

#include <qpushbutton.h>
#include <qlabel.h>
#include <qpainter.h>
#include <qmessagebox.h>
#include <iostream>
using namespace std;

Module::Module(QWidget *parent, const string& moduleName,
	       CIA::array1<std::string> & up, CIA::array1<std::string> &pp,
	       const gov::cca::Services::pointer& services,
	       const gov::cca::ComponentID::pointer& cid)
  :QFrame(parent, moduleName.c_str() ), moduleName(moduleName), up(up), services(services), cid(cid)
{
  pd=10; //distance between two ports
  pw=10; //port width
  ph=4; //prot height
		
  int dx=5;
  // int dy=10;
  //  int d=5;

  int w=120;
  int h=60;

  nameRect=QRect(QPoint(0,0), (new QLabel(moduleName.c_str(),0))->sizeHint() );
  if(nameRect.width()+dx*2>w) w=nameRect.width()+dx*2;
//	QRect uiRect(dx,nameRect.bottom()+d,20,20);

	
  setGeometry(QRect(0,0,w,h));
  setFrameStyle(Panel|Raised);
  setLineWidth(4);

  hasGoPort=hasUIPort=false;
  gov::cca::ports::BuilderService::pointer builder = pidl_cast<gov::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
  } 
  else {
    CIA::array1<string> ports = builder->getProvidedPortNames(cid);
    for(unsigned int i=0; i < ports.size(); i++){
      if(ports[i]=="ui") hasUIPort=true;
      else if(ports[i]=="go") hasGoPort=true;
      else this->pp.push_back(ports[i]); 
    }
  }

  if(hasUIPort){
      QPushButton *ui=new QPushButton("UI", this,"ui");
      //	ui->setDefault(false);
      ui->setGeometry(QRect(dx,h-dx-20,20,20));
      connect(ui,SIGNAL(clicked()), this, SLOT(ui()));

      string instanceName = cid->getInstanceName();
      string uiPortName = instanceName+" uiPort";
      services->registerUsesPort(uiPortName, "gov.cca.UIPort",
				 gov::cca::TypeMap::pointer(0));
      builder->connect(services->getComponentID(), uiPortName, cid, "ui");
  }

  menu=new QPopupMenu(this);

  if(hasGoPort){
      menu->insertItem("Go",this, SLOT(go()) );
      menu->insertItem("Stop",this,  SLOT(stop()) );
      menu->insertSeparator();	

      string instanceName = cid->getInstanceName();
      string goPortName = instanceName+" goPort";
      services->registerUsesPort(goPortName, "gov.cca.GoPort",
				 gov::cca::TypeMap::pointer(0));
      builder->connect(services->getComponentID(), goPortName, cid, "go");
  }

  menu->insertItem("Destroy",this,  SLOT(destroy()) );
  services->releasePort("cca.BuilderService");
}


void Module::paintEvent(QPaintEvent *e)
{
  QFrame::paintEvent(e);
  QPainter p( this );
  p.setPen( black );
  p.setFont( QFont( "Times", 10, QFont::Bold ) );
  p.drawText(nameRect, AlignCenter, moduleName.c_str() );
   
  p.setPen(green);
  p.setBrush(green);    
  for(unsigned int i=0;i<up.size();i++){
    p.drawRect(portRect(i, USES));
  }

  p.setPen(red);
  p.setBrush(red);
  for(unsigned int i=0;i<pp.size();i++){
    p.drawRect(portRect(i,PROVIDES));
  }
}

QPoint Module::usePortPoint(int num)
{
	int x=pd+(pw+pd)*num+pw/2;
	return QPoint(x,height());
}

QPoint Module::providePortPoint(int num)
{
	int x=pd+(pw+pd)*num+pw/2;
	return QPoint(x,0);	
}

QPoint Module::usePortPoint(const std::string &portname)
{
  for(unsigned int i=0; i<up.size();i++){
    if(up[i]==portname)
	return usePortPoint(i);
  }
  return QPoint(0,0);
}

QPoint Module::providePortPoint(const std::string &portname)
{
  for(unsigned int i=0; i<pp.size();i++){
    if(pp[i]==portname)
	return providePortPoint(i);
  }
  return QPoint(0,0);
}

std::string Module::usesPortName(int num)
{
	return up[num];
}

std::string Module::providesPortName(int num)
{
	return pp[num];
}

void Module::mousePressEvent(QMouseEvent *e)
{
	if(e->button()!=RightButton) QFrame::mousePressEvent(e);
	else{
		menu->popup(mapToGlobal(e->pos()));
	}
}

bool Module::clickedPort(QPoint localpos, PortType &porttype,
			 std::string &portname)
{
    const int ex=2;		
    for(unsigned int i=0;i<pp.size();i++){
        QRect r=portRect(i, PROVIDES);
	r=QRect(r.x()-ex, r.y()-ex,r.width()+ex*2,r.height()+ex*2);	
	if(r.contains(localpos)){
		porttype=PROVIDES ;
		portname=pp[i];
		return true;
	}
    }	
    for(unsigned int i=0;i<up.size();i++){
        QRect r=portRect(i, USES);
	r=QRect(r.x()-ex, r.y()-ex,r.width()+ex*2,r.height()+ex*2);	
        if(r.contains(localpos)){ 
                porttype=USES ;
                portname=up[i];
                return true;
        }
    }
    return false;
}

QRect Module::portRect(int portnum, PortType porttype)
{
	if(porttype==PROVIDES){ //provides	
		QPoint	r=providePortPoint(portnum);
		return QRect(r.x()-pw/2,r.y(),pw,ph);
	}
	else{
		QPoint r=usePortPoint(portnum);
		return QRect(r.x()-pw/2,r.y()-ph,pw,ph);
	}
}

void Module::go()
{
  string instanceName = cid->getInstanceName();
  string goPortName = instanceName+" goPort";
  gov::cca::Port::pointer p = services->getPort(goPortName);
  gov::cca::ports::GoPort::pointer goPort = pidl_cast<gov::cca::ports::GoPort::pointer>(p);
  if(goPort.isNull()){
    cerr << "goPort is not connected, cannot bring up Go!\n";
  } 
  else {
    goPort->go();
    services->releasePort(goPortName);
  }
}

void Module::stop()
{
	cerr<<"stop() not implemented"<<endl;	
}

void Module::destroy()
{
	cerr<<"destroy() not implemented"<<endl;	
}

void Module::ui()
{
  string instanceName = cid->getInstanceName();
  string uiPortName = instanceName+" uiPort";
  gov::cca::Port::pointer p = services->getPort(uiPortName);
  gov::cca::ports::UIPort::pointer uiPort = pidl_cast<gov::cca::ports::UIPort::pointer>(p);
  if(uiPort.isNull()){
    cerr << "uiPort is not connected, cannot bring up UI!\n";
  } else {
    uiPort->ui();
    services->releasePort(uiPortName);
  }
}
