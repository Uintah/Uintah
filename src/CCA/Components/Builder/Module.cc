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
#include <iostream.h>
#include "Module.h"

#include <qpushbutton.h>
#include <qlabel.h>
#include <qpainter.h>
#include <qmessagebox.h>
Module::Module(QWidget *parent, const char *name,
gov::cca::ports::UIPort::pointer uip, CIA::array1<std::string> & up, CIA::array1<std::string> &pp , const gov::cca::ComponentID::pointer &cid)
  :QFrame(parent, name )
{
        pd=10; //distance between two ports
        pw=10; //port width
        ph=4; //prot height

	this->cid=cid;
	uiPort=uip;
	for(unsigned int i=0; i<pp.size(); i++){
		if(pp[i]!="ui") this->pp.push_back(pp[i]);
	}
		
        this->up=up;
	//this->pp=pp;
	int dx=5;
/*	int dy=10;
	int d=5;
*/	int w=120;
	int h=60;

	nameRect=QRect(QPoint(0,0), (new QLabel(name,0))->sizeHint() );
	if(nameRect.width()+dx*2>w) w=nameRect.width()+dx*2;
//	QRect uiRect(dx,nameRect.bottom()+d,20,20);

	
	setGeometry(QRect(0,0,w,h));
  setFrameStyle(Panel|Raised);
	setLineWidth(4);

	
  QPushButton *ui=new QPushButton("UI", this,"ui");
	//	ui->setDefault(false);
	ui->setGeometry(QRect(dx,h-dx-20,20,20));
  connect(ui,SIGNAL(clicked()), this, SLOT(ui()));
	menu=new QPopupMenu(this);
	menu->insertItem("Execute",this, SLOT(execute()) );
	menu->insertSeparator();	
	menu->insertItem("Stop",this,  SLOT(stop()) );
}


void Module::paintEvent(QPaintEvent *e)
{
	QFrame::paintEvent(e);
    QPainter p( this );
    p.setPen( black );
    p.setFont( QFont( "Times", 10, QFont::Bold ) );
    p.drawText(nameRect, AlignCenter, name() );
   
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
	return QPoint(x,0);
}

QPoint Module::providePortPoint(int num)
{
	int x=pd+(pw+pd)*num+pw/2;
	return QPoint(x,height());	
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
		return QRect(r.x()-pw/2,r.y()-ph,pw,ph);
	}
	else{
		QPoint r=usePortPoint(portnum);
		return QRect(r.x()-pw/2,r.y(),pw,ph);
	}
}

void Module::execute()
{
	cerr<<"execute() "<<endl;	
}

void Module::stop()
{
	cerr<<"stop()"<<endl;	
}

void Module::ui()
{
	uiPort->ui();
	//QMessageBox::warning(this, "UI", "User iterface is not implemented!");
}






