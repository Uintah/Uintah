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
gov::cca::ports::UIPort::pointer uip, CIA::array1<std::string> & up, CIA::array1<std::string> &pp )
  :QFrame(parent, name )
{
	uiPort=uip;
        this->up=up;
	this->pp=pp;
	int dx=5;
/*	int dy=10;
	int d=5;
*/	int w=120;
	int h=60;

	nameRect=QRect(QPoint(0,0), (new QLabel(name,0))->sizeHint() );
	if(nameRect.width()+dx*2>w) w=nameRect.width()+dx*2;
//	QRect uiRect(dx,nameRect.bottom()+d,20,20);

	
	setGeometry(QRect(0,0,w,h));
  setFrameStyle(WinPanel|Raised);

	
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
    for(int i=0;i<up.size();i++){
	p.drawRect(10+i*20,0,10,4);
    }

    p.setPen(red);
    p.setBrush(red);
    for(int i=0;i<pp.size();i++){
        p.drawRect(10+i*20,height()-4,10,4);
    }		
}

QPoint Module::usePortPoint()
{
	return QPoint(10,height());
}

QPoint Module::providePortPoint()
{
	return QPoint(10,0);	
}

void Module::mousePressEvent(QMouseEvent *e)
{
	if(e->button()!=RightButton) QFrame::mousePressEvent(e);
	else{
		menu->popup(mapToGlobal(e->pos()));
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
