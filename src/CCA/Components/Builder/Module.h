/***************************************************************************
                          Module.h  -  description
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
#ifndef MODULE_H
#define MODULE_H

#include <qframe.h>
#include <qpoint.h>
#include <qrect.h>
#include <qpopmenu.h>
#include "Core/CCA/spec/cca_sidl.h"
#include "Core/CCA/Component/PIDL/PIDL.h"
class Module:public QFrame
{
	Q_OBJECT

public:
  enum PortType{USES, PROVIDES}; 
  Module(QWidget *parent, const char *name, gov::cca::ports::UIPort::pointer uip, CIA::array1<std::string> & up, CIA::array1<std::string> &pp, const gov::cca::ComponentID::pointer &cid);
	QPoint usePortPoint(int num);
	QPoint usePortPoint(const std::string &portname);
	
	QPoint providePortPoint(int num);
        QPoint providePortPoint(const std::string &portname);
	std::string providesPortName(int num);
        std::string usesPortName(int num);	
	QRect portRect(int portnum, PortType porttype);
	bool clickedPort(QPoint localpos, PortType &porttype,
			 std::string &portname);

public slots:
  void execute();
	void stop();
	void ui();

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent*);
	QRect nameRect;
        CIA::array1<std::string> up, pp;
private:
	QPopupMenu *menu;
	gov::cca::ports::UIPort::pointer uiPort;
	int pd; //distance between two ports
	int pw; //port width
	int ph; //prot height
public:
		gov::cca::ComponentID::pointer cid;
};

#endif



























