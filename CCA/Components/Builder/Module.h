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
  Module(QWidget *parent, const char *name, gov::cca::ports::UIPort::pointer uip, CIA::array1<std::string> & up, CIA::array1<std::string> &pp);
	QPoint usePortPoint();
	QPoint providePortPoint();
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
};

#endif
