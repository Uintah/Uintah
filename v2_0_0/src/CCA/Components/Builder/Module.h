/*The contents of this file are subject to the University of Utah Public
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
 *  Module.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef MODULE_H
#define MODULE_H

#include <qframe.h>
#include <qpoint.h>
#include <qrect.h>
#include <qpopmenu.h>
#include <qprogressbar.h>
#include "Core/CCA/spec/cca_sidl.h"
#include "Core/CCA/PIDL/PIDL.h"
class NetworkCanvasView;

class Module:public QFrame
{
  Q_OBJECT

public:
  enum PortType{USES, PROVIDES}; 
  Module(NetworkCanvasView *parent, const std::string& name,
	 SSIDL::array1<std::string> & up, SSIDL::array1<std::string> &pp,
	 const sci::cca::Services::pointer& services,
	 const sci::cca::ComponentID::pointer &cid);
  QPoint usePortPoint(int num);
  QPoint posInCanvas();
  QPoint usePortPoint(const std::string &portname);
	
  QPoint providePortPoint(int num);
  QPoint providePortPoint(const std::string &portname);
  std::string providesPortName(int num);
  std::string usesPortName(int num);	
  QRect portRect(int portnum, PortType porttype);
  bool clickedPort(QPoint localpos, PortType &porttype,
		   std::string &portname);

  std::string moduleName;
  std::string displayName;
public slots:
  void go();
  void stop();
  void destroy();
  void ui();
signals:
  void destroyModule(Module *);

protected:
  void paintEvent(QPaintEvent *e);
  void mousePressEvent(QMouseEvent*);
  QRect nameRect;
  //  std::string moduleName;
  std::string instanceName;
  SSIDL::array1<std::string> up, pp;
private:
  sci::cca::Services::pointer services;
  QPopupMenu *menu;
  int pd; //distance between two ports
  int pw; //port width
  int ph; //prot height
  bool hasGoPort;
  bool hasUIPort;
  QProgressBar *progress;
  NetworkCanvasView * viewWindow;
public:
  sci::cca::ComponentID::pointer cid;
};

#endif
