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
 *  NetworkCanvas.h: 
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

#ifndef SCIRun_Framework_NetworkCanvas_h
#define SCIRun_Framework_NetworkCanvas_h

#define BOOL int

#include <CCA/Components/Builder/BuilderWindow.h>
#include <qcanvas.h>
#include "Module.h"
#include "Connection.h"
#include "Core/CCA/PIDL/PIDL.h"
#include "Core/CCA/spec/cca_sidl.h"

using namespace std;

//namespace SCIRun {

class NetworkCanvasView:public QCanvasView
{
  Q_OBJECT
public:

  NetworkCanvasView(BuilderWindow* p2BuilderWindow, QCanvas* canvas, QWidget* parent=0);
  void setServices(const sci::cca::Services::pointer &services);
  virtual ~NetworkCanvasView();
  void addModule(const std::string& name, int x, int y, SSIDL::array1<std::string> & up, SSIDL::array1<std::string> &pp, const sci::cca::ComponentID::pointer &cid, bool reposition);
  void addConnection(Module *m1, const std::string & portname1, Module *m2, const std::string & portname2);	
  void removeConnection(QCanvasItem *c);
  void highlightConnection(QCanvasItem *c);
  void showPossibleConnections(Module *m, 
			       const std::string &protname, Module::PortType);
  void clearPossibleConnections();
  void removeAllConnections(Module *module);

  std::vector<Module*> getModules();
  std::vector<Connection*> getConnections();
  BuilderWindow* p2BuilderWindow;

  void addChild( Module* child, int x, int y, bool reposition);

public slots:
  void removeModule(Module *);

protected:
  void contentsMousePressEvent(QMouseEvent*);
  void contentsMouseMoveEvent(QMouseEvent*);
  void contentsMouseReleaseEvent(QMouseEvent*);
  //void contentsMouseDoubleClickEvent( QMouseEvent * );
  void NetworkCanvasView::viewportResizeEvent( QResizeEvent* );

private:
  Module* moving;
  Module* connecting;
  Module::PortType porttype;
  std::string portname;	
  QPoint moving_start;
  std::vector<Module*> modules;
  std::vector<Connection*> connections;
  std::vector<Connection*> possibleConns;
  Connection *highlightedConnection;
  NetworkCanvasView(const NetworkCanvasView&);
  NetworkCanvasView& operator=(const NetworkCanvasView&);
  sci::cca::Services::pointer services;
};
#endif









