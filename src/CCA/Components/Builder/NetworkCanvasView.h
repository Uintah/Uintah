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

#include <qcanvas.h>
#include "Module.h"
#include "Connection.h"
#include "Core/CCA/Component/PIDL/PIDL.h"
#include "Core/CCA/spec/cca_sidl.h"
//namespace SCIRun {

class NetworkCanvasView:public QCanvasView
{
public:
  NetworkCanvasView(QCanvas* canvas, QWidget* parent=0);
  void setServices(const gov::cca::Services::pointer &services);
  virtual ~NetworkCanvasView();
  void addModule(const char *name, gov::cca::ports::UIPort::pointer &uip, CIA::array1<std::string> & up, CIA::array1<std::string> &pp, const gov::cca::ComponentID::pointer &cid);
  void addConnection(Module *m1, int portnum1, Module *m2, int portnum2);	
  void removeConnection(QCanvasItem *c);
protected:
  void contentsMousePressEvent(QMouseEvent*);
  void contentsMouseMoveEvent(QMouseEvent*);
  void contentsMouseReleaseEvent(QMouseEvent*);
  //void contentsMouseDoubleClickEvent( QMouseEvent * );
private:
  Module* moving;
  Module* connecting;
  Module::PortType porttype;
  int portnum;	
  QPoint moving_start;
  std::vector<Module*> modules;
  std::vector<Connection*> connections;
  NetworkCanvasView(const NetworkCanvasView&);
  NetworkCanvasView& operator=(const NetworkCanvasView&);
  
  gov::cca::Services::pointer services;
};
//}
#endif



