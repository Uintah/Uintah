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


#include <CCA/Components/Builder/BuilderWindow.h>
#include <qcanvas.h>
#include "Module.h"
#include "Connection.h"
#include "BridgeConnection.h"
#include "Core/CCA/PIDL/PIDL.h"
#include "Core/CCA/spec/cca_sidl.h"


class NetworkCanvasView:public QCanvasView
{
  Q_OBJECT
public:

  NetworkCanvasView(BuilderWindow* p2BuilderWindow,
	    QCanvas* canvas, QWidget* parent=0);
  virtual ~NetworkCanvasView();

  void setServices(const sci::cca::Services::pointer &services);

  Module* addModule(const std::string& name,
		    int x, int y,
		    SSIDL::array1<std::string> &up,
		    SSIDL::array1<std::string> &pp,
		    const sci::cca::ComponentID::pointer &cid,
		    bool reposition);
  std::vector<Module*> getModules();

  void addConnection(Module *m1, const std::string & portname1,
		    Module *m2, const std::string & portname2);	
  void removeConnection(QCanvasItem *c);

  //Bridge:
  void addBridgeConnection(Module *m1, const std::string& portname1, Module *m2, const std::string& portname2);

  void highlightConnection(QCanvasItem *c);
  void showPossibleConnections(Module *m, 
			       const std::string &protname,
				Module::PortType);
  void clearPossibleConnections();
  void removeAllConnections(Module *module);

  std::vector<Connection*> getConnections();
  // make this private?
  BuilderWindow* p2BuilderWindow;

  void addChild(Module* child, int x, int y, bool reposition);

public slots:
  void removeModule(Module *);

protected:
  void contentsMousePressEvent(QMouseEvent*);
  void contentsMouseMoveEvent(QMouseEvent*);
  void contentsMouseReleaseEvent(QMouseEvent*);

  void contentsContextMenuEvent(QContextMenuEvent*) { p2BuilderWindow->componentContextMenu()->exec(QCursor::pos()); }

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









