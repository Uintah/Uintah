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
#include <CCA/Components/Builder/PortIcon.h>
#include <CCA/Components/Builder/Module.h>
#include <CCA/Components/Builder/Connection.h>
#include <CCA/Components/Builder/BridgeConnection.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/sci_sidl.h>

#include <qcanvas.h>

typedef std::map<std::string, Module*> ModuleMap;

class NetworkCanvasView
    : public QCanvasView,
      public sci::cca::ports::ConnectionEventListener
{
  Q_OBJECT
public:
  NetworkCanvasView(BuilderWindow* p2BuilderWindow,
                    QCanvas* canvas, QWidget* parent,
                    const sci::cca::Services::pointer &services);
  virtual ~NetworkCanvasView();

  Module* addModule(const std::string& name,
                    int x, int y,
                    const sci::cca::ComponentID::pointer &cid,
                    bool reposition);
  Module* getModule(const std::string &instanceName) const;
  inline const ModuleMap* getModules() const { return &modules; }

  void connectComponents(Module *mUses, const std::string &pUses,
                         Module *mProvides, const std::string &pProvides); 

  void addConnection(Module *mUses, const std::string &pUses,
                     Module *mProvides, const std::string &pProvides,
                     sci::cca::ConnectionID::pointer connID);

  void removeConnection(QCanvasItem *c);

  //Bridge:
  void addBridgeConnection(Module *m1, const std::string& portname1,
                           Module *m2, const std::string& portname2);

  void highlightConnection(QCanvasItem *c);
  void showPossibleConnections(Module *m, 
                               const std::string &protname,
                               PortIcon::PortType);
  void clearPossibleConnections();
  void removeAllConnections(Module *module);
  inline const std::vector<Connection*>& getConnections() const { return connections; }

  //void addPendingConnection(const sci::cca::ComponentID::pointer &uCid, const std::string &pUses, const sci::cca::ComponentID::pointer &uCid, const std::string pProvides);

  // make this private?
  BuilderWindow* p2BuilderWindow;

  void addChild(Module* child, int x, int y, bool reposition);
  virtual void connectionActivity(const sci::cca::ports::ConnectionEvent::pointer &e);

  void removeModules();

public slots:
    void removeModule(Module *);

protected:
  void contentsMousePressEvent(QMouseEvent*);
  void contentsMouseMoveEvent(QMouseEvent*);
  void contentsMouseReleaseEvent(QMouseEvent*);
  void contentsContextMenuEvent(QContextMenuEvent*);

  //void contentsMouseDoubleClickEvent( QMouseEvent * );
  void NetworkCanvasView::viewportResizeEvent(QResizeEvent*);

private:
  NetworkCanvasView(const NetworkCanvasView&);
  NetworkCanvasView& operator=(const NetworkCanvasView&);

  QFont *displayFont;
  QPoint moving_start;
  Module *moving;
  Module *connecting;
  Connection *highlightedConnection;

  PortIcon::PortType porttype;
  std::string portname;

  ModuleMap modules;
  //std::stack<sci::cca::ConnectionID::pointer> pendingConnections;
  std::vector<Connection*> connections;
  std::vector<Connection*> possibleConns;
  sci::cca::Services::pointer services;

  const static int TXT_OFFSET = 4;
};

inline void NetworkCanvasView::viewportResizeEvent(QResizeEvent* p2QResizeEvent)
{
  QScrollView::viewportResizeEvent(p2QResizeEvent);
  p2BuilderWindow->updateMiniView();  
}


#endif

