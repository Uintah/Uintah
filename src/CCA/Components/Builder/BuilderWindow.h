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
 *  BuilderWindow.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_BuilderWindow_h
#define SCIRun_Framework_BuilderWindow_h

#include <Core/CCA/spec/cca_sidl.h>
#include <map>
#include <vector>
#include <qcanvas.h>
#include <qtextedit.h>
#include <qmainwindow.h>
#include "Module.h"

class QPopupMenu;
class NetworkCanvasView;

using namespace std;

namespace SCIRun {
  class BuilderWindow;
  class MenuTree : public QObject {
    Q_OBJECT
  public:
    MenuTree(BuilderWindow* builder, const std::string &url);
    virtual ~MenuTree();
    sci::cca::ComponentClassDescription::pointer cd;
    void add(const std::vector<std::string>& name, int nameindex,
	     const sci::cca::ComponentClassDescription::pointer& desc,
	     const std::string& fullname);
    void coalesce();
    void populateMenu(QPopupMenu*);
    void clear();
  private:
    std::map<std::string, MenuTree*> child;
    BuilderWindow* builder;
    std::string url;

  private slots:
    void instantiateComponent();
    
  };

  class BuilderWindow : public QMainWindow, public sci::cca::ports::ComponentEventListener {
    Q_OBJECT
  protected:
    void closeEvent( QCloseEvent* );

  public slots:
    void updateMiniView();

  private slots:
    void save();
    void saveAs();
    void load();
    void insert();
    void clear();
    void addInfo();
    void exit();
    void cluster_add();
    void cluster_remove();
    void mxn_add();
    void performance_mngr();
    void performance_tau_add();
    void demos();
    void about();
    void addCluster();
    void rmCluster();
    void refresh();
  public:
    BuilderWindow(const sci::cca::Services::pointer& services);
    virtual ~BuilderWindow();

    void instantiateComponent(const sci::cca::ComponentClassDescription::pointer&);

    // From sci::cca::ComponentEventListener
    void componentActivity(const sci::cca::ports::ComponentEvent::pointer& e);
    void displayMsg(const char *); 
    void buildRemotePackageMenus(const sci::cca::ports::ComponentRepository::pointer &reg, const std::string &frameworkURL);

    std::vector<Module*> updateMiniView_modules;
    std::vector<int> packageMenuIDs;

    void insertHelpMenu();

  private:
    QString filename;
    sci::cca::Services::pointer services;
    void buildPackageMenus();
    BuilderWindow(const BuilderWindow&);
    BuilderWindow& operator=(const BuilderWindow&);
    NetworkCanvasView* big_canvas_view;
    QTextEdit *e;	
    QCanvas* miniCanvas;
    QCanvas* big_canvas;
    QCanvasItemList tempQCL;
    std::vector<QCanvasRectangle*> miniRect;
  };
}

#endif



