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

namespace SCIRun {

class BuilderWindow;
/**
 * \class MenuTree
 *
 */
class MenuTree : public QObject
{
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

class BuilderWindow : public QMainWindow,
                      public sci::cca::ports::ComponentEventListener
{
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
  
  Module* instantiateComponent(const sci::cca::ComponentClassDescription::pointer&);
  Module* instantiateComponent(const std::string& className, const std::string& type, const std::string& loaderName);
  
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

} // end namespace SCIRun

#endif



