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

class NetworkCanvasView;

class QAction;
class QCanvasView;
class QVBox;
class QFont;
class QPopupMenu;
class QSplitter;
class QLabel;

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
    void add(const std::vector<std::string>& name,
	     int nameindex,
	     const sci::cca::ComponentClassDescription::pointer& desc,
	     const std::string& fullname);
    void coalesce();
    void populateMenu( QPopupMenu* );
    void clear();

    sci::cca::ComponentClassDescription::pointer cd;

private:
    BuilderWindow* builder;
    std::map<std::string, MenuTree*> child;
    std::string url;

private slots:
    void instantiateComponent();  
};

/**
 *
 * \class BuilderWindow
 *  SCIRun2 main window
 */
class BuilderWindow : public QMainWindow,
		    public sci::cca::ports::ComponentEventListener
{
    Q_OBJECT
public:
    BuilderWindow(const sci::cca::Services::pointer& services);
    virtual ~BuilderWindow();
    Module* instantiateComponent(const sci::cca::ComponentClassDescription::pointer&);
    Module* instantiateComponent(const std::string& className,
				const std::string& type,
				const std::string& loaderName);
    // From sci::cca::ComponentEventListener
    void componentActivity(const sci::cca::ports::ComponentEvent::pointer& msgTextEdit);
    void displayMsg(const char *);
    void displayMsg(const QString &);
    void buildRemotePackageMenus(const sci::cca::ports::ComponentRepository::pointer &reg,
				 const std::string &frameworkURL);

    std::vector<Module*> updateMiniView_modules;
    std::vector<int> packageMenuIDs;

public slots:
    void updateMiniView();

protected:
    void closeEvent( QCloseEvent* );

private:
    BuilderWindow(const BuilderWindow&);
    BuilderWindow& operator=(const BuilderWindow&);
    void setupFileActions();
    void setupClusterActions();
    void buildPackageMenus();
    void insertHelpMenu();
    void writeFile();
    QFont *bFont;
    QSplitter *vsplit;
    QSplitter *hsplit;
    QVBox *vBox;
    QLabel *msgLabel;
    QTextEdit *msgTextEdit;
    QCanvas *miniCanvas,
	    *networkCanvas;
    QCanvasView *miniView;
    NetworkCanvasView* networkCanvasView;
    QCanvasItemList tempQCL;
    QAction *loadAction,
	*insertAction,
	*saveAction,
	*saveAsAction,
	*clearAction,
	*addInfoAction,
	*quitAction,
	*exitAction,
	*addClusterAction,
	*rmClusterAction,
	*refreshAction;

    QString filename;
    sci::cca::Services::pointer services;
    std::vector<QCanvasRectangle*> miniRect;

private slots:
    void save();
    void saveAs();
    void load();
    void insert();
    void clear();
    void addInfo();
    void exit();
    void clusterAdd();
    void clusterRemove();
    void mxn_add();
    void performance_mngr();
    void performance_tau_add();
    void demos();
    void about();
    void addCluster();
    void rmCluster();
    void refresh();
};

} // end namespace SCIRun

#endif



