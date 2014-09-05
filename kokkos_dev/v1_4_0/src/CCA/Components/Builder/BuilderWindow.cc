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
 *  BuilderWindow.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <CCA/Components/Builder/BuilderWindow.h>
#include <CCA/Components/Builder/SCIRun.xpm>
#include <CCA/Components/Builder/NetworkCanvasView.h>
#include <CCA/Components/Builder/ModuleCanvasItem.h>
#include <Core/Thread/Thread.h>
#include <Core/Containers/StringUtil.h>
#include <qaction.h>
#include <qcanvas.h>
#include <qlabel.h>
#include <qmenubar.h>
#include <qmessagebox.h>
#include <qpopupmenu.h>
#include <qsplitter.h>
#include <qstatusbar.h>
#include <qtextedit.h>
#include <qvbox.h>
#include <qwhatsthis.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

BuilderWindow::BuilderWindow(const gov::cca::Services::pointer& services)
  : QMainWindow(0, "SCIRun", WDestructiveClose|WType_TopLevel),
    services(services)
{
  addReference(); // Do something better than this!

  // Save
  QAction* saveAction = new QAction("Save", "&Save", CTRL+Key_S, this, "save");
  connect(saveAction, SIGNAL(activated()), this, SLOT(save()));

  // Save As
  QAction* saveAsAction = new QAction("Save File As", "Save &As", 0, this,
				      "save as");
  connect(saveAsAction, SIGNAL(activated()), this, SLOT(saveAs()));

  // load
  QAction* loadAction = new QAction("Load", "&Load", CTRL+Key_L, this, "load");
  connect(loadAction, SIGNAL(activated()), this, SLOT(load()));

  // insert
  QAction* insertAction = new QAction("Insert", "&Insert", 0, this, "insert");
  connect(insertAction, SIGNAL(activated()), this, SLOT(insert()));

  // clear
  QAction* clearAction = new QAction("Clear", "&Clear", 0, this, "clear");
  connect(clearAction, SIGNAL(activated()), this, SLOT(clear()));

  // addInfo
  QAction* addInfoAction = new QAction("Add Info", "Add &Info", 0, this, "addInfo");
  connect(addInfoAction, SIGNAL(activated()), this, SLOT(addInfo()));

  // quit
  QAction* quitAction = new QAction("Quit", "Quit GUI (Leave SCIRun rnning)", CTRL+Key_Q, this, "Quit");
  connect(quitAction, SIGNAL(activated()), this, SLOT(close()));

  // exit
  QAction* exitAction = new QAction("Exit", "Exit (and terminate all components)", CTRL+Key_X, this, "Exit");
  connect(exitAction, SIGNAL(activated()), this, SLOT(exit()));

  QToolBar* fileTools = new QToolBar(this, "file operations");
  fileTools->setLabel("File Operations");
  saveAction->addTo(fileTools);
  loadAction->addTo(fileTools);
  insertAction->addTo(fileTools);
  addInfoAction->addTo(fileTools);

  QPopupMenu* file = new QPopupMenu(this);
  menuBar()->insertItem("&File", file);
  file->insertTearOffHandle();
  saveAction->addTo(file);
  saveAsAction->addTo(file);
  loadAction->addTo(file);
  insertAction->addTo(file);
  clearAction->addTo(file);
  addInfoAction->addTo(file);
  file->insertSeparator();
  quitAction->addTo(file);
  exitAction->addTo(file);

  menuBar()->insertSeparator();
  QPopupMenu * help = new QPopupMenu( this );
  help->insertTearOffHandle();
  menuBar()->insertItem( "&Help", help );
  help->insertItem( "&About", this, SLOT(about()), Key_F1 );
  help->insertSeparator();
  help->insertItem( "What's &This", this, SLOT(whatsThis()),
		    SHIFT+Key_F1 );

  QColor bgcolor(0, 51, 102);
  QSplitter* vsplit = new QSplitter(Qt::Vertical, this);
  QSplitter* hsplit = new QSplitter(Qt::Horizontal, vsplit);
  QCanvas* minicanvas = new QCanvas(100, 100);
  minicanvas->setBackgroundColor(bgcolor);
  QCanvasView* miniview = new QCanvasView(minicanvas, hsplit);
  QVBox* layout3 = new QVBox(hsplit);
  QHBox* layout4 = new QHBox(layout3);
  QLabel* message_label = new QLabel(" Messages: ", layout4);
  QMimeSourceFactory::defaultFactory()->setPixmap("fileopen", QPixmap(SCIRun_logo));
  QLabel* logo_image = new QLabel("SCIRun logo", layout4);
  //QLabel* logo_image = new QLabel("<img src=\"SCIRun_logo">", layout4);
  QTextEdit* e = new QTextEdit( layout3, "editor" );
  e->setFocus();
  e->setReadOnly(true);
  e->setUndoRedoEnabled(false);
  cerr << "Should append framework URL to message window\n";
  QCanvas* big_canvas = new QCanvas(2000,2000);
  big_canvas->setBackgroundColor(bgcolor);
  NetworkCanvasView* big_view = new NetworkCanvasView(big_canvas, vsplit);
  setCentralWidget( vsplit );
  statusBar()->message( "SCIRun 2.0.0 Ready");

  buildPackageMenus();

  gov::cca::ports::ComponentEventService::pointer ces = pidl_cast<gov::cca::ports::ComponentEventService::pointer>(services->getPort("cca.componentEventService"));
  if(ces.isNull()){
    cerr << "Cannot get componentEventService!\n";
  } else {
    gov::cca::ports::ComponentEventListener::pointer listener(this);
    ces->addComponentEventListener(gov::cca::ports::AllComponentEvents,
				   listener, true);
    services->releasePort("cca.ComponentEventService");
  }
  ModuleCanvasItem* i = new ModuleCanvasItem(big_canvas);
  i->move(20,20);
  i->show();
}


BuilderWindow::~BuilderWindow()
{
  cerr << "~BuilderWindow called!\n";
}

void BuilderWindow::closeEvent( QCloseEvent* ce )
{
  switch(QMessageBox::information(this, "SCIRun",
				  "Do you want to quit the SCIRun builder and leave all components running, or exit SCIRun and terminate all components?",
				  "Quit (leave SCIRun running)", "Exit (terminate all components)", "Cancel", 0, 2)){
  case 0:
    ce->accept();
    break;
  case 1:
    ce->accept();
    exit();
    break;
  case 2:
  default:
    ce->ignore();
    break;
  }
}

MenuTree::MenuTree(BuilderWindow* builder)
  :  builder(builder)
{
}

MenuTree::~MenuTree()
{
  for(map<string, MenuTree*>::iterator iter = child.begin();
      iter != child.end(); iter++)
    delete iter->second;
}
  
void MenuTree::add(const vector<string>& name, int nameindex,
		   const gov::cca::ComponentClassDescription::pointer& desc,
		   const std::string& fullname)
{
  if(nameindex == (int)name.size()){
    if(!cd.isNull())
      cerr << "Duplicate component: " << fullname << '\n';
    else
      cd = desc;
  } else {
    const string& n = name[nameindex];
    map<string, MenuTree*>::iterator iter = child.find(n);
    if(iter == child.end())
      child[n]=new MenuTree(builder);
    child[n]->add(name, nameindex+1, desc, fullname);
  }
}

void MenuTree::coalesce()
{
  for(map<string, MenuTree*>::iterator iter = child.begin();
      iter != child.end(); iter++){
    MenuTree* c = iter->second;
    while(c->child.size() == 1){
      map<string, MenuTree*>::iterator grandchild = c->child.begin();
      string newname = iter->first+"."+grandchild->first;
      MenuTree* gc = grandchild->second;
      c->child.clear(); // So that grandchild won't get deleted...
      delete c;
      child.erase(iter);
      child[newname]=gc;
      iter=child.begin();
      c=gc;
    }
    c->coalesce();
  }
}

void MenuTree::populateMenu(QPopupMenu* menu)
{
  menu->insertTearOffHandle();
  for(map<string, MenuTree*>::iterator iter = child.begin();
      iter != child.end(); iter++){
    if(!iter->second->cd.isNull()){
      menu->insertItem(iter->first.c_str(), iter->second, SLOT(instantiateComponent()));
    } else {
      QPopupMenu* submenu = new QPopupMenu(menu);
      iter->second->populateMenu(submenu);
      menu->insertItem(iter->first.c_str(), submenu);
    }
  }
}

void MenuTree::instantiateComponent()
{
  cerr << "MenuTree::instantiate...\n";
  builder->instantiateComponent(cd);
}

void BuilderWindow::buildPackageMenus()
{
  gov::cca::ports::ComponentRepository::pointer reg = pidl_cast<gov::cca::ports::ComponentRepository::pointer>(services->getPort("cca.componentRepository"));
  if(reg.isNull()){
    cerr << "Cannot get component registry, not building component menus\n";
    return;
  }
  vector<gov::cca::ComponentClassDescription::pointer> list = reg->getAvailableComponentClasses();
  map<string, MenuTree*> menus;
  for(vector<gov::cca::ComponentClassDescription::pointer>::iterator iter = list.begin();
      iter != list.end(); iter++){
    string model = (*iter)->getModelName();
    if(menus.find(model) == menus.end())
      menus[model]=new MenuTree(this);
    string name = (*iter)->getClassName();
    vector<string> splitname = split_string(name, '.');
    menus[model]->add(splitname, 0, *iter, name);
  }
  for(map<string, MenuTree*>::iterator iter = menus.begin();
      iter != menus.end(); iter++)
    iter->second->coalesce();

  for(map<string, MenuTree*>::iterator iter = menus.begin();
      iter != menus.end(); iter++){
    QPopupMenu* menu = new QPopupMenu(this);
    iter->second->populateMenu(menu);
    menuBar()->insertItem(iter->first.c_str(), menu);
  }
  services->releasePort("cca.componentRegistry");
}

void BuilderWindow::save()
{
  cerr << "BuilderWindow::save not finished\n";
}

void BuilderWindow::saveAs()
{
  cerr << "BuilderWindow::saveAs not finished\n";
}

void BuilderWindow::load()
{
  cerr << "BuilderWindow::load not finished\n";
}

void BuilderWindow::insert()
{
  cerr << "BuilderWindow::insert not finished\n";
}

void BuilderWindow::clear()
{
  cerr << "BuilderWindow::clear not finished\n";
}

void BuilderWindow::addInfo()
{
  cerr << "BuilderWindow::addInfo not finished\n";
}

void BuilderWindow::exit()
{
  cerr << "Exit should ask framework to shutdown instead!\n";
  Thread::exitAll(0);
}


void BuilderWindow::about()
{
  cerr << "BuilderWindow::about not finished\n";
}

void BuilderWindow::instantiateComponent(const gov::cca::ComponentClassDescription::pointer& cd)
{
  cerr << "Should wait for component to be committed...\n";
  gov::cca::ports::BuilderService::pointer builder = pidl_cast<gov::cca::ports::BuilderService::pointer>(services->getPort("cca.builderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
  }
  cerr << "Should put properties on component before creating\n";
  builder->createInstance(cd->getClassName(), cd->getClassName(), gov::cca::TypeMap::pointer(0));
  services->releasePort("cca.builderService");
}

void BuilderWindow::componentActivity(const gov::cca::ports::ComponentEvent::pointer& e)
{
  cerr << "Got component activity event " << e->getEventType() << " for " << e->getComponentID()->getInstanceName() << '\n';
}

