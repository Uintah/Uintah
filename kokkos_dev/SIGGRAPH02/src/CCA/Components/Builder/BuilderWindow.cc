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
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/Builder/NetworkCanvasView.h>
#include <CCA/Components/Builder/Module.h>
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
#include <qiconset.h> 
#include <qtoolbutton.h> 

using namespace std;
using namespace SCIRun;
using namespace gov::cca;

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

#include "load.xpm"
#include "save.xpm"
#include "insert.xpm"
#include "info.xpm"
 
  new QToolButton( QIconSet( QPixmap(load)  ), "Load", QString::null,
                           this, SLOT(load()), fileTools, "load" );
  new QToolButton( QIconSet( QPixmap(save)  ), "Save", QString::null,
                           this, SLOT(save()), fileTools, "save" );
  new QToolButton( QIconSet( QPixmap(insert_xpm)  ), "Insert", QString::null,
                           this, SLOT(insert()), fileTools, "insert" );
  new QToolButton( QIconSet( QPixmap(info)  ), "Add Info", QString::null,
                           this, SLOT(addInfo()), fileTools, "addInfo" );

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

  buildPackageMenus();
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
  miniCanvas = new QCanvas();
  miniCanvas->setBackgroundColor(bgcolor);
  QCanvasView* miniview = new QCanvasView(miniCanvas, hsplit);
  miniview->setFixedHeight(204);
  miniview->setFixedWidth(204);
  int miniW=miniview->contentsRect().width();
  int miniH=miniview->contentsRect().height();
  miniCanvas->resize(miniW, miniH);



  QVBox* layout3 = new QVBox(hsplit);
  QHBox* layout4 = new QHBox(layout3);
  //QLabel* message_label = 
  new QLabel(" Messages: ", layout4);
  QMimeSourceFactory::defaultFactory()->setPixmap("SCIRun logo", QPixmap(SCIRun_logo));
  QLabel* logo_image = new QLabel("SCIRun logo", layout4);
  logo_image->setPixmap( QPixmap(SCIRun_logo));
  e = new QTextEdit( layout3, "editor" );
  e->setFocus();
  e->setReadOnly(true);
  e->setUndoRedoEnabled(false);


  gov::cca::ports::BuilderService::pointer builder = pidl_cast<gov::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
  }
  displayMsg("Framework URL: ");
  displayMsg(builder->getFramework()->getURL().getString().c_str());
  displayMsg("\n");  
  services->releasePort("cca.BuilderService");

  big_canvas = new QCanvas(2000,2000);
  big_canvas->setBackgroundColor(bgcolor);
  big_canvas_view = new NetworkCanvasView(this,big_canvas, vsplit);

  big_canvas_view->setServices(services);


  setCentralWidget( vsplit );
  statusBar()->message( "SCIRun 2.0.0 Ready");



  gov::cca::ports::ComponentEventService::pointer ces = pidl_cast<gov::cca::ports::ComponentEventService::pointer>(services->getPort("cca.ComponentEventService"));
  if(ces.isNull()){
    cerr << "Cannot get componentEventService!\n";
  } else {
    gov::cca::ports::ComponentEventListener::pointer listener(this);
    ces->addComponentEventListener(gov::cca::ports::AllComponentEvents,
				   listener, true);
    services->releasePort("cca.ComponentEventService");
  }
  updateMiniView();
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

MenuTree::MenuTree(BuilderWindow* builder, const std::string &url)
  :  builder(builder)
{
  this->url=url;
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
      child[n]=new MenuTree(builder, url);
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
  cerr << "MenuTree::instantiate...URL="<<url<<endl;
  builder->instantiateComponent(cd, url);
}

void BuilderWindow::buildRemotePackageMenus(const  gov::cca::ports::ComponentRepository::pointer &reg,
					    const std::string &frameworkURL)
{
  if(reg.isNull()){
    cerr << "Cannot get component registry, not building component menus\n";
    return;
  }
  vector<gov::cca::ComponentClassDescription::pointer> list = reg->getAvailableComponentClasses();
  map<string, MenuTree*> menus;
  for(vector<gov::cca::ComponentClassDescription::pointer>::iterator iter = list.begin();
      iter != list.end(); iter++){
    string model = (*iter)->getModelName();
    if(model!="CCA") continue;
    model=frameworkURL;
    if(menus.find(model) == menus.end())
      menus[model]=new MenuTree(this, frameworkURL);
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
}


void BuilderWindow::buildPackageMenus()
{
  gov::cca::ports::ComponentRepository::pointer reg = pidl_cast<gov::cca::ports::ComponentRepository::pointer>(services->getPort("cca.ComponentRepository"));
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
      menus[model]=new MenuTree(this,"");
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
  services->releasePort("cca.ComponentRepository");
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
  //should stop and close socket in CCACommunicator first
  Thread::exitAll(0);
}


void BuilderWindow::about()
{
  cerr << "BuilderWindow::about not finished\n";
  (new QMessageBox())->about(this, "About", "CCA Builder (SCIRun Implementation)");
}

void BuilderWindow::instantiateComponent(const gov::cca::ComponentClassDescription::pointer& cd,
					 const std::string &url)
{
  cerr << "Should wait for component to be committed...\n";
  gov::cca::ports::BuilderService::pointer builder = pidl_cast<gov::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
    return;
  }
  cerr << "Should put properties on component before creating\n";

 
  gov::cca::ComponentID::pointer cid=builder->createInstance(cd->getClassName(), cd->getClassName(), gov::cca::TypeMap::pointer(0),url);

  CIA::array1<std::string> usesPorts=builder->getUsedPortNames(cid);
  CIA::array1<std::string> providesPorts=builder->getProvidedPortNames(cid);

  services->releasePort("cca.BuilderService");
  if(cd->getClassName()!="SCIRun.Builder"){
    big_canvas_view->addModule(cd->getClassName(), usesPorts, providesPorts,
			       cid);
  }
}

void BuilderWindow::componentActivity(const gov::cca::ports::ComponentEvent::pointer& e)
{
  cerr << "Got component activity event " << e->getEventType() << " for " << e->getComponentID()->getInstanceName() << '\n';
  displayMsg("Some event occurs\n");
}

void BuilderWindow::displayMsg(const char *msg)
{
  e->insert(msg);
} 	

void BuilderWindow::updateMiniView()
{
  // assign the temporary list
  // needed for coordinates of each module
  QCanvasItemList tempQCL = miniCanvas->allItems();

  for(unsigned int i=0;i<tempQCL.size();i++){
    delete tempQCL[i];
  }
  
  // assign modules to local variable
  std::vector<Module*> modules = big_canvas_view->getModules();

  double scaleH=double(big_canvas->width())/miniCanvas->width();
  double scaleV=double(big_canvas->height())/miniCanvas->height();

  QCanvasRectangle* viewableRect = new QCanvasRectangle( //int(hSBar->value()/scaleH),int(vSBar->value()/scaleV), 
							   int(big_canvas_view->contentsX()/scaleH), 
							   int(big_canvas_view->contentsY()/scaleV),
							   int(big_canvas_view->visibleWidth()/scaleH), 
							   int(big_canvas_view->visibleHeight()/scaleV), miniCanvas );
							   
  viewableRect->show();

  for(unsigned int i = 0; i < modules.size(); i++ ) 
  {
     
    QPoint pm=modules[i]->posInCanvas();

    QCanvasRectangle *rect=new QCanvasRectangle( int(pm.x()/scaleH), int(pm.y()/scaleV), 
						 int(modules[i]->width()/scaleH), int(modules[i]->height()/scaleV),
						 miniCanvas );
    rect->setBrush( Qt::white );
    rect->show();
  }
  miniCanvas->update();
}






