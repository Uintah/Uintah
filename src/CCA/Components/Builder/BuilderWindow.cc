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
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/Builder/NetworkCanvasView.h>
#include <CCA/Components/Builder/Module.h>
#include <SCIRun/TypeMap.h>
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
#include <qfiledialog.h>
#include <fstream>
#include <qtextstream.h>

using namespace std;
using namespace SCIRun;
using namespace sci::cca;

BuilderWindow::BuilderWindow(const sci::cca::Services::pointer& services)
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

  // addCluster
  QAction* addClusterAction = new QAction("Add Cluster", "&Add Cluster", CTRL+Key_A, this, "Add Cluster");
  connect(addClusterAction, SIGNAL(activated()), this, SLOT(addCluster()));

  // rmCluster
  QAction* rmClusterAction = new QAction("Remove Cluster", "&Remove Cluster", CTRL+Key_R, this, "Remove Cluster");
  connect(rmClusterAction, SIGNAL(activated()), this, SLOT(rmCluster()));

  // refresh
  QAction* refreshAction = new QAction("Refresh", "Rfresh Menu", CTRL+Key_R, this, "refresh");
  connect(refreshAction, SIGNAL(activated()), this, SLOT(refresh()));

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


  QPopupMenu* clusters = new QPopupMenu(this);
  menuBar()->insertItem("&Clusters", clusters);
  clusters->insertTearOffHandle();
  addClusterAction->addTo(clusters);
  rmClusterAction->addTo(clusters);
  refreshAction->addTo(clusters);
/*

  QPopupMenu* cluster = new QPopupMenu(this);
  menuBar()->insertItem("&Clusters", cluster );
  cluster->insertItem( "&Add a cluster", this, SLOT( cluster_add() ), Key_F1 );
  cluster->insertItem( "&Remove a cluster", this, SLOT( cluster_remove() ), Key_F2 );
  QPopupMenu* mxn = new QPopupMenu( this );
  menuBar()->insertItem("&MxN", mxn );
  mxn->insertItem( "&MxN-enabled Component_1", this, SLOT( mxn_add() ), Key_F1 );
  mxn->insertItem( "&MxN-enabled Component_2", this, SLOT( mxn_add() ), Key_F2 );
  mxn->insertItem( "&MxN-enabled Component_3", this, SLOT( mxn_add() ), Key_F3 );
  QPopupMenu* performance = new QPopupMenu(this);
  menuBar()->insertItem( "&Performance", performance );
  performance->insertItem( "&Performance Manager", this, SLOT( performance_mngr() ), Key_F1 );
  performance->insertItem( "&Add Tau component", this, SLOT( performance_tau_add() ), Key_F2 );
*/
  buildPackageMenus();

  insertHelpMenu();
/*  
help->insertTearOffHandle();
  menuBar()->insertItem( "&Help", help );
  help->insertItem( "&Demos", this, SLOT( demos() ), Key_F1 );
  help->insertItem( "What's &This", this, SLOT(whatsThis()),
		    SHIFT+Key_F2 );
  help->insertSeparator();
  help->insertItem( "&About", this, SLOT(about()), Key_F3 );
*/
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
  new QLabel(" Messages: ", layout4);
  QMimeSourceFactory::defaultFactory()->setPixmap("SCIRun logo", QPixmap(SCIRun_logo));
  QLabel* logo_image = new QLabel("SCIRun logo", layout4);
  logo_image->setPixmap( QPixmap(SCIRun_logo));
  e = new QTextEdit( layout3, "editor" );
  e->setFocus();
  e->setReadOnly(true);
  e->setUndoRedoEnabled(false);

  sci::cca::ports::BuilderService::pointer builder = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
  }
  displayMsg("Framework URL: ");
  //displayMsg(builder->getFramework()->getURL().getString().c_str());
  displayMsg("Framework URL should be displaied here");
  displayMsg("\n"); 
  services->releasePort("cca.BuilderService");

  big_canvas = new QCanvas(2000,2000);
  big_canvas->setBackgroundColor(bgcolor);
  big_canvas_view = new NetworkCanvasView(this,big_canvas, vsplit);
  big_canvas_view->setServices(services);

  setCentralWidget( vsplit );
  statusBar()->message( "SCIRun 2.0.0 Ready");

  sci::cca::ports::ComponentEventService::pointer ces = pidl_cast<sci::cca::ports::ComponentEventService::pointer>(services->getPort("cca.ComponentEventService"));
  if(ces.isNull()){
    cerr << "Cannot get componentEventService!\n";
  } else {
    sci::cca::ports::ComponentEventListener::pointer listener(this);
    ces->addComponentEventListener(sci::cca::ports::AllComponentEvents,
				   listener, true);
    services->releasePort("cca.ComponentEventService");
  }
  updateMiniView();
  filename=QString::null;
}

BuilderWindow::~BuilderWindow()
{
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

void MenuTree::clear()
{
  child.clear();
}

MenuTree::~MenuTree()
{
  for(map<string, MenuTree*>::iterator iter = child.begin();
      iter != child.end(); iter++)
    delete iter->second;
}
  
void MenuTree::add(const vector<string>& name, int nameindex,
		   const sci::cca::ComponentClassDescription::pointer& desc,
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
  builder->instantiateComponent(cd);
}

void BuilderWindow::buildRemotePackageMenus(const  sci::cca::ports::ComponentRepository::pointer &reg,
					    const std::string &frameworkURL)
{
  if(reg.isNull()){
    cerr << "Cannot get component registry, not building component menus\n";
    return;
  }
  vector<sci::cca::ComponentClassDescription::pointer> list = reg->getAvailableComponentClasses();
  map<string, MenuTree*> menus;
  for(vector<sci::cca::ComponentClassDescription::pointer>::iterator iter = list.begin();
      iter != list.end(); iter++){
    //model name could be obtained somehow locally.
    //and we can assume that the remote component model is always "CCA"
    string model = "CCA"; //(*iter)->getModelName();
    if(model!="CCA") continue;
    model=frameworkURL;
    if(menus.find(model) == menus.end())
      menus[model]=new MenuTree(this, frameworkURL);
    string name = (*iter)->getComponentClassName();
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
  //remove outdated package menu first
  for(unsigned int i=0; i<packageMenuIDs.size(); i++){
    menuBar()->removeItem(packageMenuIDs[i]);
  }

  sci::cca::ports::ComponentRepository::pointer reg = pidl_cast<sci::cca::ports::ComponentRepository::pointer>(services->getPort("cca.ComponentRepository"));
  if(reg.isNull()){
    cerr << "Cannot get component registry, not building component menus\n";
    return;
  }

  vector<sci::cca::ComponentClassDescription::pointer> list = reg->getAvailableComponentClasses();
  map<string, MenuTree*> menus;
  for(vector<sci::cca::ComponentClassDescription::pointer>::iterator iter = list.begin();
    iter != list.end(); iter++){
    //model name could be obtained somehow locally.
    //and we can assume that the remote component model is always "CCA"
    string model = (*iter)->getComponentModelName();
    string loaderName = (*iter)->getLoaderName();
    if(menus.find(model) == menus.end())
      menus[model]=new MenuTree(this,"");
    string name = (*iter)->getComponentClassName();

    if(loaderName!="") name+="\t@"+loaderName;
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
    int menuID= menuBar()->insertItem(iter->first.c_str(), menu);
    packageMenuIDs.push_back(menuID);
  }
  services->releasePort("cca.ComponentRepository");

  insertHelpMenu();
}

void BuilderWindow::save()
{
  if(filename.isEmpty()){
    QString fn = QFileDialog::getSaveFileName( QString::null, "Network File (*.net)", this );
    if( !fn.isEmpty() ) {
      filename = fn;
      save();
    }
    else {
      statusBar()->message( "Saving aborted", 2000 );
      return;
    }
  }
  QCanvasItemList tempQCL = miniCanvas->allItems();
  ofstream saveOutputFile(filename);

  std::vector<Module*> saveModules = big_canvas_view->getModules();
  std::vector<Connection*> saveConnections = big_canvas_view->getConnections();

  saveOutputFile << saveModules.size() << endl;
  saveOutputFile << saveConnections.size() << endl;
  
  if( saveOutputFile.is_open() )
  {
    for( unsigned int j = 0; j < saveModules.size(); j++ )
    {
      saveOutputFile << saveModules[j]->moduleName << endl;
      saveOutputFile << saveModules[j]->x() << endl;
      saveOutputFile << saveModules[j]->y() << endl;
    }

    for( unsigned int k = 0; k < saveConnections.size(); k++ )
    {
      Module * getProvidesModule();

      Module * um=saveConnections[k]->getUsesModule();
      Module * pm=saveConnections[k]->getProvidesModule();
      unsigned int iu=0;
      unsigned int ip=0;
      for(unsigned int i=0; i<saveModules.size();i++){
	if(saveModules[i]==um) iu=i;
	if(saveModules[i]==pm) ip=i;
      }
      saveOutputFile << iu<<" "<< saveConnections[k]->getUsesPortName() << " "
		     << ip<<" "<< saveConnections[k]->getProvidesPortName() <<endl;
    }
  }

  saveOutputFile.close();
}

void BuilderWindow::saveAs()
{
  //cerr << "BuilderWindow::saveAs not finished\n";

  QString fn = QFileDialog::getSaveFileName( QString::null, "Network File (*.net)", this );

  if( !fn.isEmpty() ) {
    filename = fn;
    save();
  }
  else {
    statusBar()->message( "Saving aborted", 2000 );
  }
}

void BuilderWindow::load()
{
  std::vector<Module*> grab_latest_Modules;
  std::vector<Module*> ptr_table;
  QString fn = QFileDialog::getOpenFileName( QString::null, "Network File (*.net)", this );
  if(fn.isEmpty()) return;
  filename=fn;
  ifstream is( fn ); 

  int load_Modules_size = 0;
  int load_Connections_size = 0;
  string tmp_moduleName;
  int tmp_moduleName_x;
  int tmp_moduleName_y;

  is >> load_Modules_size >> load_Connections_size;
  cout<<"load_Modules_size"<<load_Modules_size<<endl;
  cout<<"load_Connections_size"<<load_Connections_size<<endl;
  for( int i = 0; i < load_Modules_size; i++ )
  {
    is >> tmp_moduleName >> tmp_moduleName_x >> tmp_moduleName_y;

    sci::cca::ports::BuilderService::pointer builder = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    sci::cca::ComponentID::pointer cid=builder->createInstance(tmp_moduleName, tmp_moduleName, sci::cca::TypeMap::pointer(0));
    SSIDL::array1<std::string> usesPorts=builder->getUsedPortNames(cid);
    SSIDL::array1<std::string> providesPorts=builder->getProvidedPortNames(cid);
    services->releasePort("cca.BuilderService");

    if( tmp_moduleName != "SCIRun.Builder" )
      big_canvas_view->addModule( tmp_moduleName, tmp_moduleName_x, tmp_moduleName_y, usesPorts, providesPorts, cid, false); //fixed position
    
    grab_latest_Modules = big_canvas_view->getModules();
    
    ptr_table.push_back( grab_latest_Modules[grab_latest_Modules.size()-1] );
  }

  for(int i = 0; i < load_Connections_size; i++ ) 
    {
    int iu, ip;
    std::string up, pp;
    
    is>>iu>>up>>ip>>pp;
    
    big_canvas_view->addConnection(ptr_table[iu],up,ptr_table[ip],pp);
    }

  is.close();
  cout<<"Loading is Done"<<endl;
  return;
}

void BuilderWindow::insert()
{
  cerr << "BuilderWindow::insert not finished\n";
}

void BuilderWindow::clear()
{
  cerr << "BuilderWindow::clear(): deleting the following: " << endl;

  // assign modules to local variable
  std::vector<Module*> clearModules = big_canvas_view->getModules();

  for( unsigned int j = 0; j < clearModules.size(); j++ )
  {
    cerr << "modules->getName = " << clearModules[j]->moduleName << endl;
    clearModules[j]->destroy();
  }
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

void BuilderWindow::cluster_add()
{
  ( new QMessageBox())->about( this, "Cluster: Add", "Under Construction\n\nThis dialog will guide\n the user through the steps of adding\na cluster.\n\n" );
}

void BuilderWindow::cluster_remove()
{
  ( new QMessageBox())->about( this, "Cluster: Remove", "Under Construction\n\nThis dialog will guide\n the user through the steps of removing\na cluster.\n\n" );
}

void BuilderWindow::mxn_add()
{
  ( new QMessageBox())->about( this, "MxN: Add", "Under Construction\n\nWhen this menu item is activated, the chosen parallel component will be added to the \ncanvas.  This will occur in the same manner as when other CCA components \nare instantiated.\n\n" );
}

void BuilderWindow::performance_mngr()
{
  ( new QMessageBox())->about( this, "Performance Manager", "Under Construction\n\nThis dialog will display all components active on the canvas and assist the user in setting performance settings for each.\n\n" );
}

void BuilderWindow::performance_tau_add()
{
  ( new QMessageBox())->about( this, "Tau: Add", "Under Construction\n\nWhen this menu item is activated, a tau component will be added to the canvas.  \nThis will occur in the same manner as when other CCA components are \ninstantiated.\n\n" );
}

void BuilderWindow::demos()
{
  ( new QMessageBox())->about( this, "Demos", "CCA Demos\nComing Soon!" );
}

void BuilderWindow::about()
{
  cerr << "BuilderWindow::about not finished\n";
  (new QMessageBox())->about(this, "About", "CCA Builder (SCIRun Implementation)");
}

void BuilderWindow::instantiateComponent(const sci::cca::ComponentClassDescription::pointer& cd)
{
  sci::cca::ports::BuilderService::pointer builder = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
    return;
  }

  TypeMap *tm=new TypeMap;
  tm->putString("LOADER NAME", cd->getLoaderName());
  sci::cca::ComponentID::pointer cid=builder->createInstance(cd->getComponentClassName(), cd->getComponentClassName(), sci::cca::TypeMap::pointer(tm));
  if(cid.isNull()){
    cerr << "instantiateFailed...\n";
    return;
  }
  SSIDL::array1<std::string> usesPorts=builder->getUsedPortNames(cid);
  SSIDL::array1<std::string> providesPorts=builder->getProvidedPortNames(cid);

  services->releasePort("cca.BuilderService");
  if(cd->getComponentClassName()!="SCIRun.Builder"){

    int x = 20;
    int y = 20;
    
    big_canvas_view->addModule(cd->getComponentClassName(), x, y, usesPorts, providesPorts, cid, true); //reposition module
  }
}

void BuilderWindow::componentActivity(const sci::cca::ports::ComponentEvent::pointer& e)
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

  QCanvasRectangle* viewableRect = new QCanvasRectangle(   int(big_canvas_view->contentsX()/scaleH), 
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


void BuilderWindow::addCluster()
{

  sci::cca::ports::BuilderService::pointer builder = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
  }

  ////////////////////////
  //Assume a QT Diaglog will return
  string loaderName="qwerty";
  string domainName="qwerty.sci.utah.edu";
  string login="kzhang";
  string loaderPath="mpirun -np 2 ploader";
  string password="****"; //not used;


  builder->addLoader(loaderName, login, domainName, loaderPath);


  services->releasePort("cca.BuilderService");
  
  //buildPackageMenus(loaderName);

  /*
  slaveServer_impl* ss=new slaveServer_impl;
  cerr << "Waiting for slave connections...\n";
  cerr << ss->getURL().getString() << '\n';
  
  //Wait until we have some slave available
  ss->d_slave_sema.down();
  
  slaveClient::pointer sc = ss->rr->getPtrToAll();
  
  //Set up server's requirement of the distribution array
  Index** dr = new Index* [1];
  dr[0] = new Index(0,ss->rr->getSize(),1);
  MxNArrayRep* arrr = new MxNArrayRep(1,dr);
  sc->setCallerDistribution("dURL",arrr);
  */  
  /******** Simulate invocations:*/
  /*
    string arg1 = "bb";
  SSIDL::array1<std::string> urls;
  sc->instantiate("aa",arg1,urls);
  for(unsigned int i=0; i<urls.size(); i++)
    cout << "URL = " << urls[i] << "\n";
  */
}

void BuilderWindow::rmCluster()
{
  sci::cca::ports::BuilderService::pointer builder = pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
  }
  builder->removeLoader("buzz");
  services->releasePort("cca.BuilderService");
    
}

void BuilderWindow::refresh(){
  buildPackageMenus();
}

void BuilderWindow::insertHelpMenu()
{
  static bool firstTime=true;
  static int id;
  if(firstTime){
    firstTime=false;
  }
  else{
    menuBar()->removeItem(id);
  }
  //menuBar()->insertSeparator();
  QPopupMenu * help = new QPopupMenu( this );
  help->insertTearOffHandle();
  id=menuBar()->insertItem( "&Help", help );
  help->insertItem( "&About", this, SLOT(about()), Key_F1 );
  help->insertSeparator();
  help->insertItem( "What's &This", this, SLOT(whatsThis()),
		    SHIFT+Key_F1 );
}

