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
 *  Hello.c:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#include <CCA/Components/Hello/Hello.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

#include <qapplication.h>
#include <qpushbutton.h>
#include <qmessagebox.h>



using namespace std;
using namespace SCIRun;

extern "C" gov::cca::Component::pointer make_SCIRun_Hello()
{
  return gov::cca::Component::pointer(new Hello());
}


Hello::Hello()
{

}

Hello::~Hello()
{
  cerr << "called ~Hello()\n";
}

void Hello::setServices(const gov::cca::Services::pointer& svc)
{
  //cerr<<"Hello::serService is  called#################\n";

  services=svc;
  //register provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer p(&port0);
  svc->addProvidesPort(p,"ui","UIPort", props);
  svc->addProvidesPort(p,"UIPort1","UIPort", props);
  svc->addProvidesPort(p,"UIPort2","UIPort", props);

  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  svc->registerUsesPort("UU#0", "UIPort", props);
  svc->registerUsesPort("UU#1", "UIPort", props);
  svc->registerUsesPort("UU#2", "UIPort", props);


  QApplication* app = QtUtils::getApplication();
  

#ifdef QT_THREAD_SUPPORT
  app->lock();
#endif
  //int argc=0;
  //char* argv[]={""};
  //QApplication a( argc, argv );
  //QPushButton hello( "Hello world!", 0 );
  //hello.resize( 100, 30 );

  //app->setMainWidget( &hello );
  //hello.show();
  // a.exec();
  //builder = new BuilderWindow(services);
  //builder->addReference();
  //builder->show(); 
#ifdef QT_THREAD_SUPPORT 
	app->unlock(); 
#endif

}

void myUIPort::ui() 
{
  QMessageBox::warning(0, "myUIPort", "Hello!");
  //cerr<<"$$$ui() is not implemented."<<endl;
}

