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
 *  Builder.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <CCA/Components/Builder/Builder.h>
#include <CCA/Components/Builder/BuilderWindow.h>
#include <CCA/Components/Builder/QtUtils.h>
#include <Core/CCA/spec/cca_sidl.h>

#include <qapplication.h>
#include <qpushbutton.h>

#include <iostream>

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Builder()
{
  return sci::cca::Component::pointer(new Builder());
}

Builder::Builder()
{
  cerr<<"Builder()"<<endl;
}

Builder::~Builder()
{
  cerr<<"~Builder()"<<endl;
}

void Builder::setServices(const sci::cca::Services::pointer& services)
{
  cerr<<"Builder::setServices"<<endl;
  builderPort.setServices(services);
  sci::cca::TypeMap::pointer props = services->createTypeMap();
  myBuilderPort::pointer bp(&builderPort);
  services->addProvidesPort(bp,"builderPort","sci.cca.ports.BuilderPort", props);
  services->registerUsesPort("builder", "sci.cca.ports.BuilderPort", props);


  sci::cca::ports::BuilderService::pointer builder 
    = pidl_cast<sci::cca::ports::BuilderService::pointer>
    (services->getPort("cca.BuilderService"));
  if(builder.isNull()){
    cerr << "Fatal Error: Cannot find builder service\n";
    return;
  } 
  //do not delelet the following line	
  //builder->registerServices(services);
    
  services->releasePort("cca.BuilderService"); 
      
}

void myBuilderPort::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  cerr<<"BuilderPort::setServices"<<endl;

  QApplication* app = QtUtils::getApplication();
#ifdef QT_THREAD_SUPPORT
  app->lock();
#endif
    cerr<<"creating builderwindow"<<endl;
  builder = new BuilderWindow(services);
  cerr<<"builderwindow created"<<endl;
  builder->addReference();
  builder->show();
  cerr<<"Builderwindow is shown"<<endl;
#ifdef QT_THREAD_SUPPORT
  app->unlock();
#endif
}

void myBuilderPort::buildRemotePackageMenus(const  sci::cca::ports::ComponentRepository::pointer &reg,
				    const std::string &frameworkURL)
{
  builder->buildRemotePackageMenus(reg, frameworkURL);
}





