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





