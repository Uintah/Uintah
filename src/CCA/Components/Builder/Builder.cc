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
#include <iostream>
using namespace std;
using namespace SCIRun;

#include <qapplication.h>
#include <qpushbutton.h>

extern "C" gov::cca::Component make_SCIRun_Builder()
{
  return new Builder();
}

Builder::Builder()
{
}

Builder::~Builder()
{
  cerr << "called ~Builder()\n";
}

void Builder::setServices(const gov::cca::Services& svc)
{
  services=svc;

  QApplication* app = QtUtils::getApplication();
#ifdef QT_THREAD_SUPPORT
  app->lock();
#endif
  builder = new BuilderWindow(services);
  builder->_addReference();
  builder->show();
#ifdef QT_THREAD_SUPPORT
  app->unlock();
#endif
}
