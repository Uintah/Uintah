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
 *  QtUtils.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <CCA/Components/Builder/QtUtils.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

#include <qapplication.h>

static QApplication* theApp;
static Semaphore* startup;

class QtThread : public Runnable {
public:
  QtThread() {}
  ~QtThread() {}
  void run();
};

QApplication* QtUtils::getApplication()
{
  if(!theApp){
    startup=new Semaphore("Qt Thread startup wait", 0);
    Thread* t = new Thread(new QtThread(), "SCIRun Builder", 0, Thread::NotActivated);
    t->setStackSize(8*256*1024);
    t->activate(false);
    t->detach();
    startup->down();
  }
  return theApp;
}

void QtThread::run()
{
  cerr<<"******************QtThread::run()**********************\n"; 
  int argc=1;
  char* argv[4];
  argv[0]="SCIRun2";
  argv[1]="-im";
  argv[2]="";
  argv[3]=0;
  theApp = new QApplication(argc, argv);
  startup->up();
  theApp->exec();
}
