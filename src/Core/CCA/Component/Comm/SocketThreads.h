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
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#include <CCA/Components/Builder/QtUtils.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/CCA/Component/Comm/SocketEpChannel.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

static Semaphore* startup;

class SocketAcceptThread : public Runnable {
public:
  SocketAcceptThread(SocketEpChannel *sep) {
    this->sep=sep;
  }
  ~SocketAcceptThread() {}
  void run();
 private:
  SocketEpChannel *sep;
  
};

void 
SocketAcceptThread::run()
{
  sep->runAccept();
}

class SocketHandlerThread : public Runnable {
public:
  SocketHandlerThread(SocketEpChannel *sep) {this->sep=sep;}
  ~SocketHandlerThread() {}
  void run();
 private:
  SocketEpChannel *sep;
};

void 
SocketHandlerThread::run()
{
  cerr<<"SocketAcceptThread is running\n";
}
