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
 *   DTThread.cc: Threads used by the DataTransmitter
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#include <Core/CCA/Comm/DT/DTThread.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>

using namespace SCIRun;
  
DTThread::DTThread(DataTransmitter *dt, int id){
  this->dt=dt;
  this->id=id;
}

void 
DTThread::run()
{
  switch(id){
  case 1:
    dt->runSendingThread();
    break;
  case 2:
    dt->runRecvingThread();
    break;
  }
}
