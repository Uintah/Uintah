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
 *  DTPoint.cc: Data Communication Point (Sender/Receiver)
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */
#include <Core/Thread/Semaphore.h>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/CCA/Comm/DT/DTMessage.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>


using namespace SCIRun;

DTPoint::DTPoint(DataTransmitter *dt){
  this->dt=dt;
  object=NULL;
  sema=new Semaphore("DTPoint semaphore", 0);
  dt->registerPoint(this);
  service=NULL;
}

DTPoint::~DTPoint(){
  delete sema;
  dt->unregisterPoint(this);
}

DTMessage *
DTPoint::getMessage(){
  sema->down();
  return dt->getMessage(this);
}

void 
DTPoint::putMessage(DTMessage *msg){
  msg->sender=this;
  dt->putMessage(msg);
}







