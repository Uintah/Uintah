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
 *  DTPoint.h: Data Communication Point (Sender/Receiver)
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMPONENT_COMM_DT_DTPOINT_H
#define CORE_CCA_COMPONENT_COMM_DT_DTPOINT_H

namespace SCIRun {
  class Semaphore;
  class DTMessage;
  class DTPoint{
  public:
    friend class DataTransmitter;
    void *object;
    DTPoint();
    ~DTPoint();
    
    ///////////
    //This method blocks until a message is available in the 
    //DataTransmitter and then return this message.
    DTMessage* getMessage();
    
    ///////////
    //Put msg into the sending message queue.
    //the sender field is automaticly filled.
    void putMessage(DTMessage *msg);

    //callback function
    void (*service)(DTMessage *msg);

  private:
    Semaphore *sema;
  };

}//namespace SCIRun

#endif
