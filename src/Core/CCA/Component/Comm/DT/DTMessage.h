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
 *  DTMessage.h defines the message structure used in the data transmitter
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMPONENT_COMM_DT_DTMESSAGE_H
#define CORE_CCA_COMPONENT_COMM_DT_DTMESSAGE_H

#include <Core/CCA/Component/Comm/DT/DTAddress.h>
#include <iostream>
#include <string.h>
namespace SCIRun {

  class DTPoint;
  class DTMessage{
  public:
    //The message being sent has the following structure:
    //DTMessage | buf

    char *buf;
    int length;
    bool autofree;
    DTPoint *recver;  //recver sp/ep  
    DTPoint *sender;  //sender sp/ep   
    DTAddress fr_addr;  //filled by sender
    DTAddress to_addr;  //filled by recver

    ~DTMessage();
    void display();
  };

}// namespace SCIRun
#endif
