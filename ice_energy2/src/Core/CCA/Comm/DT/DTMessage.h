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


#ifndef CORE_CCA_COMM_DT_DTMESSAGE_H
#define CORE_CCA_COMM_DT_DTMESSAGE_H

#include <Core/CCA/Comm/DT/DTAddress.h>
#include <Core/CCA/Comm/DT/DTMessageTag.h>
#include <iostream>
#include <string.h>
namespace SCIRun {

  class DTPoint;
  class DTMessage{
  public:
    friend class DataTransmitter;
    //The message being sent has the following structure:
    //DTMessage | buf

    char *buf;
    int length;
    bool autofree;
    DTPoint *recver;  //recver sp/ep  
    DTPoint *sender;  //sender sp/ep   
    DTAddress fr_addr;  //filled by sender
    DTAddress to_addr;  //filled by recver
    DTMessageTag tag; 
    ~DTMessage();

    void display();
    /*
    DTPacketID getPacketID(){
      return DTPacketID(getDestination(), fr_addr);
    }
    */
    DTDestination getDestination(){
      return DTDestination(recver,to_addr);
    }
  private:
    int offset; //used by DataTransmitter only. It is reset in the DataTransmitter.
  };

}// namespace SCIRun
#endif
