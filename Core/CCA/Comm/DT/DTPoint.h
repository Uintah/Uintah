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


#ifndef CORE_CCA_COMM_DT_DTPOINT_H
#define CORE_CCA_COMM_DT_DTPOINT_H

#include <Core/CCA/Comm/DT/DTMessageTag.h>

namespace SCIRun {
  class DTMessage;
  class DTMessageTag;
  class DataTransmitter;

  class DTPoint{
  public:
    friend class DataTransmitter;
    friend class SocketEpChannel;
    void *object;
    DTPoint(DataTransmitter *dt);
    ~DTPoint();
    
    ///////////
    //This method blocks until a message is available in the 
    //DataTransmitter and then return this message.
    DTMessage* getMessage(const DTMessageTag& tag);

    //This method blocks until a message with the default tag is available in the 
    //DataTransmitter and then return this message.
    DTMessage* getMsg();
    
    ///////////
    //Put msg into the sending message queue.
    //the sender field and tag are filled by this method.
    DTMessageTag putInitialMessage(DTMessage *msg);

    ///////////
    //Put msg into the sending message queue.
    //the sender field is filled by this method.
    void putReplyMessage(DTMessage *msg);

    ///////////
    //Put msg into the sending message queue with a default tag.
    //the sender field is filled by this method.
    void putMsg(DTMessage *msg);

  private:
    //callback function
    void (*service)(DTMessage *msg);
    DataTransmitter *dt;
  };

}//namespace SCIRun

#endif
