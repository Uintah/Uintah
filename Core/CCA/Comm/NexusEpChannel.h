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



#ifndef NEXUS_EP_CHANNEL_H
#define NEXUS_EP_CHANNEL_H 

#include <Core/CCA/Comm/EpChannel.h>
#include <stdio.h>
#undef IOV_MAX
#include <globus_nexus.h>

namespace SCIRun {
  class Object;
  /**************************************
 
  CLASS
     NexusEpChannel
   
  DESCRIPTION
     A Globus-Nexus implementation of the EpChannel
     abstract interface. This implementation routs
     all incoming nexus messages to its appropriate
     handler via the unknown_handler. 

  SEE ALSO
     EpChannel.h

  ****************************************/

  class NexusEpChannel : public EpChannel {
  public:

    NexusEpChannel();
    virtual ~NexusEpChannel();
    void openConnection();
    void activateConnection(void *);
    void closeConnection();
    std::string getUrl(); 
    Message* getMessage();
    void allocateHandlerTable(int size);
    void registerHandler(int num, void* handle);
    int approve(globus_nexus_startpoint_t* sp, Object* obj);
    void bind(SpChannel* spchan);

    ////////////
    // The table of handlers from the sidl generated file
    // used to relay a message to them 
    HPF* handler_table;
    int table_size;

    /////////////
    // Buffer in which we store the message we recieve
    globus_nexus_buffer_t msgbuffer;

  private:

    //////////
    // The endpoint associated with this object.
    globus_nexus_endpoint_t d_endpoint;

    void printDebug(const std::string& );
  };
}

#endif
