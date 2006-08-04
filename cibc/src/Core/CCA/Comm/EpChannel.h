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



#ifndef EP_CHANNEL_INTERFACE_H
#define EP_CHANNEL_INTERFACE_H 

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

  class Message;
  class SpChannel;
  class TypeInfo;

  /**************************************
 
  CLASS
  EpChannel
   
  DESCRIPTION
     The base class for all communication-specific endpoint 
     channel abstractions. An endpoint channel is a server
     channel in the sense that waits for one (or more depending
     on implementation) clients to bind to it. It also contains
     handler functions which are invoked when a message bound 
     for them is recieved.

  ****************************************/

  class EpChannel {
  public:

    //Handler function type
    typedef void (*HPF)(Message* ); 

    //Initializing methods used to establish the
    //connection. These are called from the PIDL.
    virtual void openConnection() = 0;
    virtual void activateConnection(void* obj) = 0;
    virtual void closeConnection() = 0;
    virtual std::string getUrl() = 0; 
  
    ///////////////
    // Retrieves a Message associated with this particular object.
    virtual Message* getMessage() = 0;

    //////////////
    // Used in order to directly bind a SpChannel to
    // this EpChannel.
    virtual void bind(SpChannel* spchan) = 0;

    //////////////
    // Functions used to establish the handler functions
    virtual void allocateHandlerTable(int size)=0;
    virtual void registerHandler(int num, void* handle)=0;
  
    //////////
    // A pointer to the type information.
    const TypeInfo* d_typeinfo;
  };
}

#endif
