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



#ifndef SP_CHANNEL_INTERFACE_H
#define SP_CHANNEL_INTERFACE_H 

namespace SCIRun {
  class Message;
  class URL;

  /**************************************
 
  CLASS
     SpChannel
   
  DESCRIPTION
     The base class for all communication-specific startpoint 
     channel abstractions. A startpoint channel is a client
     channel in the sense that it binds to a particular server
     based on the server's url. The channel abstraction itself
     is meant to establish the connection and provide methods
     in relation to it. The communication sends and recieves are
     performed by the message class.

  ****************************************/

  class SpChannel {
  public:

    virtual ~SpChannel(){};

    /////////////////
    // Methods to establish communication
    virtual void openConnection(const URL& ) = 0;
    virtual void closeConnection() = 0;
  
    /////////////////
    // Creates a message associated with this communication
    // channel. The message can then be used to perform
    // sends/recieves. There is only one message corresponding
    // to each SpChannel.
    virtual Message* getMessage() = 0;

    ////////////////
    // Abstract Factory method meant to create a copy
    // from an instance of the class down the class
    // hierarchy
    virtual SpChannel* SPFactory(bool deep) = 0;
  };
}

#endif
