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
