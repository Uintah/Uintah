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


#ifndef INTRACOMM_INTERFACE_H
#define INTRACOMM_INTERFACE_H 

namespace SCIRun {
  /**************************************
  CLASS
     IntraComm 
   
  DESCRIPTION
     An interface which provides methods to 
     perform communication within a component. 
     This is in the case of parallel components
     and communication among different parallel
     processes. 
  ****************************************/

  class IntraComm {
  public:

    /////////////////
    // Send a message to a parallel process
    // specified by rank number. Returns success or failure.
    virtual int send(int rank, char* bytestream, int length) = 0;

    /////////////////
    // Receive a message from a parallel process
    // specified by rank number. Returns success or failure..
    virtual int receive(int rank, char* bytestream, int length) = 0;
   
    ////////////////
    // Broadcast a message from root to all other parallel processes
    // Returns success or failure..
    virtual int broadcast(int root, char* bytestream, int length) = 0;
  };
}

#endif
