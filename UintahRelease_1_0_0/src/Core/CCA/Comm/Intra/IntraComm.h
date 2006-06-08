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
    virtual ~IntraComm(){};

    ////////////////
    // Retrieved this process' rank
    virtual int get_rank() = 0;

    ////////////////
    // Retrieved this process' size    
    virtual int get_size() = 0;

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

    /////////////////
    // Send an asynchronous  message to a parallel process
    // specified by rank number. Returns success or failure.
    virtual int async_send(int rank, char* bytestream, int length) = 0;
    
    /////////////////
    // Receive an asynchronous message from a parallel process
    // specified by rank number. Returns success or failure..
    virtual int async_receive(int rank, char* bytestream, int length) = 0;
   
    ////////////////
    // A barrier primitive 
    virtual void barrier() = 0;
  };
}

#endif
