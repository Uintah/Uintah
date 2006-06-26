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
 *  SocketEpChannel.h: Socket implemenation of Ep Channel
 *
 *  Written by:
 *   Kosta Damevski and Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMM_SOCKETEPCHANNEL_H
#define CORE_CCA_COMM_SOCKETEPCHANNEL_H

#include <Core/CCA/Comm/EpChannel.h>

namespace SCIRun {
  class SocketMessage;
  class Message;
  class SpChannel;
  class Thread;
  class DTPoint;
  class DTMessage;
  class SocketEpChannel : public EpChannel {
    friend class SocketMessage;
    friend class SocketThread;
    friend class PRMI;
  public:

    SocketEpChannel();
    virtual ~SocketEpChannel();
    void openConnection();
    void activateConnection(void* obj);
    void closeConnection();
    std::string getUrl(); 
    Message* getMessage();
    void allocateHandlerTable(int size);
    void registerHandler(int num, void* handle);
    void bind(SpChannel* spchan);
    int getTableSize();
    DTPoint *getEP();

    static const int ADD_REFERENCE=-101;
    static const int DEL_REFERENCE=-102;
    static const int MPI_LOCKSERVICE=-103;
    static const int MPI_ORDERSERVICE=-104;
  private:
    DTPoint *ep;

    ////////////
    // The table of handlers from the sidl generated file
    // used to relay a message to them
    HPF* handler_table;

    /////////////
    // Handler table size
    int table_size;

  };
}// namespace SCIRun

#endif
