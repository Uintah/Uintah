/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_BufferInfo_H
#define UINTAH_HOMEBREW_BufferInfo_H

#include <sci_defs/mpi_defs.h> // For mpi.h

#include <vector>

namespace Uintah {

class RefCounted;
class ProcessorGroup;

class AfterCommunicationHandler {
  public:
    virtual ~AfterCommunicationHandler()
    {
    }

    virtual void finishedCommunication(const ProcessorGroup*,
                                       MPI_Status& status) = 0;
};

class Sendlist : public AfterCommunicationHandler {

  public:
    Sendlist( Sendlist* next, RefCounted* obj ) : next(next), obj(obj) {}

    virtual ~Sendlist();

    Sendlist*   next;
    RefCounted* obj;

    // Sendlist is to be an AfterCommuncationHandler object for the
    // MPI_CommunicationRecord template in MPIScheduler.cc.  The only task
    // it needs to do to handle finished send requests is simply get deleted.
    virtual void finishedCommunication( const ProcessorGroup *, MPI_Status & status ) {}

};

class BufferInfo {

  public:
    BufferInfo();

    virtual ~BufferInfo();

    unsigned int count() const;

    void get_type(void*&,
                  int&,
                  MPI_Datatype&);

    void add( void*          startbuf,
              int            count,
              MPI_Datatype   datatype,
              bool           free_datatype );

    void addSendlist( RefCounted* );

    Sendlist* takeSendlist();

  private:
    BufferInfo(const BufferInfo&);
    BufferInfo& operator=(const BufferInfo&);

  protected:
    Sendlist*                   d_sendlist;
    std::vector<void*>          d_startbufs;
    std::vector<int>            d_counts;
    std::vector<MPI_Datatype>   d_datatypes;
    std::vector<bool>           d_free_datatypes;

    void*          buf;
    int            cnt;
    MPI_Datatype   datatype;

    bool d_free_datatype;
    bool d_have_datatype;
};
}

#endif

