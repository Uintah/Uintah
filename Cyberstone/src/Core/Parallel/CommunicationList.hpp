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

#ifndef CCA_COMPONENTS_SCHEDULERS_COMMUNICATIONLIST_H
#define CCA_COMPONENTS_SCHEDULERS_COMMUNICATIONLIST_H

#include <CCA/Components/Schedulers/BatchReceiveHandler.h>

#include <Core/Lockfree/Lockfree_Pool.hpp>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Parallel/UintahMPI.h>

#include <utility>

namespace Uintah {

class CommHandle
{

public:

  CommHandle()                                = default;
  CommHandle( const CommHandle & )            = default;
  CommHandle& operator=( const CommHandle & ) = default;
  CommHandle( CommHandle && )                 = default;
  CommHandle& operator=( CommHandle && )      = default;

  virtual ~CommHandle(){}

  virtual void finishedCommunication ( const ProcessorGroup * pg
                                     , MPI_Status           & status )
  {
    // do nothing
  }
};

class RecvHandle : public CommHandle
{

public:

  RecvHandle(  PackBufferInfo      * unpack_handler
             , BatchReceiveHandler * batch_handler
             )
    : m_unpack_handler{unpack_handler}
    , m_batch_handler{batch_handler}
   {}

   ~RecvHandle()
   {
     if (m_unpack_handler) {
       delete m_unpack_handler;
     }

     if (m_batch_handler) {
       delete m_batch_handler;
     }
   }

   void finishedCommunication (  const ProcessorGroup * pg
                               ,       MPI_Status     & status
                              )
   {
     // The order is important: it should unpack before informing the
     // DependencyBatch that the data has been received.
     if (m_unpack_handler) {
       m_unpack_handler->finishedCommunication(pg, status);
     }

     if (m_batch_handler) {
       m_batch_handler->finishedCommunication(pg);
     }
   }


 private:

   PackBufferInfo      * m_unpack_handler{nullptr};
   BatchReceiveHandler * m_batch_handler{nullptr};

};

class SendHandle : public CommHandle
{

public:

  // Ugly, nasty hack, but honors original nasty contract to take ownership of the SendList
  SendHandle( Sendlist* send_list )
    : m_send_list{send_list}
   {}

   // Because we took ownership of the send list, we delete it. we should use std::unique_ptr
  ~SendHandle()
  {
    if (m_send_list) {
      delete m_send_list;
    }
  }

 private:

   Sendlist   * m_send_list{nullptr};

};


class CommRequest
{

public:

  CommRequest(CommHandle* handle)
    : m_handle{handle}
  {}

  CommRequest(std::unique_ptr<CommHandle>&& handle)
    : m_handle{std::move(handle)}
  {}

  CommRequest(std::unique_ptr<CommHandle>& handle)
    : m_handle{std::move(handle)}
  {}

  CommRequest() = default;

  CommRequest( const CommRequest & )            = default;
  CommRequest& operator=( const CommRequest & ) = default;
  CommRequest( CommRequest && )                 = default;
  CommRequest& operator=( CommRequest && )      = default;

  virtual ~CommRequest(){}

  virtual void finishedCommunication(){}

  MPI_Request* request() const
  {
    return const_cast<MPI_Request*>(&m_request);
  }

  bool test() const
  {
    int flag;
    Uintah::MPI::Test(request(), &flag, MPI_STATUS_IGNORE);
    return flag;
  }

  bool wait() const
  {
    Uintah::MPI::Wait(request(), MPI_STATUS_IGNORE);
    return true;
  }

  void finishedCommunication ( const ProcessorGroup * pg
                             , MPI_Status           & status
                             )
  {
    if (m_handle) {
      m_handle->finishedCommunication(pg, status);
    }
  }


private:

  mutable MPI_Request          m_request{};
  std::unique_ptr<CommHandle>  m_handle{};

};


using CommRequestPool = Lockfree::Pool< CommRequest
                                      , uint64_t
                                      , 1
                                      , std::allocator
                                      >;

} // end namespace Uintah

#endif // end CCA_COMPONENTS_SCHEDULERS_COMMUNICATIONLIST_H
