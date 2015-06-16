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

#ifndef CORE_PARALLEL_COMMUNICATIONLIST_H
#define CORE_PARALLEL_COMMUNICATIONLIST_H

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

#include <CCA/Components/Schedulers/BatchReceiveHandler.h>

#include <Core/Malloc/AllocatorTags.hpp>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Parallel/PackBufferInfo.h>

#include <utility>

namespace Uintah {

class RecvHandle
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
                                 , MPI_Status           & status )
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

     PackBufferInfo      * m_unpack_handler;
     BatchReceiveHandler * m_batch_handler;

};

class SendHandle
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

     void finishedCommunication (  const ProcessorGroup * /* pg */
                                 , MPI_Status           & /* status */ )
     {}


   private:

     Sendlist   * m_send_list;

};

template<typename CommHandle>
class CommNode
{
  public:

    template <typename... Args>
    CommNode (Args... args)
      : m_request{}
      , m_comm_handle{std::forward<Args>(args)...}
    {}

    void finishedCommunication (  const ProcessorGroup * pg
                                , MPI_Status           & status )
    {
      m_comm_handle.finishedCommunication(pg, status);
    }

    MPI_Request* request() const
    {
      return const_cast<MPI_Request*>(&m_request);
    }

    bool test() const
    {
      int flag;
      MPI_SAFE_CALL(MPI_Test(request(), &flag, MPI_STATUS_IGNORE));
      return flag;
    }

    bool wait() const
    {
      MPI_SAFE_CALL(MPI_Wait(request(), MPI_STATUS_IGNORE));
      return true;
    }

  private:

    mutable MPI_Request  m_request;
    CommHandle           m_comm_handle;

};

using SendCommNode = CommNode<SendHandle>;
using RecvCommNode = CommNode<RecvHandle>;


template < typename T >
using ListPoolAllocator = Lockfree::PoolAllocator<   T
                                                   , Uintah::MMapAllocator
                                                 >;

template < typename T >
using TrackingListPoolAllocator = Lockfree::TrackingAllocator<   T
                                                               , ListPoolAllocator
                                                               , CommListTag
                                                               , false    // don't track globally
                                                             >;

using SendCommList = Lockfree::UnorderedList<  SendCommNode
                                             , Lockfree::EXCLUSIVE_INSTANCE // usage model
                                             , TrackingListPoolAllocator    // allocator
                                            >;


using RecvCommList = Lockfree::UnorderedList<  RecvCommNode
                                             , Lockfree::EXCLUSIVE_INSTANCE // usage model
                                             , TrackingListPoolAllocator    // allocator
                                            >;

} // end namespace Uintah

#endif // end CORE_PARALLEL_COMMUNICATIONLIST_H
