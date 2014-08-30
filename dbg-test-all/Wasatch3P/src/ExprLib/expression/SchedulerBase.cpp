/*
 * SchedulerBase.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: "James C. Sutherland"
 *
 * Copyright (c) 2012 The University of Utah
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


#include "SchedulerBase.h"

namespace Expr{

  void Scheduler::run_pollers( VertexProperty& target,
                               FieldManagerList* const fml )
  {
    if( pollers_.size() == 0 && nonBlockingPollers_.size() == 0 ) return;

#   ifdef ENABLE_THREADS
    // note that this mutex type is the same as the one in block_pollers().
    // This ensures that neither method can be executed by multiple threads
    // concurrently, which is important.
    //
    // jcs Potential problem: since we are protecting this with a mutex but this
    //     results in a callback, could we end up blocking numerous threads here?
    ExecMutex<100> lock; // only one thread can execute pollers at a time.
#   endif

    if( target.poller.get() ){
      // jcs shouldn't this be the same as target.poller ???
      PollerPtr p = *pollers_.find( target.poller );
      p->activate_all();
      p->set_blocking( false );
    }
    if( target.nonBlockPoller.get() ){
      NonBlockingPollerPtr p = *nonBlockingPollers_.find( target.nonBlockPoller );
      p->activate_all();
      p->set_blocking( false );
    }

    // Run the blocking pollers.  If ready, then these force a call-back on the
    // vertex they are associated with.
    BOOST_FOREACH( PollerPtr poller, pollers_ ){
      if( poller->is_active() ){
        if( poller->run( fml ) ){
          // fire a call-back on the vertex associated with this poller.
          VertexProperty* vp = poller->get_vertex_property();
          (*vp->execSignalCallback)(vp->self_);
        }
      }
    }

    // Run the non-blocking pollers. There are no call-backs associated with this.
    if( target.nonBlockPoller.get() ){
      target.nonBlockPoller->activate_all();
    }
    BOOST_FOREACH( NonBlockingPollerPtr poller, nonBlockingPollers_ ){
      poller->run();
    }
  }

  void Scheduler::block_pollers()
  {
    if( pollers_.size() == 0 && nonBlockingPollers_.size() == 0 ) return;

    // ensure that all pollers have completed.  If not, force them to finish.
#   ifdef ENABLE_THREADS
    ExecMutex<100> lock; // only one thread can execute pollers at a time.
#   endif
    bool anyActive = false;
    do{
      anyActive = false;
      BOOST_FOREACH( PollerPtr poller, pollers_ ){
        if( poller->is_active() ){
          poller->set_blocking( true );
          anyActive = true;
          VertexProperty* const vp = poller->get_vertex_property();
          FieldManagerList* const fml = extract_field_manager_list( this->fmls_, vp->fmlid );
          const bool done = poller->run(fml);
          if( done ){
            // fire a call-back on the vertex associated with this poller and then
            // remove the poller from the list of pollers since it is done.
            (*vp->execSignalCallback)(vp->self_);
          }
        }
      }
      BOOST_FOREACH( NonBlockingPollerPtr poller, nonBlockingPollers_){
        if( poller->is_active() ){
          poller->set_blocking( true );
          anyActive = true;
          poller->run();
        }
      }
    } while( anyActive );
  }

  void Scheduler::set_poller( PollerPtr p )
  {
    pollers_.insert( p );
  }

  void Scheduler::set_nonblocking_poller( NonBlockingPollerPtr p ){
    nonBlockingPollers_.insert( p );
  }


} // namespace Expr
