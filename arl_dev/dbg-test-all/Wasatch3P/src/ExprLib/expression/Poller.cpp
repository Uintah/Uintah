/*
 * Poller.cpp
 *
 *  Created on: Nov 5, 2012
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


#include "Poller.h"
#include <expression/VertexProperty.h>
#include <expression/FieldManagerList.h>

#include <boost/foreach.hpp>

namespace Expr{

  PollWorker::PollWorker()
  {
    isActive_  = true;
    isBlocked_ = false;
  }

  PollWorker::~PollWorker(){}

  //=================================================================

  Poller::Poller( const Tag& monitoredField )
  : fieldTag_( monitoredField )
  {
    vp_      = NULL;
    nActive_ = 0;
    doBlock_ = false;
  }

  //-----------------------------------------------------------------

  Poller::~Poller(){}

  //-----------------------------------------------------------------

  void Poller::add_new( PollWorkerPtr worker )
  {
    workers_.push_back( worker );
    ++nActive_;
  }

  //-----------------------------------------------------------------

  bool Poller::run( FieldManagerList* fml )
  {
    if( nActive_==0 ) return true;  // short-circuit

    BOOST_FOREACH( PollWorkerPtr worker, workers_ ){
      if( worker->is_active() ){
        worker->set_blocking( doBlock_ );
        if( (*worker)(fml) ){
          worker->deactivate();
          --nActive_;
        }
      }
    }

    return nActive_ == 0;
  }

  //-----------------------------------------------------------------

  void Poller::activate_all()
  {
    BOOST_FOREACH( PollWorkerPtr worker, workers_ ){
      worker->activate();
    }
    nActive_ = workers_.size();
  }

  //-----------------------------------------------------------------

  void Poller::deactivate_all()
  {
    BOOST_FOREACH( PollWorkerPtr worker, workers_ ){
      worker->deactivate();
    }
    nActive_ = 0;
  }

  //-----------------------------------------------------------------

  bool Poller::is_active() const
  {
    return nActive_ > 0;
  }

  //-----------------------------------------------------------------

  void Poller::set_blocking( const bool block )
  {
    doBlock_ = block;
  }

  //-----------------------------------------------------------------

  void Poller::set_vertex_property( VertexProperty* vp )
  {
    vp_ = vp;
  }

  //=================================================================

  NonBlockingPoller::NonBlockingPoller( const Tag& tag )
  : fieldTag_( tag ),
    nActive_( 0 ),
    doBlock_( false )
  {}

  //-----------------------------------------------------------------

  NonBlockingPoller::~NonBlockingPoller(){}

  //-----------------------------------------------------------------

  void NonBlockingPoller::add_new( PollWorkerPtr worker )
  {
    workers_.push_back( worker );
    ++nActive_;
  }

  //-----------------------------------------------------------------

  void NonBlockingPoller::activate_all()
  {
    BOOST_FOREACH( PollWorkerPtr p, workers_ ){
      p->activate();
    }
    nActive_ = workers_.size();
  }

  //-----------------------------------------------------------------

  void NonBlockingPoller::deactivate_all()
  {
    BOOST_FOREACH( PollWorkerPtr p, workers_ ){
      p->deactivate();
    }
    nActive_ = 0;
  }

  //-----------------------------------------------------------------

  bool NonBlockingPoller::is_active() const
  {
    return nActive_ > 0;
  }

  //-----------------------------------------------------------------

  void NonBlockingPoller::set_blocking( const bool block )
  {
    doBlock_ = block;
  }

  //-----------------------------------------------------------------

  bool NonBlockingPoller::run()
  {
    BOOST_FOREACH( PollWorkerPtr worker, workers_ ){
      if( worker->is_active() ){
        worker->set_blocking( doBlock_ );
        if( (*worker)() ){
          worker->deactivate();
          --nActive_;
        }
      }
    }
    return nActive_ == 0;
  }

  //-----------------------------------------------------------------

} // namespace Expr
