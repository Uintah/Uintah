/*
 * VertexProperty.cpp
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

#include "VertexProperty.h"
#include <boost/foreach.hpp>

namespace Expr{

  VertexProperty::
  VertexProperty( const int _ix, ExpressionID _id, ExpressionBase* _expr )
  : priority(0), index(_ix), visited(0),
    nparents(0), nremaining(0),
    nconsumers(0), ncremaining(0),
    deviceIndex_(CPU_INDEX), // abhi : consider ripping it
    chainTail_(true),
    chainID_(-1),
    nodeMemoryBound_(0),
    id(_id),
    expr(_expr),
    execTarget(CPU_INDEX),
    mm(MEM_EXTERNAL),
    fmlid( _expr->field_manager_list_id() ),
    self_(NULL),
    execSignalCallback(new Signal()),
    color(boost::white_color)
  {
    set_is_persistent(true);
  }

  VertexProperty::VertexProperty()
  : priority(0), index(-1), visited(0),
    nparents(0), nremaining(0),
    nconsumers(0), ncremaining(0),
    deviceIndex_(CPU_INDEX),
    chainTail_(true),
    chainID_(-1),
    id(ExpressionID::null_id()),
    expr(NULL),
    execTarget(CPU_INDEX),
    mm(MEM_EXTERNAL),
    fmlid(-99999),
    self_(NULL),
    execSignalCallback(new Signal()),
    color(boost::white_color)
  {
    set_is_persistent(true);
  }

  bool VertexProperty::operator==(const ExpressionID& eid) const {
    return eid == id;
  }

  bool VertexProperty::operator==(const VertexProperty& vp) const {
    return vp.index == index;
  }

  bool VertexProperty::ancestor_finished() {
#     ifdef ENABLE_THREADS
    ExecMutex lock;
#     endif

    --(nremaining);
    if (nremaining <= 0) {
      return true;
    }

    return false;
  }

  /* \brief Notifies this node that one of its consumers has finished */
  bool VertexProperty::consumer_finished() {
#  ifndef ENABLE_CUDA
    if (persistent_) {
      return false;
    }
#   endif
#   ifdef ENABLE_THREADS
    ExecMutex lock;
#   endif

    --(ncremaining);
    if (ncremaining <= 0) {
      return true;
    }

    return false;
  }

  void VertexProperty::execute_expression()
  {
    boost::posix_time::ptime start, stop;
    boost::posix_time::time_duration elapsed;

    /** Execute expression **/
    start = boost::posix_time::microsec_clock::universal_time();
    expr->base_evaluate();
    stop = boost::posix_time::microsec_clock::universal_time();
    elapsed = stop - start;
    /** ------------------ **/

    // Record the timing information
    if      ( execTarget == CPU_INDEX  ) vtp.eTimeCPU_ = ( elapsed.total_microseconds()*1e-6 );
    else if ( IS_GPU_INDEX(execTarget) ) vtp.eTimeGPU_ = ( elapsed.total_microseconds()*1e-6 );

    (*execSignalCallback)(self_); // Notify scheduler that we are done
  }

  void VertexProperty::set_is_persistent( const bool b )
  {
    persistent_ = b;
    if( persistent_ ){
      if     ( execTarget == CPU_INDEX  ) mm = MEM_EXTERNAL;
      else if( IS_GPU_INDEX(execTarget) ) mm = MEM_STATIC_GPU;
    }
    else{
      if     ( execTarget == CPU_INDEX  ) mm = MEM_DYNAMIC;
      else if( IS_GPU_INDEX(execTarget) ) mm = MEM_DYNAMIC_GPU;
    }
  }

  bool VertexProperty::get_is_persistent() const{
    return persistent_;
  }

  void VertexProperty::set_is_edge( const bool b ){
    isEdge_ = b;
  }

  bool VertexProperty::get_is_edge() const{
    return isEdge_;
  }

  VertexProperty&
  VertexProperty::operator=( const VertexProperty& vp )
  {
    priority          = vp.priority;
    index             = vp.index;
    visited           = vp.visited;
    nparents          = vp.nparents;
    nremaining        = vp.nremaining;
    nconsumers        = vp.nconsumers;
    ncremaining       = vp.ncremaining;
    deviceIndex_      = vp.deviceIndex_;
    chainTail_        = vp.chainTail_;
    chainID_          = vp.chainID_;
    nodeMemoryBound_  = vp.nodeMemoryBound_;
    id                = vp.id;
    expr              = vp.expr;
    vtp               = vp.vtp;
    execTarget        = vp.execTarget;
    mm                = vp.mm;
    fmlid             = vp.fmlid;
    self_             = vp.self_;
    poller            = vp.poller;
    nonBlockPoller    = vp.nonBlockPoller;
    execSignalCallback= vp.execSignalCallback;
    color             = vp.color;
    ancestorList.clear();
    consumerList.clear();
#   ifdef ENABLE_CUDA
    consumerStreamList.clear();
#   endif
    BOOST_FOREACH( VertexProperty* v, vp.ancestorList ) ancestorList.push_back(v);
    BOOST_FOREACH( VertexProperty* v, vp.consumerList ) consumerList.push_back(v);
#   ifdef ENABLE_CUDA
    BOOST_FOREACH( cudaStream_t stream, vp.consumerStreamList ) consumerStreamList.push_back(stream);
#   endif
    set_is_persistent( vp.get_is_persistent() );
    set_is_edge      ( vp.get_is_edge()       );

    return *this;
  }

  std::ostream&
  operator<<( std::ostream& os, const VertexProperty& vp )
  {
    os << "Properties for vertex: " << vp.expr->get_tags()[0].name() << std::endl
       << "\t #parents    : " << vp.nparents << std::endl
       << "\t #consumers  : " << vp.nconsumers << std::endl
       << "\t self        : " << vp.self_ << std::endl
       << "\t priority    : " << vp.priority << std::endl
       << "\t Exec target : " << (vp.execTarget == CPU_INDEX ? "CPU" : "GPU" ) << std::endl
       << "\t Mem mgr     : " << vp.mm << std::endl
       << "\t Peristent   : " << (vp.get_is_persistent() ? "true" : "false") << std::endl;
    return os;
  }

} // namespace Expr

