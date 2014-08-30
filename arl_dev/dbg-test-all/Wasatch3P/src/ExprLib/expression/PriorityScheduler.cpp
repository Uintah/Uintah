/**
 *  \file   PriorityScheduler.cpp
 *  \date   Aug 23, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
 *
 */

#include <expression/PriorityScheduler.h>

//-- SpatialOps
#include <spatialops/structured/MemoryTypes.h>
#include <spatialops/structured/ExternalAllocators.h>

//-- Boost includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/transpose_graph.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>

namespace Expr{

  /**
   * \brief Boost visitor structure for setting node priorities
   */
  struct ExecPriorityVisitor: public boost::default_bfs_visitor {

    ExecPriorityVisitor(){}

    inline void examine_edge( const Edge e, const Graph& g ){
      const Vertex src = boost::source(e,g);
      const Vertex dest = boost::target(e,g);
      const int srcPriority = g[src].priority;
      Graph& g2 = const_cast<Graph&>(g);
      int& priority = g2[dest].priority;
      priority = std::max(priority, srcPriority + 1);
    }
  };

  //=================================================================

  PriorityScheduler::PriorityScheduler( Graph& depGraph, Graph& execGraph )
  : Scheduler(execGraph,depGraph)
  {}

  void PriorityScheduler::call( VertexProperty& target )
  {
    FieldManagerList* const fml = extract_field_manager_list( this->fmls_, target.fmlid );

    try{
      target.expr->base_bind_fields(*fml);
      target.execute_expression();
      this->run_pollers( target, fml );
#     ifdef ENABLE_THREADS
      if( pool_.active() == 1 ) block_pollers();
#     endif
    }
    catch( std::exception& err ){
      std::ostringstream msg;
      msg << std::endl << "Error trapped while executing expression: "
          << target.expr->get_tags()[0] << std::endl
          << "details follow..." << std::endl
          << err.what() << std::endl;
      throw std::runtime_error( msg.str() );
    }
  }

  //-------------------------------------------------------------------

  void PriorityScheduler::exec_callback_handler( void* expr_vertex )
  {
    const Vertex v = (Vertex) expr_vertex;
    VertexProperty& vpJustFinished = execGraph_[v];

    // Notify the vertex that a consumer has finished, it returns true, it can be freed.
    std::vector<VertexProperty*>::iterator vpit  = vpJustFinished.ancestorList.begin();
    std::vector<VertexProperty*>::iterator vpend = vpJustFinished.ancestorList.end();
    for( ; vpit != vpend; ++vpit ){
      if( (*vpit)->consumer_finished() ){
        if( this->fdm_ != NULL ){
          FieldDeps& fd = *((*this->fdm_)[(*vpit)->id]);
          FieldManagerList* const fml = extract_field_manager_list( this->fmls_, (*vpit)->fmlid );
#         ifndef DEBUG_NO_FIELD_RELEASE
          fd.release_fields(*fml);
#         endif
        }
      }
    }

    // Notify the vertex that an ancestor has finished, it returns true if it is ready.
    // If it is, we either toss it to the thread pool or run it.
    vpend = vpJustFinished.consumerList.end();
    for( vpit = vpJustFinished.consumerList.begin(); vpit!=vpend; ++vpit ) {
      if ((*vpit)->ancestor_finished()) {
#       ifdef ENABLE_THREADS
        this->pool_.schedule(
            boost::threadpool::prio_task_func( (*vpit)->priority,
                                               boost::bind( &PriorityScheduler::call, this, **vpit ) ) );
#       else
        this->call(**vpit);
#       endif
      }
    }

    dec_remaining();
  }

  //-------------------------------------------------------------------

  void PriorityScheduler::dec_remaining()
  {
#   ifdef ENABLE_THREADS
    ExecMutex<200> lock;
#   endif
    --nremaining_;
#   ifdef ENABLE_THREADS
    if( nremaining_ == 0 ) this->schedBarrier_.post();
#   endif
  }

  //-------------------------------------------------------------------

  void PriorityScheduler::setup( const bool hasRegisteredFields )
  {
   /*
    * If we have been invalidated
    *  - reset all callback handles in the graph
    *  - recalculate all nparent_ counts for each vertex
    *
    * Notes on whats going on here
    *  - execGraph is the execution graph
    *  - depGraph is the consumer (dependency) graph
    *
    *  Both are used to build up information required for determining node priority,
    *  node consumers, and node execution requirements.
    *
    *  Step 1: reset and reconnect all variables to place the graph into state which is execute ready.
    *
    *  Step 2: inspect the dependency graph in order to determine each nodes execution priority and
    *    determine its consumer count. ( The number of nodes that consume an expression and its fields ).
    *
    *  Step 3: inspect the execution graph to determine the number of parent nodes for each expression;
    *    during execution this will allow us to know when an expression is ready to run.
    *
    *  Step 4: determine each node's memory constraints, currently this is limited to deciding if the
    *    expression can use dynamic memory.
    */

    if( !invalid_ ) return; // Quick return if we're already valid

    rootList_.clear();

    ID2VP execVertexMap;  // jcs this duplicates the exprVertexMapT_ container on the ExpressionTree

    // Update element counts
    nelements_ = boost::num_vertices(execGraph_);
    nremaining_ = nelements_;

    const std::pair<VertIter, VertIter> execGraphVertices = boost::vertices(execGraph_);
    const std::pair<VertIter, VertIter> depGraphVertices  = boost::vertices(depGraph_ );

    // ------------------------------ **/

    /* jcs
       need to see if we can make this more efficient.
        - Can we leave callback signals in place?
        - Do we need to reset execution counts?
     */

    //-- Step 1
    // Reconnect all signals and reset execution counts
    for( VertIter viter = execGraphVertices.first; viter != execGraphVertices.second; ++viter ){
      const Vertex& vert = *viter;
      execVertexMap.insert( std::make_pair(execGraph_[vert].id, vert) );

      VertexProperty& vp = execGraph_[vert];
      vp.self_      = (void*) (vert);
      vp.nparents   = 0;
      vp.nconsumers = 0;
      vp.priority   = 0;
      vp.execSignalCallback.reset( new VertexProperty::Signal() );
      vp.execSignalCallback->connect( boost::bind(&PriorityScheduler::exec_callback_handler, this, vp.self_) );
      vp.ancestorList.clear();
      vp.consumerList.clear();
      vp.set_is_edge(false);
    }

    for( VertIter vit = depGraphVertices.first; vit != depGraphVertices.second; ++vit ){
      VertexProperty& evp = execGraph_[ execVertexMap[depGraph_[*vit].id] ];
      VertexProperty& dvp = depGraph_[*vit];
      if     ( dvp.poller ) evp.poller = dvp.poller;
      else if( evp.poller ) dvp.poller = evp.poller;
      if( evp.poller ) evp.poller->set_vertex_property(&evp);
      if     ( dvp.nonBlockPoller ) evp.nonBlockPoller = dvp.nonBlockPoller;
      else if( evp.nonBlockPoller ) dvp.nonBlockPoller = evp.nonBlockPoller;
    }

    // set node information - must be done in a separate loop because the
    // ancestorList and consumerList must be cleared before we push back on them.
    for( VertIter viter = execGraphVertices.first; viter != execGraphVertices.second; ++viter ){
      VertexProperty& evp = execGraph_[*viter];
      execVertexMap.insert( std::make_pair(evp.id, *viter) );
      std::pair<OutEdgeIter, OutEdgeIter> edges = boost::out_edges(*viter, execGraph_);
      for( OutEdgeIter eit = edges.first; eit != edges.second; ++eit ){
        VertexProperty& tvp = execGraph_[ boost::target(*eit, execGraph_) ];

        evp.consumerList.push_back(&tvp);
        tvp.ancestorList.push_back(&evp);
        (evp.nconsumers)++;
        (tvp.nparents)++;
        if( evp.poller ) (tvp.nparents)++;
        if( tvp.poller ) (evp.nconsumers)++;
      }
    }

    //-- Step 2
    //*** top down priority scheduling ***//

    // bfs from each top down 'root'
    // Since this is the dependence graph, root nodes are at the 'top' and will have no consumers
    // Since no edge nodes can be 'scratch' we find root nodes in the consumer graph and use exprIDs
    // to flag them as persistent in the execution graph.

    for( VertIter vit = depGraphVertices.first; vit != depGraphVertices.second; ++vit ){
      const VertexProperty& vp = depGraph_[*vit];
      if (vp.nconsumers == 0) {
        boost::breadth_first_search( depGraph_, *vit,
            boost::color_map(boost::get(&VertexProperty::color, depGraph_)).visitor( ExecPriorityVisitor()));
      }
    }

    //*** end of top down scheduling ***///

    //-- Step 3 & 4
    for( VertIter iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
      VertexProperty& vp = execGraph_[*iter];

      //For the execution graph nodes at the bottom of the tree are roots and have no parents.
      //Since edge nodes cannot be 'dynamic' we flag these nodes as persistent
      if( vp.nparents == 0 ){
        vp.set_is_edge(true);
        rootList_.push_back(*iter);
      }

      if( vp.nconsumers == 0 ) vp.set_is_edge(true);
      if( vp.get_is_edge()   ) vp.set_is_persistent(true);

      vp.nremaining  = vp.nparents;
      vp.ncremaining = vp.nconsumers;
    }

    // synchronize properties.  Dependency graph has priority, Execution graph has everything else.
    // note that some things are not synchronized (e.g. signals, etc.)
    for( VertIter vit = depGraphVertices.first; vit != depGraphVertices.second; ++vit ){
      VertexProperty& evp = execGraph_[ execVertexMap[depGraph_[*vit].id] ];
      VertexProperty& dvp = depGraph_[*vit];
      if     ( dvp.poller ) evp.poller = dvp.poller;
      else if( evp.poller ) dvp.poller = evp.poller;
      if     ( dvp.nonBlockPoller ) evp.nonBlockPoller = dvp.nonBlockPoller;
      else if( evp.nonBlockPoller ) dvp.nonBlockPoller = evp.nonBlockPoller;
      evp.priority = dvp.priority;
      dvp = evp;
    }

    invalid_ = false;
  }

  //-------------------------------------------------------------------

  void PriorityScheduler::run()
  {
    // Execute everything in the root list
    for( RootIter rit = rootList_.begin(); rit != rootList_.end(); ++rit ) {
      VertexProperty& vp = execGraph_[*rit];
#     ifdef ENABLE_THREADS
      this->pool_.schedule(
          boost::threadpool::prio_task_func( vp.priority,
                                             boost::bind(&PriorityScheduler::call,this,vp))
      );
#     else
      this->call(vp);
#     endif
    }

#   ifdef ENABLE_THREADS
    this->schedBarrier_.wait();
#   endif

    // ensure that all pollers have completed
    this->block_pollers();

    finish();
  }

  //-------------------------------------------------------------------

  void PriorityScheduler::finish()
  {
    this->nelements_ = boost::num_vertices(execGraph_);
    this->nremaining_ = this->nelements_;

    const std::pair<VertIter, VertIter> execGraphVertices = boost::vertices( execGraph_ );

    for( VertIter iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
      VertexProperty& vp = execGraph_[*iter];
      vp.nremaining  = vp.nparents;
      vp.ncremaining = vp.nconsumers;
    }
  }

  //=================================================================

} // namespace Expr
