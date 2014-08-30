/**
 *  \file   HybridScheduler.h
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

#ifndef HYBRIDSCHEDULER_H_
#define HYBRIDSCHEDULER_H_

#include <expression/SchedulerBase.h>

//-- Boost includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/transpose_graph.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

namespace Expr{

  class GPULoadBalancer;  // forward declaration

  /**
   * \class HybridScheduler
   * \author Devin Robison
   * \brief a priority scheduler for graphs deploying on mixed CPU/GPU architectures.
   */
  class HybridScheduler: public Scheduler {

  public:

    HybridScheduler( Graph& depGraph, Graph& execGraph );

    ~HybridScheduler(){}

    /**
     * \brief Return this scheduler as its base type
     */
    Scheduler* get_base_pointer(){
      return dynamic_cast<Scheduler*>(this);
    }

    /**
     * \brief Initialize any data structures and information required for
     *  'run()' to complete. After this function runs, our scheduler should be
     *  in a runnable state
     *
     *  \todo To make scheduling smarter, we may want to enumerate all
     *   available devices and memory in order to come up with a kind of
     *   'banker's algorithm' to avoid over scheduling.
     */
    void setup( const bool hasRegisteredFields = false );


    /**
     * \brief Begin executing on the graph by loading root expressions onto the queue.
     */
    void run();

    /**
     * \brief Begin executing on the graph by loading root expressions onto the queue.
     */
    void finish();

    /**
     * \brief Process tasks as they finish.
     *
     * This function is called by an expression when it has finished executing
     * we do introspection and determine which nodes are ready to run from here.
     */
    void exec_callback_handler( void* );

    /**
     * \brief return a string identifying which scheduler we are.
     */
    const std::string get_identity() const{
      return std::string("GPU Scheduler -- Testing");
    }

    /**
     * \brief Setup and run a specific task.  Intermediary for executing a node
     *  when it is ready, used so that we can control when we bind memory to each field
     */
    void call( VertexProperty& target );

    /**
     *  Decrement the number of expression resources remaining to be computed
     */
    void dec_remaining();

#   ifdef ENABLE_CUDA
    /**
     * \brief sets device index for the components in Scheduler.
     *
     * Note : call to this method doesn't create cuda streams. It will only set the
     *        device index on the scheduler.
     *
     * @param deviceIndex set on the scheduler
     */
    void set_device_index( int deviceIndex );
#   endif

  protected:

    typedef std::vector<Vertex>             VertList;
    typedef VertList::iterator              RootIter;
    typedef std::map<ExpressionID, Vertex>  ID2VP;
    typedef std::multimap<ExpressionID, short int>   EID2DIDX;

    unsigned int nelements_;  ///< Holds info on number of vertices (nodes) in the graph
    unsigned int nremaining_;
    bool rungpu_;
    bool flip_;

    int runCount_;
    int dummyRun_;  ///< Dummy iterations to record the Node timing info

    int deviceID_;          ///< GPU device index set to the scheduler
    bool coalescingChain_;  ///< determines if coalescing chains algorithm is active

    VertList rootList_;
    EID2DIDX eid2didx_; ///< keeps a track of all the vertices that have consumers with various device index

    GPULoadBalancer* const gpuLoadBalancer_;

#   ifdef ENABLE_CUDA

    /**
     * \brief creates and records information of cuda stream for each expression on the
     *        device index set to the scheduler.
     *
     * Note : Cuda Stream for each dependency vertices are recorded to keep a track of completion status
     *        during execution
     */
    void setup_cuda_stream();

    /**
     * \brief blocks host thread so that cuda streams of the dependency expressions attain completion before
     *        start executing the target expression.
     *
     * @param VertexProperty of the current expression is needed as it keeps a track of streams for its
     *        dependency expressions
     */
    void wait_on_cuda_stream( VertexProperty& target );

#   endif

  };

} // namespace Expr


#endif /* HYBRIDSCHEDULER_H_ */
