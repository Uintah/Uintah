/**
 *  \file   PriorityScheduler.h
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

#ifndef PRIORITYSCHEDULER_H_
#define PRIORITYSCHEDULER_H_

#include <expression/SchedulerBase.h>

//-- Boost includes
#include <boost/graph/graph_traits.hpp>

namespace Expr{

  /**
   * \class PriorityScheduler
   * \author Devin Robison
   * \brief Implements a scheduler based on a priority queue.
   */
  class PriorityScheduler: public Scheduler
  {
  public:

    PriorityScheduler( Graph& depGraph, Graph& execGraph );

    ~PriorityScheduler(){}

    /**
     * \brief Return this scheduler as its base type
     */
    inline Scheduler* get_base_pointer() {
      return dynamic_cast<Scheduler*>(this);
    }

    /**
     * \brief after this function runs, our scheduler should be in a runnable state
     */
    void setup( const bool hasRegisteredFields = false );

    /**
     * \brief begin executing graph nodes
     */
    void run();

    /**
     * \brief perform any cleanup activities
     */
    void finish();

    /**
     * \brief this function is called by an expression when it has finished executing
     * we do introspection and determine which nodes are ready to run from here.
     */
    void exec_callback_handler(void*);

    /**
     * \brief return a string identifying which scheduler we are.
     */
    inline const std::string get_identity() const{
      return std::string("Default Priority Scheduler");
    }

    /**
     * \brief intermediary for executing a node when it is ready, used so that we can control
     * when we bind memory to each field
     */
    void call( VertexProperty& target );

  protected:

    typedef std::vector<Vertex>             VertList;
    typedef VertList::iterator              RootIter;
    typedef std::map<ExpressionID, Vertex>  ID2VP;

    int nelements_;
    int nremaining_;

    VertList rootList_;

    void dec_remaining();
  };  // class PriorityScheduler


} // namespace Expr

#endif /* PRIORITYSCHEDULER_H_ */
