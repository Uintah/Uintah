/*
 * Poller.h
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

#ifndef POLLER_H_
#define POLLER_H_

#include <expression/ExprFwd.h>
#include <expression/Tag.h>

namespace Expr{

  class FieldManagerList;
  struct VertexProperty;

  /**
   * \class  PollWorker
   * \author James C. Sutherland
   * \date   November, 2012
   * \brief  abstract base class to accomplish work in a Poller.
   *
   * PollWorker objects have two states: active and inactive.  When inactive,
   * the Poller will not call the () method on the PollWorker.  PollWorker
   * objects will be deactivated once their condition is satisfied.
   *
   * Note that the isBlocked_ flag should be monitored by derived classes.  If
   * it is true, then the derived class should ensure that the () operator does
   * not return until their condition is satisfied.
   */
  class PollWorker
  {
  protected:
    bool isActive_;   ///< turn on/off this worker.
    bool isBlocked_;  ///< force blocking of the () operator until the condition is satisfied
  public:
    PollWorker();
    virtual ~PollWorker();
    inline bool is_active() const{ return isActive_; }
    inline void deactivate(){ isActive_=false; }
    inline void activate(){ isActive_=true; }
    inline void set_blocking( const bool block ){ isBlocked_ = block; }

    /**
     * @brief trigger this PollWorker.
     *
     * @param fml the FieldManagerList.  If a PollWorker modifies a field when
     *        its condition is satisfied, then this allows access to the field.
     *        Note that this can be very dangerous since the Scheduler cannot
     *        guarantee that the field is available, so this should be done with
     *        great caution!
     * @return true if the condition is satisfied, false otherwise.
     */
    virtual bool operator()( FieldManagerList* fml ){ assert(false); return false; }


    /**
     * @brief trigger this PollWorker.
     * @return true if the condition is satisfied, false otherwise.
     */
    virtual bool operator()(){ assert(false); return false; }
  };

  /**
   * \class  Poller
   * \author James C. Sutherland
   * \date   November, 2012
   *
   * This allows indirect dependencies to be introduced into a graph.  For
   * example, an MPI poller could be constructed that would check status of
   * incoming messages and be satisfied once all incoming messages have been
   * received.  Then an expression could be triggered.
   *
   * When a node in the graph has a Poller attached to it, it will not be marked
   * as completed until all of the Poller's PollWorkers are completed.
   *
   * PollWorker objects are added to a Poller to accomplish work.  Each
   * PollWorker can be "active" or "inactive" depending on whether it has its
   * conditions satisfied.  When \c run() is called, each active PollWorker has
   * its () operator called on it.  If the result is "true" then the PollWorker
   * will be deactivated.
   *
   * Note that in general you should not create a poller directly. Rather, obtain
   * one from the factory via:
   * \code
   *  PollerPtr poller = factory.get_poller( myFieldTag );
   * \endcode
   * where \c myFieldTag is the tag for the field for which a poller is desired.
   */
  class Poller
  {
    const Tag fieldTag_;
    size_t nActive_;
    std::vector<PollWorkerPtr> workers_;
    VertexProperty* vp_;
    bool doBlock_;
  public:

    /**
     * @brief create a Poller associated with the given Tag.
     * Note that Poller objects need to be added to the ExpressionTree via the
     *  \c add_poller() method.
     * @param monitoredField the tag that this Poller is associated with.
     */
    Poller( const Tag& monitoredField );
    ~Poller();

    /** \brief add a new worker */
    void add_new( PollWorkerPtr worker );

    /** \brief activate all PollWorkers */
    void activate_all();

    /** \brief deactivate all PollWorkers */
    void deactivate_all();

    /** \brief determine if this Poller has active PollWorkers */
    bool is_active() const;

    /**
     *  \brief next time the Poller is run, it will block on each PollWorker until it is satisfied
     */
    void set_blocking( const bool block );

    inline const Tag& target_tag() const{ return fieldTag_; }

    /**
     * By setting the vertex property here, we can later pull this out to deal
     * with callbacks and such.  See Scheduler classes.
     *
     * @param vp the VertexProperty associated with this Poller.  This should be
     *           used by the ExpressionTree.
     */
    void set_vertex_property( VertexProperty* vp );

    inline VertexProperty* get_vertex_property() const{ assert(vp_!=NULL); return vp_; }

    /**
     * @param  fml the FieldManagerList.  If a PollWorker modifies a field when
     *        its condition is satisfied, then this allows access to the field.
     *        Note that this can be very dangerous since the Scheduler cannot
     *        guarantee that the field is available, so this should be done with
     *        great caution!
     * @return true if all PollWorkers have completed.  False otherwise.
     */
    bool run( FieldManagerList* );
  };


  /**
   * \class NonBlockingPoller
   * \author James C. Sutherland
   * \date July, 2014
   * \brief Provides non-blocking poller functionality
   *
   * Non-blocking pollers can be attached to an expression to perform periodic
   * checks on some condition and perform work when the condition is satisfied.
   *
   * Unlike regular pollers, non-blocking pollers do not block execution of the
   * expression that they are associated with.
   *
   * For example, a NonBlockingPoller might be used in conjunction with a
   * non-blocking MPI send to check the status on the sender and manage any
   * temporary storage buffers associated with it.
   */
  class NonBlockingPoller
  {
    const Tag fieldTag_;
    size_t nActive_;
    std::vector<PollWorkerPtr> workers_;
    bool doBlock_;
  public:

    NonBlockingPoller( const Tag& tag );
    virtual ~NonBlockingPoller();

    /** \brief add a new worker */
    void add_new( PollWorkerPtr worker );

    /** \brief activate all PollWorkers */
    void activate_all();

    /** \brief deactivate all PollWorkers */
    void deactivate_all();

    /** \brief determine if this Poller has active PollWorkers */
    bool is_active() const;

    /**
     *  \brief next time the Poller is run, it will block on each PollWorker until it is satisfied
     */
    void set_blocking( const bool block );

    inline const Tag& target_tag() const{ return fieldTag_; }

    /**
     * \brief run all of the PollWorkers associated with this poller.
     * \return true if all are complete; false if at least one has not yet completed
     */
    bool run();

  };

} // namespace Expr

#endif /* POLLER_H_ */
