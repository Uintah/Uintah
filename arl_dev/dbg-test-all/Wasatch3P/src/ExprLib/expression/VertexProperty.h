/**
 *  \file   VertexProperty.h
 *
 * Copyright (c) 2011 The University of Utah
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

/* --- standard includes --- */
#include <stdio.h>

/* --- boost includes --- */
#ifndef Expr_VertexProperty_h
#define Expr_VertexProperty_h

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

/* --- expression includes --- */
#include <expression/ManagerTypes.h>
#include <expression/ExpressionID.h>
#include <expression/ExpressionBase.h>
#include <expression/Poller.h>

#include <spatialops/SpatialOpsConfigure.h> // defines thread stuff.
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace Expr {

//todo, gpuTransTime_ needs to be computed once somewhere and then slotted in here.
class VertexTimingProperties {
  public:
    VertexTimingProperties() :
        eTimeCPU_(0),
        eTimeGPU_(0),
        sTime_(0),
        fTime_(0)
    {}

    ~VertexTimingProperties() {}

    static double gpuTransTime_;
    static void gen_gputrans_time(){
#     ifdef ENABLE_CUDA
      //TODO: determine a generic way to test transfer delay
#     endif
    }

    double eTimeCPU_, eTimeGPU_, sTime_, fTime_;
};

/**
 * \struct VertexProperty
 * \brief holds information on each vertex (node) of the graph
 * \todo make some of these member variables private
 */
struct VertexProperty {
  public:
    typedef boost::signals2::signal<void(void*)> Signal;

    VertexProperty(const int _ix, ExpressionID _id, ExpressionBase* _expr);

    VertexProperty();

    VertexProperty& operator=( const VertexProperty& vp );

    bool operator==(const ExpressionID& eid) const;

    bool operator==(const VertexProperty& vp) const;

    int priority, index, visited;  ///< Node execution priority, index, and 'visited' indicator for a graph algorithms
    int nparents, nremaining;      ///< Total number of requirements of this node, and the number who have yet to be ready
    int nconsumers, ncremaining;   ///< Total number of consumers of this node, and the number of have not yet consumed

    short int deviceIndex_;         ///< Which device this vertex should be run on
    bool chainTail_;                ///< Is this the tail node in a chain?
    int chainID_;                   ///< Identity of the chain this element belongs to

    unsigned long int nodeMemoryBound_; ///< Maximum memory required by this node during computation == field_size * nparents

    ExpressionID id;               ///< Associated expression's id
    ExpressionBase* expr;          ///< Associated expression
    VertexTimingProperties vtp;    ///< Future use object
    short int execTarget;          ///< Execution target ( device hardware where it will be computed )
    MemoryManager mm;              ///< Memory manager type for this object
    int fmlid;                     ///< when using multiple FieldManagerList, this indicates which one is used for this vertex.

    void* self_;                   ///< Pointer to the boost graph vertex possessing this VertexProperty

    std::vector<VertexProperty*> ancestorList; ///< Pointers to all vertex properties in the execution graph
                                               ///< which are dependencies for this vertex.
    std::vector<VertexProperty*> consumerList; ///< Pointers to all vertex properties in the execution graph
                                               ///< which are consumers of this vertex.

#   ifdef ENABLE_CUDA
    std::vector<cudaStream_t> consumerStreamList; ///< Container for all the dependency expression cudaStreams
#   endif
    PollerPtr poller;
    NonBlockingPollerPtr nonBlockPoller;

    boost::shared_ptr<Signal> execSignalCallback; ///< Signal object that is called when this node finishes execution
    boost::default_color_type color;

    /** \brief Notifies this node that one of its ancestors has finished executing. */
    bool ancestor_finished();

    /** \brief Notifies this node that one of its consumers has finished */
    bool consumer_finished();

    /** \brief calls base_evaluate method of the underlying expression and obtains timing information */
    void execute_expression();

    /** \brief force the persistence tag for this node */
    void set_is_persistent(const bool b);

    /** \brief returns this node's persistence flag */
    bool get_is_persistent() const;

    /** \brief flags this as being an 'edge' node in the graph */
    void set_is_edge(const bool b);

    /** \brief return this node's edge flag */
    bool get_is_edge() const;

  private:

    bool persistent_; ///< Flag, node is persistent ( not eligible for dynamic allocation )
    bool isEdge_; ///< Flag, this is an edge node in the graph

    class ExecMutex {
#   ifdef ENABLE_THREADS
        const boost::mutex::scoped_lock lock;
        inline boost::mutex& get_mutex() const
        {
          static boost::mutex m; return m;
        }
        public:
        ExecMutex() : lock( get_mutex() ) {}
        ~ExecMutex() {}
#   else
      public:
        ExecMutex() {
        }
        ~ExecMutex() {
        }
#   endif
    };

};

std::ostream&
operator<<( std::ostream& os, const VertexProperty& vp );

} // namespace Expr

#endif // Expr_VertexProperty_h
