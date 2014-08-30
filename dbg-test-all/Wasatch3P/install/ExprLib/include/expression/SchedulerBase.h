//DEBUG FLAGS: DEBUG_NO_FIELD_RELEASE ( default undefined )
/*
 * Contract implementation for a task scheduler
 *
 */
#ifndef Expr_TaskSchedulerBase_h
#define Expr_TaskSchedulerBase_h

//Standard libraries
#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>

//Expressions
#include <expression/FieldDeps.h>
#include <expression/ExpressionID.h>
#include <expression/FieldManagerList.h>
#include <expression/VertexProperty.h>
#include <expression/Poller.h>
#include <expression/GraphType.h>

#include <spatialops/structured/MemoryTypes.h>

#include <spatialops/SpatialOpsConfigure.h>  // defines thread stuff.
#ifdef ENABLE_THREADS
# include <spatialops/SpatialOpsTools.h>
# include <spatialops/ThreadPool.h>
# include <spatialops/Semaphore.h>
#endif

namespace Expr {

  /**
   *  \class ExecMutex
   *  \brief Scoped lock. An instance should be constructed within any function
   *   that touches Scheduler member variables.
   *
   *  \tparam An integer to provide a unique ExecMutex.  Objects with the same
   *   provided integer will result in the same locking mechanism being used.
   */
  template<int I>
  class ExecMutex {
#   ifdef ENABLE_THREADS
    const boost::mutex::scoped_lock lock;
    inline boost::mutex& get_mutex() const {static boost::mutex m; return m;}

  public:
    ExecMutex() : lock( get_mutex() ) {}
    ~ExecMutex() {}
#   else
  public:
    ExecMutex(){}
    ~ExecMutex(){}
#   endif
  };


#ifdef ENABLE_THREADS
/**
 * @brief set the number of worker threads to use in executing expressions (task parallel)
 * @param nthread the number of threads
 */
inline void set_soft_thread_count( const int nthread ){
  SpatialOps::ThreadPool::resize_pool(nthread);
}

/**
 * @brief set the maximum number of allowed worker threads to use in executing expressions (task parallel)
 * @param nthread the number of threads
 */
inline void set_hard_thread_count( const int nthread ){
  SpatialOps::ThreadPool::set_pool_capacity(nthread);
}

/** \brief get current soft (active) thread count */
inline int get_soft_thread_count(){ return SpatialOps::ThreadPool::get_pool_size(); }

/** \brief get the current hard (maximum/total) thread count */
inline int get_hard_thread_count(){ return SpatialOps::ThreadPool::get_pool_capacity(); }

#endif // ENABLE_THREADS

/**
 *  \class Scheduler
 *  \author Devin Robison
 *  \brief base class for schedulers that control execution of a graph
 */
class Scheduler {
protected:

  public:
    Scheduler( Graph& execGraph,
               Graph& depGraph )
    : execGraph_(execGraph),
      depGraph_ (depGraph ),
      invalid_(true)
#     ifdef ENABLE_THREADS
      , pool_ ( SpatialOps::ThreadPool::self() )
      , poolx_( SpatialOps::ThreadPoolFIFO::self() )
      , schedBarrier_(0)
#     endif
    {
//#     ifdef ENABLE_THREADS
//      SpatialOps::set_hard_thread_count( NTHREADS );
//      SpatialOps::set_soft_thread_count( NTHREADS );
//      Expr::set_soft_thread_count( NTHREADS );
//      Expr::set_hard_thread_count( 2 );
//#     endif
    }

    virtual ~Scheduler(){}

    //------------------ Interface requirements ------------------

    virtual Scheduler* get_base_pointer() = 0;

    /** \brief Perform any required setup action and pre-processing */
    virtual void setup( const bool hasRegisteredFields ) = 0;

    /** Invalidate the current schedule **/
    virtual void invalidate(){ invalid_ = true; }

    /** \brief Execute the supplied task graph */
    virtual void run() = 0;

    /** \brief Perform any cleanup or post processing */
    virtual void finish() = 0;

    /** \brief Process 'finished' method from a vertex element */
    virtual void exec_callback_handler(void*) = 0;

    /** \brief Return a string identifying the scheduler in use */
    virtual const std::string get_identity() const = 0;

    /** \brief Assign a field manager list to the scheduler */
    virtual void set_fml( FieldManagerList* fml ) {
      this->fmls_[0] = fml;
    }
    virtual void set_fmls( FMLMap& fmls ) {
      this->fmls_ = fmls;
    }

    /** \brief Store a copy of the field dependencies for this graph */
    virtual void set_fdm(
        std::map<ExpressionID, boost::shared_ptr<FieldDeps> >* fdm) {
      this->fdm_ = fdm;
    }

    void set_poller( PollerPtr p );
    void set_nonblocking_poller( NonBlockingPollerPtr p );

#   ifdef ENABLE_CUDA
    /** \brief set up device index for the scheduler */
    virtual void set_device_index( int deviceIndex ) = 0;
#   endif

  protected:
    Graph& execGraph_, depGraph_;

    FMLMap fmls_;
    std::map<ExpressionID, boost::shared_ptr<FieldDeps> >* fdm_;

    PollerList pollers_;
    NonBlockingPollerList nonBlockingPollers_;

    bool invalid_;

#   ifdef ENABLE_THREADS
    SpatialOps::ThreadPool& pool_;
    SpatialOps::ThreadPoolFIFO& poolx_;
    SpatialOps::Semaphore schedBarrier_;
#   endif

    /**
     * Executes all pollers associated with this Scheduler.  If a poller
     * indicates that its conditions are satisfied, then a callback occurs on
     * the vertices connected to this node to indicate that one of their
     * dependencies has been satisfied.
     *
     * @param vp the vertex property that we are currently executing.  If there
     *  is a poller object attached to this vertex property then it will be
     *  added to the list of Pollers.
     *
     * @param fml the FieldManagerList
     */
    void run_pollers( VertexProperty& vp, FieldManagerList* const fml );

    /**
     * Force all pollers that remain active (unsatisfied) to complete.
     */
    void block_pollers();
};

} // namespace Expr

#endif // Expr_TaskScheduler_h
