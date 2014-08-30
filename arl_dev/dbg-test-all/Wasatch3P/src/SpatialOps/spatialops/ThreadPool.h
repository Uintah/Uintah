/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef Field_Expr_ThreadPool_h
#define Field_Expr_ThreadPool_h

#include <stdio.h>
#include <map>
#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/threadpool/threadpool.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace SpatialOps{

  /**
   * \brief Wrapper for a priority thread pool
   */
  class ThreadPool : public boost::threadpool::prio_pool
  {
    /* constructor is private because this is implemented as a singleton. */
    ThreadPool( const int nthreads );
    ~ThreadPool();

  public:
    /** \brief obtain the singleton instance of ThreadPool */
    static ThreadPool& self();

    /**
     * @brief set the number of active worker threads in the pool.
     * @param threadCount the number of active threads in the pool
     * @return the number of threads in the pool
     */
    static int resize_pool( const int threadCount );

    /**
     * @return the number of active worker threads in the pool
     */
    static int get_pool_size();

    /**
     * @brief set the maximum number of worker threads in the pool.
     * @param threadCount the maximum number of threads in the pool
     * @return the maximum number of threads in the pool
     */
    static int set_pool_capacity( const int threadCount );

    /**
     * @return the maximum number of threads in the pool
     */
    static int get_pool_capacity();

  private:
    bool init_;
  };

  /**
   * \brief Wrapper for a FIFO thread pool
   */
  class ThreadPoolFIFO : public boost::threadpool::fifo_pool{
    ThreadPoolFIFO( const int nthreads );
    ~ThreadPoolFIFO();

  public:

    /** \brief obtain the singleton instance of ThreadPoolFIFO */
    static ThreadPoolFIFO& self();

    /**
     * @brief set the number of active worker threads in the pool.
     * @param threadCount the number of active threads in the pool
     * @return the number of threads in the pool
     */
    static int resize_pool( const int threadCount );

    /**
     * @return the number of active worker threads in the pool
     */
    static int get_pool_size();

    /**
     * @brief set the maximum number of worker threads in the pool.
     * @param threadCount the maximum number of threads in the pool
     * @return the maximum number of threads in the pool
     */
    static int set_pool_capacity( const int threadCount );

    /**
     * @return the maximum number of threads in the pool
     */
    static int get_pool_capacity();

    static bool is_thread_parallel();

  private:
    bool init_;
  };
} // namespace SpatialOps

#endif // Field_Expr_ThreadPool_h
