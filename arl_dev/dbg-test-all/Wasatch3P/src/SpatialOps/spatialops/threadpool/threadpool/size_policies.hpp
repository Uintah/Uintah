/*! \file
* \brief Size policies.
*
* This file contains size policies for thread_pool. A size 
* policy controls the number of worker threads in the pool.
*
* Copyright (c) 2005-2007 Philipp Henkel
*
* Modified 2012 Devin Robison
* 	- Fix ambiguous or incorrect naming conventions
* 	- Add functionality
* 	- Remove incomplete functionality
*
* Use, modification, and distribution are  subject to the
* Boost Software License, Version 1.0. (See accompanying  file
* LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*
* http://threadpool.sourceforge.net
*
*/

#ifndef THREADPOOL_SIZE_POLICIES_HPP_INCLUDED
#define THREADPOOL_SIZE_POLICIES_HPP_INCLUDED

/// The namespace threadpool contains a thread pool and related utility classes.
namespace boost { namespace threadpool {
  /**
   * \brief SizePolicyController which provides no functionality.
   *
   * \param Pool The pool's core type.
  */ 
  template<typename Pool>
  struct empty_controller
  {
    empty_controller(typename Pool::size_policy_type&, shared_ptr<Pool>) {}
  };

  /**
   * @brief SizePolicyController which allows resizing.
   * @param Pool The pool's core type.
  */ 
  template< typename Pool >
  class resize_controller
  {
    typedef typename Pool::size_policy_type size_policy_type;

    reference_wrapper<size_policy_type> m_policy;
    shared_ptr<Pool> m_pool;                           //!< to make sure that the pool is alive (the policy pointer is valid) as long as the controller exists

  public:
    resize_controller(size_policy_type& policy, shared_ptr<Pool> pool)
      : m_policy(policy)
      , m_pool(pool)
    {}

    /**
     * @brief Resize the number of thread pool workers.
     * @param worker_count -- number of total workers
     * @return
     */
    bool resize(size_t worker_count) {
    	return m_policy.get().resize(worker_count);
    }

    /**
     * @brief Resize cap on the number of simultaneously active workers
     * @param worker_count -- number of workers which can be active
     * @return
     */
    bool resize_active(size_t const worker_count) {
    	return m_policy.get().resize_active(worker_count);
    }
  };

  /**
   * @brief SizePolicy which preserves the thread count.
   * @param Pool The pool's core type.
  */ 
  template<typename Pool>
  class dynamic_resize_policy
  {
    reference_wrapper<Pool volatile> m_pool;

  public:
    dynamic_resize_policy(Pool volatile & pool)
      : m_pool(pool)
    {}

    static void init(Pool& pool, size_t const worker_count) {
      pool.resize(worker_count);
      pool.resize_active(worker_count);
    }

    bool resize(size_t const worker_count) {
      return m_pool.get().resize(worker_count);
    }

    bool resize_active(size_t const worker_count) {
      return m_pool.get().resize_active(worker_count);
    }

    void worker_died_unexpectedly(size_t const new_worker_count) {
      m_pool.get().resize(new_worker_count + 1);
    }

    /**
    	void task_scheduled() {}
    	void task_finished() {}
    **/
  };

} } // namespace boost::threadpool

#endif // THREADPOOL_SIZE_POLICIES_HPP_INCLUDED
