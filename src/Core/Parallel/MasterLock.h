/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef CORE_PARALLEL_MASTERLOCK_H
#define CORE_PARALLEL_MASTERLOCK_H

#include <Core/Parallel/Parallel.h>

#include <sci_defs/kokkos_defs.h>

#include <mutex>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace Uintah {

class MasterLock
{

  // Specific to OpenMP- and std::thread-based implementations
  // To avoid mixing thread environment types (e.g. using mutexes in an OpenMP threaded environment),
  // this class determines which kind of locking to use dependent on the parallel thread environment.

  // This lock can be used as follows:
  //
  // 1.) Same functionality as std::mutex, e.g., create, lock(), and unlock()
  //
  //      OR
  //
  // 2.) std::unique_lock<MasterLock>
  //
  //       std::unique_lock is a general-purpose mutex ownership wrapper allowing deferred locking, time-constrained
  //       attempts at locking, recursive locking, transfer of lock ownership, and use with condition variables.
  //
  //      OR
  //
  // 3.) std::lock_guard<MasterLock>
  //
  //       std::lock_guard is a mutex wrapper that provides a convenient RAII-style mechanism for
  //       owning a mutex for the duration of a scoped block.


  public:
    // per OMP standard, a flush region without a list is implied for omp_{set/unset}_lock
    void lock()   {
#if defined(_OPENMP)
      if (Parallel::getCpuThreadEnvironment() == Parallel::CpuThreadEnvironment::OPEN_MP_THREADS) {
        omp_set_lock( &m_lock );
        m_mutex_used = false;
        return;
      }
#endif
      m_mutex.lock();
      m_mutex_used = true;
    }

    void unlock()   {
#if defined(_OPENMP)
      if (!m_mutex_used) {
        omp_unset_lock( &m_lock );
        return;
      }
#endif
      m_mutex.unlock();
    }

    MasterLock()  {
    // Initialize the locks (mutexes initialize themselves)
#if defined(_OPENMP)
      omp_init_lock( &m_lock );
#endif
    }

    ~MasterLock()  {
#if defined(_OPENMP)
      omp_destroy_lock( &m_lock );
#endif
    }

  private:

    // disable copy, assignment, and move
    MasterLock( const MasterLock & )            = delete;
    MasterLock& operator=( const MasterLock & ) = delete;
    MasterLock( MasterLock && )                 = delete;
    MasterLock& operator=( MasterLock && )      = delete;

#if defined(_OPENMP)

    omp_lock_t m_lock;

#endif

    std::mutex m_mutex;
    bool m_mutex_used{true};

};

} // end namespace Uintah

#endif // end CORE_PARALLEL_MASTERLOCK_H
