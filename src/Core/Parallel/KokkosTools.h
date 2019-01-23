/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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

// The purpose of this file is to provide a single include for application developers to use
// to access all needed tools for portability (the loops, macros, cell range objects,
// execution objects containing architecture specific data members, etc.).

#ifndef UINTAH_CORE_KOKKOS_TOOLS_H
#define UINTAH_CORE_KOKKOS_TOOLS_H

#if defined( UINTAH_ENABLE_KOKKOS ) 

#include <Core/Parallel/MasterLock.h>
#include <Kokkos_Random.hpp>
#include <sci_defs/kokkos_defs.h>
#include <memory>

namespace {
  Uintah::MasterLock rand_init_mutex{};
}

namespace Uintah {


//----------------------------Portable Random Number Generation-------------------------------------

    // Note, these should be accessed after Kokkos::initialize and before Kokkos::finalize 
    // We want Kokkos Random Number Generator objects to persist through the program.  This
    // is especially important for CUDA asynchronous kernels.
    // Note, I'd rather these random objects all stored in a single collection.  But I couldn't figure out
    // a way in C++11 to do that.  The Type Erasure Idiom was close, but our need relied on templated return types
    // and it couldn't mesh with Type Erasure Idiom's polymorphism.  std::variant seems like a decent idea, but that's C++17.
    // For now, I'm hard coding the two options we use, Kokkos::OpenMP and Kokkos::Cuda.  -- Brad P.

// Prototype declaration
template <typename RandomGenerator>
class KokkosRandom;

#if defined(KOKKOS_ENABLE_OPENMP)
    std::unique_ptr< KokkosRandom< Kokkos::Random_XorShift1024_Pool< Kokkos::OpenMP > > > openMPRandomPool;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    std::unique_ptr< KokkosRandom< Kokkos::Random_XorShift1024_Pool< Kokkos::Cuda > > > cudaRandomPool;
#endif

template < typename RandomGenerator>
class KokkosRandom {

public:

  // Initialize once within host code (synchronizes streams on GPU)
  KokkosRandom( bool seedWithTime ) {

    // Seed using time
    uint64_t ticks{0};

    if (seedWithTime) {
      ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    m_rand_pool = std::make_shared< RandomGenerator >(ticks);
  }

  RandomGenerator& getRandPool() { 
    return *m_rand_pool; 
  }

private:

  std::shared_ptr<RandomGenerator> m_rand_pool;

};  // end class KokkosRandom

void cleanupKokkosTools() {
  {
    std::lock_guard<Uintah::MasterLock> rand_init_mutex_guard(rand_init_mutex);
#if defined(KOKKOS_ENABLE_OPENMP)
    if (openMPRandomPool) {
      openMPRandomPool.release();
    }
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    if (cudaRandomPool) {
      cudaRandomPool.release();
    }
#endif
  }
}

// Don't create any pool until a user first requests one.  Once one is requested, reuse it.
#if defined(KOKKOS_ENABLE_OPENMP)
template <typename ExecSpace>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, Kokkos::Random_XorShift1024_Pool< Kokkos::OpenMP >>::type
GetKokkosRandom1024Pool() {
  {
    std::lock_guard<Uintah::MasterLock> rand_init_mutex_guard(rand_init_mutex);
    if (!openMPRandomPool) {
      std::unique_ptr<KokkosRandom< Kokkos::Random_XorShift1024_Pool< Kokkos::OpenMP >>> temp( new KokkosRandom< Kokkos::Random_XorShift1024_Pool< Kokkos::OpenMP >>(true) );
      openMPRandomPool = std::move(temp);
    }
  }
  return openMPRandomPool->getRandPool();
}
#endif


#if defined(KOKKOS_ENABLE_CUDA)
template <typename ExecSpace>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, Kokkos::Random_XorShift1024_Pool< Kokkos::Cuda >>::type
GetKokkosRandom1024Pool() {
  {
    std::lock_guard<Uintah::MasterLock> rand_init_mutex_guard(rand_init_mutex);
    if (!cudaRandomPool) {
      std::unique_ptr<KokkosRandom< Kokkos::Random_XorShift1024_Pool< Kokkos::Cuda >>> temp( new KokkosRandom< Kokkos::Random_XorShift1024_Pool< Kokkos::Cuda >>(true) );
      cudaRandomPool = std::move(temp);
    }
  }
  return cudaRandomPool->getRandPool();
}
#endif

} // end namespace Uintah

#endif // UINTAH_ENABLE_KOKKOS 
#endif // UINTAH_CORE_KOKKOS_TOOLS_H
