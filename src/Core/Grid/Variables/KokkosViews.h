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
 
 #ifndef UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#define UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H

#if defined( UINTAH_ENABLE_KOKKOS )
#include <Kokkos_Core.hpp>
#include <Core/Parallel/LoopExecution.hpp>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/Array3Data.h>

namespace Uintah {

template <typename T>
class Array3Data;

template <typename T, typename MemSpace>
using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;


//For the default memory space
template <typename T, typename MemSpace>
class KokkosView3
{
public:

  ~KokkosView3() {

    if( m_A3Data && m_A3Data->removeReference())  // Race condition
    {
      delete m_A3Data;
      m_A3Data = nullptr;
    }
  }
  using view_type = Kokkos::View<T***, Kokkos::LayoutStride, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  using reference_type = typename view_type::reference_type;

  template< typename IType, typename JType, typename KType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i, const JType & j, const KType & k ) const
  { return m_view( i - m_i, j - m_j, k - m_k ); }

  template< typename IType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i ) const
  { return m_view( i, 0, 0 ); }

  KokkosView3( const view_type & v, int i, int j, int k, Array3Data<T>* A3Data)
    : m_view(v)
    , m_i(i)
    , m_j(j)
    , m_k(k)
    , m_A3Data(A3Data)
  {
    if (this->m_A3Data) {  // These two lines are currently an OnDemand DW race condition
      this->m_A3Data->addReference();
    }
  }

  KokkosView3() = default;

  //template <typename U, typename MemSpaceSource,
  //          typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value >,
  //          typename = std::enable_if< std::is_same<MemSpaceSource, MemSpace>::value > >
  //KokkosView3( const KokkosView3<U, MemSpaceSource> & v)
  KokkosView3( const KokkosView3<T, MemSpace> & v)
    : m_view(v.m_view)
    , m_i(v.m_i)
    , m_j(v.m_j)
    , m_k(v.m_k)
  {
    if( this->m_A3Data && this->m_A3Data->removeReference())  // Race condition
    {
      delete this->m_A3Data;
      this->m_A3Data = nullptr;
    }
    this->m_A3Data = v.m_A3Data;  // Copy the pointer
    if (this->m_A3Data) {  // Race condition
      this->m_A3Data->addReference();
    }
  }

  //template <typename U, typename MemSpaceSource,
  //          typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value >,
  //          typename = std::enable_if< std::is_same<MemSpaceSource, MemSpace>::value > >
  //KokkosView3 & operator=( const KokkosView3<U, MemSpaceSource> & v)
  KokkosView3 & operator=( const KokkosView3<T, MemSpace> & v)
  {
    m_view = v.m_view;
    m_i = v.m_i;
    m_j = v.m_j;
    m_k = v.m_k;
    if( this->m_A3Data && this->m_A3Data->removeReference())  // Race condition
    {
      delete this->m_A3Data;
      this->m_A3Data = nullptr;
    }
    this->m_A3Data = v.m_A3Data;  // Copy the pointer
    if (this->m_A3Data) {  // Race condition
      this->m_A3Data->addReference();
    }
    return *this;
  }


  template <typename ExecSpace>
  inline  void
  initialize( T init_val){
    Uintah::parallel_for<ExecSpace>(*this,init_val );
  }

  view_type m_view;
  int       m_i{0};
  int       m_j{0};
  int       m_k{0};

  Array3Data<T>* m_A3Data{nullptr};   // Uintah's Host Memory (OnDemand Data Warehouse) grid variables can be created on-the-fly
                                    // when ghost cells are requested, and those grid variables (and the underlying Array3Data
                                    // d_data) will clean themselves up once they go out of scope.  However, with Kokkos, these
                                    // were going out of scope before the task loops started.  So KokkosView3 needs to also
                                    // take on responsibility for Array3Data's ref counting so that when the KokkosView3 is done
                                    // and goes out of scope, the KokkosView3 can decrement the Array3Data letting it finally
                                    // deallocate d_data if needed.
                                    // If something like the GPU Data Warehouse, which doesn't use Array3Data, uses KokkosView3,
                                    // then just keep this boolean false and the d_data pointer nullptr


};

template <typename T>
struct KokkosView3<const Uintah::Patch *, T>
{

};

} // End namespace Uintah
#endif //UINTAH_ENABLE_KOKKOS
#endif //UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
