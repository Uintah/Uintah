#ifndef UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#define UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

namespace Uintah {

//template <typename T, typename Space>
//using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;


template <typename T>
using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

//For the default memory space
template <typename T>
struct KokkosView3
{
  using view_type = Kokkos::View<T***, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  using reference_type = typename view_type::reference_type;

  template< typename IType, typename JType, typename KType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i, const JType & j, const KType & k ) const
  { return m_view( i - m_i, j - m_j, k - m_k ); }

  KokkosView3( const view_type & v, int i, int j, int k )
    : m_view(v)
    , m_i(i)
    , m_j(j)
    , m_k(k)
  {}

  KokkosView3() = default;

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3( const KokkosView3<U> & v)
    : m_view(v.m_view)
    , m_i(v.m_i)
    , m_j(v.m_j)
    , m_k(v.m_k)
  {}

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3 & operator=( const KokkosView3<U> & v)
  {
    m_view = v.m_view;
    m_i = v.m_i;
    m_j = v.m_j;
    m_k = v.m_k;
    return *this;
  }

  view_type m_view;
  int       m_i{0};
  int       m_j{0};
  int       m_k{0};
};

} // End namespace Uintah
#endif //UINTAH_ENABLE_KOKKOS
#endif //UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#ifndef UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#define UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

namespace Uintah {

//template <typename T, typename Space>
//using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;


template <typename T>
using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

//For the default memory space
template <typename T>
struct KokkosView3
{
  using view_type = Kokkos::View<T***, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  using reference_type = typename view_type::reference_type;

  template< typename IType, typename JType, typename KType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i, const JType & j, const KType & k ) const
  { return m_view( i - m_i, j - m_j, k - m_k ); }

  KokkosView3( const view_type & v, int i, int j, int k )
    : m_view(v)
    , m_i(i)
    , m_j(j)
    , m_k(k)
  {}

  KokkosView3() = default;

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3( const KokkosView3<U> & v)
    : m_view(v.m_view)
    , m_i(v.m_i)
    , m_j(v.m_j)
    , m_k(v.m_k)
  {}

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3 & operator=( const KokkosView3<U> & v)
  {
    m_view = v.m_view;
    m_i = v.m_i;
    m_j = v.m_j;
    m_k = v.m_k;
    return *this;
  }

  view_type m_view;
  int       m_i;
  int       m_j;
  int       m_k;
};

} // End namespace Uintah
#endif //UINTAH_ENABLE_KOKKOS
#endif //UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#ifndef UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#define UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

namespace Uintah {

//template <typename T, typename Space>
//using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;


template <typename T>
using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

//For the default memory space
template <typename T>
struct KokkosView3
{
  using view_type = Kokkos::View<T***, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  using reference_type = typename view_type::reference_type;

  template< typename IType, typename JType, typename KType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i, const JType & j, const KType & k ) const
  { return m_view( i - m_i, j - m_j, k - m_k ); }

  KokkosView3( const view_type & v, int i, int j, int k )
    : m_view(v)
    , m_i(i)
    , m_j(j)
    , m_k(k)
  {}

  KokkosView3() = default;

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3( const KokkosView3<U> & v)
    : m_view(v.m_view)
    , m_i(v.m_i)
    , m_j(v.m_j)
    , m_k(v.m_k)
  {}

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3 & operator=( const KokkosView3<U> & v)
  {
    m_view = v.m_view;
    m_i = v.m_i;
    m_j = v.m_j;
    m_k = v.m_k;
    return *this;
  }

  view_type m_view;
  int       m_i;
  int       m_j;
  int       m_k;
};

} // End namespace Uintah
#endif //UINTAH_ENABLE_KOKKOS
#endif //UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#ifndef UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#define UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

namespace Uintah {

//template <typename T, typename Space>
//using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;


template <typename T>
using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

//For the default memory space
template <typename T>
struct KokkosView3
{
  using view_type = Kokkos::View<T***, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  using reference_type = typename view_type::reference_type;

  template< typename IType, typename JType, typename KType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i, const JType & j, const KType & k ) const
  { return m_view( i - m_i, j - m_j, k - m_k ); }

  KokkosView3( const view_type & v, int i, int j, int k )
    : m_view(v)
    , m_i(i)
    , m_j(j)
    , m_k(k)
  {}

  KokkosView3() = default;

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3( const KokkosView3<U> & v)
    : m_view(v.m_view)
    , m_i(v.m_i)
    , m_j(v.m_j)
    , m_k(v.m_k)
  {}

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3 & operator=( const KokkosView3<U> & v)
  {
    m_view = v.m_view;
    m_i = v.m_i;
    m_j = v.m_j;
    m_k = v.m_k;
    return *this;
  }

  view_type m_view;
  int       m_i;
  int       m_j;
  int       m_k;
};

} // End namespace Uintah
#endif //UINTAH_ENABLE_KOKKOS
#endif //UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#ifndef UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
#define UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

namespace Uintah {

//template <typename T, typename Space>
//using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;


template <typename T>
using KokkosData = Kokkos::View<T***, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

//For the default memory space
template <typename T>
struct KokkosView3
{
  using view_type = Kokkos::View<T***, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  using reference_type = typename view_type::reference_type;

  template< typename IType, typename JType, typename KType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i, const JType & j, const KType & k ) const
  { return m_view( i - m_i, j - m_j, k - m_k ); }

  KokkosView3( const view_type & v, int i, int j, int k )
    : m_view(v)
    , m_i(i)
    , m_j(j)
    , m_k(k)
  {}

  KokkosView3() = default;

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3( const KokkosView3<U> & v)
    : m_view(v.m_view)
    , m_i(v.m_i)
    , m_j(v.m_j)
    , m_k(v.m_k)
  {}

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView3 & operator=( const KokkosView3<U> & v)
  {
    m_view = v.m_view;
    m_i = v.m_i;
    m_j = v.m_j;
    m_k = v.m_k;
    return *this;
  }

  view_type m_view;
  int       m_i;
  int       m_j;
  int       m_k;
};

} // End namespace Uintah
#endif //UINTAH_ENABLE_KOKKOS
#endif //UINTAH_GRID_VARIABLES_KOKKOSVIEWS_H
