/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_HOMEBREW_Array1Window_H
#define UINTAH_HOMEBREW_Array1Window_H

#include <Core/Util/RefCounted.h>
#include <Core/Grid/Variables/Array1Data.h>
#include <climits>

#include <sci_defs/kokkos_defs.h>

#if SCI_ASSERTION_LEVEL >= 3
// test the range and throw a more informative exception
// instead of testing one index's range
#include <Core/Exceptions/InternalError.h>
#endif

#include <type_traits>

/**************************************

CLASS
   Array1Window

GENERAL INFORMATION

   Array1Window.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   Array1Window

DESCRIPTION

   The Array1Window class supports a windowed access into memory.  For
   multiple patches per process, this allows one large block of memory
   to be accessed by each patch with local 0-based access from the
   patch.  The offset contained herein is a GLOBAL offset.

WARNING

****************************************/

namespace Uintah {

#ifdef UINTAH_ENABLE_KOKKOS
template <typename T>
struct KokkosView1
{
  using view_type = Kokkos::View<T*, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  using reference_type = typename view_type::reference_type;

  template< typename IType >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type operator()(const IType & i) const
  { return m_view( i - m_i ); }

  KokkosView1( const view_type & v, int i )
    : m_view(v)
    , m_i(i)
  {}

  KokkosView1() = default;

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView1( const KokkosView1<U> & v)
    : m_view(v.m_view)
    , m_i(v.m_i)
  {}

  template <typename U, typename = std::enable_if< std::is_same<U,T>::value || std::is_same<const U,T>::value> >
  KokkosView1 & operator=( const KokkosView1<U> & v)
  {
    m_view = v.m_view;
    m_i = v.m_i;
    return *this;
  }

  view_type m_view;
  int       m_i;
};
#endif //UINTAH_ENABLE_KOKKOS


template<class T> class Array1Window : public RefCounted {
   public:
      Array1Window(Array1Data<T>*);
      Array1Window(Array1Data<T>*, const int& offset,
                   const int& lowIndex, const int& highIndex);
      virtual ~Array1Window();

      inline const Array1Data<T>* getData() const {
         return data;
      }

      inline Array1Data<T>* getData() {
         return data;
      }

      void copy(const Array1Window<T>*);
      void copy(const Array1Window<T>*, const int& low, const int& high);
      void initialize(const T&);
      void initialize(const T&, const int& s, const int& e);
      inline int getLowIndex() const {
         return lowIndex;
      }
      inline int getHighIndex() const {
         return highIndex;
      }
      inline int getOffset() const {
         return offset;
      }
#if 0
      inline T& get(const int& idx) {
         ASSERT(data);
#if SCI_ASSERTION_LEVEL >= 3
          // used to be several CHECKARRAYBOUNDS, but its lack of information
          // has bitten me a few times.... BJW
          bool bad = false;
          if (idx < lowIndex || idx >= highIndex) bad = true;
          if (bad) {
            std::ostringstream ostr;
            ostr << "Index not in range of window (on get): index: " << idx << " window low "
                 << lowIndex << " window high " << highIndex;
            throw Uintah::InternalError(ostr.str(), __FILE__, __LINE__);
          }
#endif
         return data->get(idx-offset);
      }
#endif
#if 1
      inline T& get(int i) {
        return data->get(i-offset);
      }
#endif

#ifdef UINTAH_ENABLE_KOKKOS
      inline KokkosView1<T> getKokkosView() const
      {
        return KokkosView1<T>(  Kokkos::subview(   data->getKokkosData()
                                 , Kokkos::pair<int,int>( lowIndex - offset, highIndex - offset)
                              ,offset
                            );
      }
#endif //UINTAH_ENABLE_KOKKOS

      ///////////////////////////////////////////////////////////////////////
      // Return pointer to the data
      // (**WARNING**not complete implementation)
      inline T* getPointer() {
        return data ? (data->getPointer()) : 0;
      }

      ///////////////////////////////////////////////////////////////////////
      // Return const pointer to the data
      // (**WARNING**not complete implementation)
      inline const T* getPointer() const {
        return (data->getPointer());
      }

   private:

      Array1Data<T>* data;
      int offset;
      int lowIndex;
      int highIndex;
      Array1Window(const Array1Window<T>&);
      Array1Window<T>& operator=(const Array1Window<T>&);
   };

   template<class T>
      void Array1Window<T>::initialize(const T& val)
      {
         data->initialize(val, lowIndex-offset, highIndex-offset);
      }

   template<class T>
      void Array1Window<T>::copy(const Array1Window<T>* from)
      {
         data->copy(lowIndex-offset, highIndex-offset, from->data,
                    from->lowIndex-from->offset, from->highIndex-from->offset);
      }

   template<class T>
      void Array1Window<T>::copy(const Array1Window<T>* from,
                                 const int& low, const int& high)
      {
         data->copy(low-offset, high-offset, from->data,
                    low-from->offset, high-from->offset);
      }

   template<class T>
      void Array1Window<T>::initialize(const T& val,
                                       const int& s,
                                       const int& e)
      {
         CHECKARRAYBOUNDS(s, lowIndex, highIndex);
         CHECKARRAYBOUNDS(e, s, highIndex+1);
         data->initialize(val, s-offset, e-offset);
      }

   template<class T>
      Array1Window<T>::Array1Window(Array1Data<T>* data)
      : data(data), offset(0), lowIndex(0), highIndex(data->size())
      {
         data->addReference();
      }

   template<class T>
      Array1Window<T>::Array1Window(Array1Data<T>* data,
                                    const int& offset,
                                    const int& lowIndex,
                                    const int& highIndex)
      : data(data), offset(offset), lowIndex(lowIndex), highIndex(highIndex)
      {
        // null data can be used for a place holder in OnDemandDataWarehouse
        if (data != 0) {
#if SCI_ASSERTION_LEVEL >= 3
          // used to be several CHECKARRAYBOUNDS, but its lack of information
          // has bitten me a few times.... BJW
          int low(lowIndex-offset);
          int high(highIndex-offset);
          bool bad = false;
          if (low < 0 || low >= data->size()) bad = true;
          if (high < 0 || high > data->size()) bad = true;
          if (bad) {
            std::ostringstream ostr;
            ostr << "Data not in range of new window: data size: " << data->size() << " window low "
                 << lowIndex << " window high " << highIndex;
            throw Uintah::InternalError(ostr.str(), __FILE__, __LINE__);
          }
#endif
          data->addReference();
        }
        else {
          // To use null data, put the offset as {INT_MAX, INT_MAX, INT_MAX}.
          // This way, when null is used accidentally, this assertion will
          // fail while allowing purposeful (and hopefully careful) uses
          // of null data.
          ASSERT(offset == int(INT_MAX));
        }
      }

   template<class T>
      Array1Window<T>::~Array1Window()
      {
        if(data && data->removeReference())
        {
          delete data;
          data=0;
        }
      }
} // End namespace Uintah

#endif
