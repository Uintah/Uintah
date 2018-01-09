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

#ifndef UINTAH_HOMEBREW_ARRAY3_H
#define UINTAH_HOMEBREW_ARRAY3_H

#include <Core/Grid/Variables/BlockRange.hpp>

#include <Core/Grid/Variables/Array3Window.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/Stencil4.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Math/Matrix3.h>

#include <Core/Util/Endian.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>

#include <iosfwd>

#include <type_traits>

#include <sci_defs/kokkos_defs.h>

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif //UINTAH_ENABLE_KOKKOS

namespace Uintah {


/**************************************

  CLASS
  Array3

  GENERAL INFORMATION

  Array3.h

  Steven G. Parker
  Department of Computer Science
  University of Utah

  Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


  KEYWORDS
  Array3

  DESCRIPTION
  Long description...

  WARNING

 ****************************************/

template<class T> class Array3
{
public:
  typedef T value_type;

  Array3() {}

  Array3(int size1, int size2, int size3) {
    d_window=scinew Array3Window<T>(new Array3Data<T>( IntVector(size1, size2, size3) ));
    d_window->addReference();
#if defined(UINTAH_ENABLE_KOKKOS)
    if (d_window) {
      m_view = d_window->getKokkosView();
    }
#endif
  }
  Array3(const IntVector& lowIndex, const IntVector& highIndex) {
    d_window = 0;
    resize(lowIndex,highIndex);
  }
  Array3(const Array3& copy)
    : d_window(copy.d_window)
  {
    if(d_window) {
      d_window->addReference();
    }
#if defined(UINTAH_ENABLE_KOKKOS)
    if (d_window) {
      m_view = d_window->getKokkosView();
    }
#endif
  }
  Array3(Array3Window<T>* window)
    : d_window(window)
  {
    if(d_window) {
      d_window->addReference();
    }
#if defined(UINTAH_ENABLE_KOKKOS)
    if (d_window) {
      m_view = d_window->getKokkosView();
    }
#endif
  }
  virtual ~Array3()
  {
    if(d_window && d_window->removeReference())
    {
      delete d_window;
      d_window=0;
    }
  }

  void copyPointer(const Array3& copy) {
    if(copy.d_window)
      copy.d_window->addReference();
    if(d_window && d_window->removeReference())
    {
      delete d_window;
      d_window=0;
    }
    d_window = copy.d_window;
#if defined(UINTAH_ENABLE_KOKKOS)
    if (d_window) {
      m_view = d_window->getKokkosView();
    }
#endif
  }

  IntVector size() const {
    if (d_window)
      return d_window->getData()->size();
    else
      return IntVector(0,0,0);
  }
  void initialize(const T& value) {
    d_window->initialize(value);
  }
  void copy(const Array3<T>& from) {
    ASSERT(d_window != 0);
    ASSERT(from.d_window != 0);
    d_window->copy(from.d_window);
  }

  void copy(const Array3<T>& from, const IntVector& low, const IntVector& high) {
    ASSERT(d_window != 0);
    ASSERT(from.d_window != 0);
    d_window->copy(from.d_window, low, high);
  }

  void initialize(const T& value, const IntVector& s,
      const IntVector& e) {
    d_window->initialize(value, s, e);
  }

  void resize(const IntVector& lowIndex, const IntVector& highIndex) {
    if(d_window && d_window->removeReference())
    {
      delete d_window;
      d_window=0;
    }
    IntVector size = highIndex-lowIndex;
    d_window=scinew Array3Window<T>(new Array3Data<T>(size), lowIndex, lowIndex, highIndex);
    d_window->addReference();
#if defined(UINTAH_ENABLE_KOKKOS)
    if (d_window) {
      m_view = d_window->getKokkosView();
    }
#endif
  }

  void offset(const IntVector offset) {
    Array3Window<T>* old_window = d_window;
    d_window=scinew Array3Window<T>(d_window->getData(), d_window->getOffset() + offset, getLowIndex() + offset, getHighIndex() + offset);
    d_window->addReference();
    if(old_window && old_window->removeReference())
    {
      delete old_window;
      old_window=0;
    }
  }

  // return true iff no reallocation is needed
  bool rewindow(const IntVector& lowIndex, const IntVector& highIndex);

  inline const Array3Window<T>* getWindow() const {
    return d_window;
  }
  inline Array3Window<T>* getWindow() {
    return d_window;
  }

#ifdef UINTAH_ENABLE_KOKKOS
  inline KokkosView3<T> getKokkosView() const
  {
    return m_view;
  }

  KOKKOS_FORCEINLINE_FUNCTION
    const T& operator[](const IntVector& idx) const
    {
      return m_view(idx[0],idx[1],idx[2]);
    }

  KOKKOS_FORCEINLINE_FUNCTION
    T& operator[](const IntVector& idx)
    {
      return m_view(idx[0],idx[1],idx[2]);
    }

  KOKKOS_FORCEINLINE_FUNCTION
    const T& operator()(int i, int j, int k) const
    {
      return m_view(i,j,k);
    }

  KOKKOS_FORCEINLINE_FUNCTION
    T& operator()(int i, int j, int k)
    {
      return m_view(i,j,k);
    }
#else
  inline const T& operator[](const IntVector& idx) const {
    return d_window->get(idx);
  }

  inline T& operator[](const IntVector& idx) {
    return d_window->get(idx);
  }

  inline const T& operator()(int i, int j, int k) const {
    return (*this)[IntVector(i,j,k)];
  }

  inline T& operator()(int i, int j, int k) {
    return d_window->get(i,j,k);
  }
#endif

  BlockRange range() const
  {
    return BlockRange{getLowIndex(), getHighIndex()};
  }

  IntVector getLowIndex() const {
    ASSERT(d_window != 0);
    return d_window->getLowIndex();
  }

  IntVector getHighIndex() const {
    ASSERT(d_window != 0);
    return d_window->getHighIndex();
  }

  ///////////////////////////////////////////////////////////////////////
  // Get low index for fortran calls
  inline IntVector getFortLowIndex() const {
    return d_window->getLowIndex();
  }

  ///////////////////////////////////////////////////////////////////////
  // Get high index for fortran calls
  inline IntVector getFortHighIndex() const {
    return d_window->getHighIndex()-IntVector(1,1,1);
  }

  ///////////////////////////////////////////////////////////////////////
  // Return pointer to the data
  // (**WARNING**not complete implementation)
  inline T* getPointer() {
    return (d_window->getPointer());
  }

  ///////////////////////////////////////////////////////////////////////
  // Return const pointer to the data
  // (**WARNING**not complete implementation)
  inline const T* getPointer() const {
    return (d_window->getPointer());
  }

  inline void write(std::ostream& out, const IntVector& l, const IntVector& h, bool /*outputDoubleAsFloat*/ )
  {
    // This could be optimized...
    ssize_t linesize = (ssize_t)(sizeof(T)*(h.x()-l.x()));
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        out.write((char*)&(*this)[IntVector(l.x(),y,z)], linesize);
      }
    }
  }

  inline void read(std::istream& in, bool swapBytes)
  {
    // This could be optimized...
    IntVector l(getLowIndex());
    IntVector h(getHighIndex());
    ssize_t linesize = (ssize_t)(sizeof(T)*(h.x()-l.x()));
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        in.read((char*)&(*this)[IntVector(l.x(),y,z)], linesize);
        if (swapBytes) {
          for (int x=l.x();x<h.x();x++) {
            Uintah::swapbytes((*this)[IntVector(x,y,z)]);
          }
        }
      }
    }
  }

  void print(std::ostream& out) const {
    using std::endl;
    IntVector l = d_window->getLowIndex();
    IntVector h = d_window->getHighIndex();
    out << "Variable from " << l << " to " << h << '\n';
    for (int ii = l.x(); ii < h.x(); ii++) {
      out << "variable for ii = " << ii << endl;
      for (int jj = l.y(); jj < h.y(); jj++) {
        for (int kk = l.z(); kk < h.z(); kk++) {
          out.width(10);
          out << (*this)[IntVector(ii,jj,kk)] << " " ;
        }
        out << endl;
      }
    }
  }

protected:
  Array3& operator=(const Array3& copy);

private:
  Array3Window<T>* d_window{nullptr};
#if defined(UINTAH_ENABLE_KOKKOS)
  KokkosView3<T> m_view{};
#endif
};

// return true iff no reallocation is needed
template <class T>
bool Array3<T>::rewindow(const IntVector& lowIndex,
    const IntVector& highIndex) {
  if (!d_window) {
    resize(lowIndex, highIndex);
    return false; // reallocation needed
  }
  bool inside = true;
  IntVector relLowIndex = lowIndex - d_window->getOffset();
  IntVector relHighIndex = highIndex - d_window->getOffset();
  IntVector size = d_window->getData()->size();
  for (int i = 0; i < 3; i++) {
    ASSERT(relLowIndex[i] < relHighIndex[i]);
    if ((relLowIndex[i] < 0) || (relHighIndex[i] > size[i])) {
      inside = false;
      break;
    }
  }
  Array3Window<T>* oldWindow = d_window;
  bool no_reallocation_needed = false;
  if (inside) {
    // just rewindow
    d_window=
      scinew Array3Window<T>(oldWindow->getData(), oldWindow->getOffset(),
          lowIndex, highIndex);
    no_reallocation_needed = true;
  }
  else {
    // will have to re-allocate and copy
    IntVector encompassingLow = Uintah::Min(lowIndex, oldWindow->getLowIndex());
    IntVector encompassingHigh = Uintah::Max(highIndex, oldWindow->getHighIndex());

    Array3Data<T>* newData =
      new Array3Data<T>(encompassingHigh - encompassingLow);
    Array3Window<T> tempWindow(newData, encompassingLow,
        oldWindow->getLowIndex(),
        oldWindow->getHighIndex());
    tempWindow.copy(oldWindow); // copies into newData

    Array3Window<T>* new_window=
      scinew Array3Window<T>(newData, encompassingLow, lowIndex,highIndex);
    d_window = new_window;  //Note, this has concurrency problems.
    //If two tasks running on two cores try to rewindow the same variable at the
    //same time, they could both write to d_window effectively at the same time
    //We hope a 64 bit write is atomic, but we're never sure exactly how
    //hardware will manage it.
    //Brad Peterson and Alan Humphrey June 15th 2015
  }
  d_window->addReference();
  if(oldWindow->removeReference())
  {
    delete oldWindow;
    oldWindow=0;
  }

#if defined(UINTAH_ENABLE_KOKKOS)
  if (d_window) {
    m_view = d_window->getKokkosView();
  }
#endif

  return no_reallocation_needed;
}

// return true iff no reallocation is needed
template <>
inline void Array3<double>::write(std::ostream& out, const IntVector& l, const IntVector& h, bool outputDoubleAsFloat)
{
  // This could be optimized...
  if (outputDoubleAsFloat) {
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        for(int x=l.x();x<h.x();x++){
          float tempFloat = static_cast<float>((*this)[IntVector(x,y,z)]);
          out.write((char*)&tempFloat, sizeof(float));
        }
      }
    }
  } else {
    ssize_t linesize = (ssize_t)(sizeof(double)*(h.x()-l.x()));
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        out.write((char*)&(*this)[IntVector(l.x(),y,z)], linesize);
      }
    }
  }
}

} // End namespace Uintah

#endif
