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

#ifndef UINTAH_HOMEBREW_ARRAY1_H
#define UINTAH_HOMEBREW_ARRAY1_H

#include <Core/Grid/Variables/BlockRange.hpp>

#include <Core/Grid/Variables/Array1Window.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/Stencil4.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Math/Matrix3.h>

#include <Core/Util/Endian.h>
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
  Array1

  GENERAL INFORMATION

  Array1.h

  Steven G. Parker
  Department of Computer Science
  University of Utah

  Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


  KEYWORDS
  Array1

  DESCRIPTION
  Long description...

  WARNING

 ****************************************/

template<class T> class Array1
{
public:
  typedef T value_type;

  Array1() {}

  Array1(int size1) {
    d_window=scinew Array1Window<T>(new Array1Data<T>( int(size1) ));
    d_window->addReference();
#if defined(UINTAH_ENABLE_KOKKOS)
    if (d_window) {
      m_view = d_window->getKokkosView();
    }
#endif
  }
  Array1(const int& lowIndex, const int& highIndex) {
    d_window = 0;
    resize(lowIndex,highIndex);
  }
  Array1(const Array1& copy)
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
  Array1(Array1Window<T>* window)
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
  virtual ~Array1()
  {
    if(d_window && d_window->removeReference())
    {
      delete d_window;
      d_window=0;
    }
  }

  void copyPointer(const Array1& copy) {
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

  int size() const {
    if (d_window)
      return d_window->getData()->size();
    else
      return int(0);
  }
  void initialize(const T& value) {
    d_window->initialize(value);
  }
  void copy(const Array1<T>& from) {
    ASSERT(d_window != 0);
    ASSERT(from.d_window != 0);
    d_window->copy(from.d_window);
  }

  void copy(const Array1<T>& from, const int& low, const int& high) {
    ASSERT(d_window != 0);
    ASSERT(from.d_window != 0);
    d_window->copy(from.d_window, low, high);
  }

  void initialize(const T& value, const int& s,
      const int& e) {
    d_window->initialize(value, s, e);
  }

  void resize(const int& lowIndex, const int& highIndex) {
    if(d_window && d_window->removeReference())
    {
      delete d_window;
      d_window=0;
    }
    int size = highIndex-lowIndex;
    d_window=scinew Array1Window<T>(new Array1Data<T>(size), lowIndex, lowIndex, highIndex);
    d_window->addReference();
#if defined(UINTAH_ENABLE_KOKKOS)
    if (d_window) {
      m_view = d_window->getKokkosView();
    }
#endif
  }

  void offset(const int offset) {
    Array1Window<T>* old_window = d_window;
    d_window=scinew Array1Window<T>(d_window->getData(), d_window->getOffset() + offset, getLowIndex() + offset, getHighIndex() + offset);
    d_window->addReference();
    if(old_window && old_window->removeReference())
    {
      delete old_window;
      old_window=0;
    }
  }

  // return true iff no reallocation is needed
  bool rewindow(const int& lowIndex, const int& highIndex);

  inline const Array1Window<T>* getWindow() const {
    return d_window;
  }
  inline Array1Window<T>* getWindow() {
    return d_window;
  }

#ifdef UINTAH_ENABLE_KOKKOS
  inline KokkosView1<T> getKokkosView() const
  {
    return m_view;
  }

  KOKKOS_FORCEINLINE_FUNCTION
    const T& operator[](const int& idx) const
    {
      return m_view(idx);
    }

  KOKKOS_FORCEINLINE_FUNCTION
    T& operator[](const int& idx)
    {
      return m_view(idx);
    }

  KOKKOS_FORCEINLINE_FUNCTION
    const T& operator()(int i) const
    {
      return m_view(i);
    }

  KOKKOS_FORCEINLINE_FUNCTION
    T& operator()(int i)
    {
      return m_view(i);
    }
#else
  inline const T& operator[](const int& idx) const {
    return d_window->get(idx);
  }

  inline T& operator[](const int& idx) {
    return d_window->get(idx);
  }

  inline const T& operator()(int i) const {
    return (*this)[int(i)];
  }

  inline T& operator()(int i) {
    return d_window->get(i);
  }
#endif

  BlockRange range() const
  {
    return BlockRange{getLowIndex(), getHighIndex()};
  }

  int getLowIndex() const {
    ASSERT(d_window != 0);
    return d_window->getLowIndex();
  }

  int getHighIndex() const {
    ASSERT(d_window != 0);
    return d_window->getHighIndex();
  }

  ///////////////////////////////////////////////////////////////////////
  // Get low index for fortran calls
  inline int getFortLowIndex() const {
    return d_window->getLowIndex();
  }

  ///////////////////////////////////////////////////////////////////////
  // Get high index for fortran calls
  inline int getFortHighIndex() const {
    return d_window->getHighIndex()-int(1);
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

  inline void write(std::ostream& out, const int& l, const int& h, bool /*outputDoubleAsFloat*/ )
  {
    // This could be optimized...
    ssize_t linesize = (ssize_t)(sizeof(T)*(h-l));
    out.write((char*)&(*this)[int(l)], linesize);
  }
 

  inline void read(std::istream& in, bool swapBytes)
  {
    // This could be optimized...
    int l(getLowIndex());
    int h(getHighIndex());
    ssize_t linesize = (ssize_t)(sizeof(T)*(h-l));
    in.read((char*)&(*this)[int(l)], linesize);
    if (swapBytes) {
      for (int x=l;x<h;x++) {
	Uintah::swapbytes((*this)[int(x)]);
      }
    }
  }


  void print(std::ostream& out) const {
    using std::endl;
    int l = d_window->getLowIndex();
    int h = d_window->getHighIndex();
    out << "Variable from " << l << " to " << h << '\n';
    for (int ii = l; ii < h; ii++) {
      out << "variable for ii = " << ii << endl;
      out.width(10);
      out << (*this)[int(ii)] << " " ;
    }
    out << endl;
  }


protected:
  Array1& operator=(const Array1& copy);

private:
  Array1Window<T>* d_window{nullptr};
#if defined(UINTAH_ENABLE_KOKKOS)
  KokkosView1<T> m_view{};
#endif
};

// return true iff no reallocation is needed
template <class T>
bool Array1<T>::rewindow(const int& lowIndex,
    const int& highIndex) {
  if (!d_window) {
    resize(lowIndex, highIndex);
    return false; // reallocation needed
  }
  bool inside = true;
  int relLowIndex = lowIndex - d_window->getOffset();
  int relHighIndex = highIndex - d_window->getOffset();
  int size = d_window->getData()->size();
  
  ASSERT(relLowIndex < relHighIndex);
  if ((relLowIndex < 0) || (relHighIndex > size)) {
    inside = false;
  }
  Array1Window<T>* oldWindow = d_window;
  bool no_reallocation_needed = false;
  if (inside) {
    // just rewindow
    d_window=
      scinew Array1Window<T>(oldWindow->getData(), oldWindow->getOffset(),
          lowIndex, highIndex);
    no_reallocation_needed = true;
  }
  else {
    // will have to re-allocate and copy
    int encompassingLow = std::min(lowIndex, oldWindow->getLowIndex());
    int encompassingHigh = std::max(highIndex, oldWindow->getHighIndex());

    Array1Data<T>* newData =
      new Array1Data<T>(encompassingHigh - encompassingLow);
    Array1Window<T> tempWindow(newData, encompassingLow,
        oldWindow->getLowIndex(),
        oldWindow->getHighIndex());
    tempWindow.copy(oldWindow); // copies into newData

    Array1Window<T>* new_window=
      scinew Array1Window<T>(newData, encompassingLow, lowIndex,highIndex);
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
inline void Array1<double>::write(std::ostream& out, const int& l, const int& h, bool outputDoubleAsFloat)
{
  // This could be optimized...
  if (outputDoubleAsFloat) {
    for(int x=l;x<h;x++){
      float tempFloat = static_cast<float>((*this)[int(x)]);
      out.write((char*)&tempFloat, sizeof(float));
    }
  } else {
    ssize_t linesize = (ssize_t)(sizeof(double)*(h-l));
    out.write((char*)&(*this)[int(l)], linesize);
  }

}

} // End namespace Uintah

#endif
