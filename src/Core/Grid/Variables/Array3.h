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

#ifndef UINTAH_HOMEBREW_ARRAY3_H
#define UINTAH_HOMEBREW_ARRAY3_H

#include <Core/Grid/Variables/Array3Window.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Parallel/LoopExecution.hpp>
#include <Core/Exceptions/InternalError.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Int130.h>

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
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
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
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
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
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
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
    if(copy.d_window){
      copy.d_window->addReference();
    }
    if(d_window && d_window->removeReference())
    {
      delete d_window;
      d_window=0;
    }
    d_window = copy.d_window;
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
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
  template <typename ExecSpace = Kokkos::OpenMP>
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
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
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
  bool rewindowExact(const IntVector& lowIndex, const IntVector& highIndex);

  inline const Array3Window<T>* getWindow() const {
    return d_window;
  }
  inline Array3Window<T>* getWindow() {
    return d_window;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  inline KokkosView3<T, Kokkos::HostSpace> getKokkosView() const
  {
    // Kokkos Views don't reference count, but OnDemand Data Warehouse's GridVariables will clean themselves up
    // when it goes out of scope and the ref count hits zero.  Uintah's KokkosView3 API means that the GridVariables
    // are out of scope but the KokkosView3 remains.  So we have the KokkosView3 also manage Array3Data ref counting.
	  if (!m_view.m_A3Data) {
      m_view.m_A3Data = d_window->getData();
      m_view.m_A3Data->addReference();
	  }
    return m_view;
  }
#endif

//For now, if it's a homogeneous only Kokkos environment, use Kokkos Views
//If it's a legacy environment or a CUDA environment, use the original way of accessing data.
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP ) && !defined( HAVE_CUDA )

  //Note: Dan Sunderland used a Kokkos define KOKKOS_FORCEINLINE_FUNCTION,
  //however, this caused problems when trying to compile with CUDA, as it tried
  //to put a __device__ __host__ header (needed for GPU builds)
  //instead of __attribute__((always_inline)) (which makes sense in a CPU build as
  //Array3.h won't ever run on the GPU).
  //I couldn't find a way to ask Kokkos to be smart and choose a non-CUDA version
  //here, so I'll just explicitly provide the gcc one as this will never be compiled
  //as CUDA code.  Brad Peterson Nov 23 2017
  //KOKKOS_FORCEINLINE_FUNCTION
  __attribute__((always_inline))
    T& operator[](const IntVector& idx) const
    {
      return m_view(idx[0],idx[1],idx[2]);
    }

  //KOKKOS_FORCEINLINE_FUNCTION
  __attribute__((always_inline))
    T& operator[](const IntVector& idx)
    {
      return m_view(idx[0],idx[1],idx[2]);
    }

  //KOKKOS_FORCEINLINE_FUNCTION
  __attribute__((always_inline))
    T& operator()(int i, int j, int k) const
    {
      return m_view(i,j,k);
    }

  //KOKKOS_FORCEINLINE_FUNCTION
  __attribute__((always_inline))
    T& operator()(int i, int j, int k)
    {
      return m_view(i,j,k);
    }
#else

  inline T& operator[](const IntVector& idx) const {
    return d_window->get(idx);
  }

  inline T& operator[](const IntVector& idx) {
    return d_window->get(idx);
  }

  inline T& operator()(int i, int j, int k) const {
    return d_window->get(i,j,k);
  }

  inline T& operator()(int i, int j, int k) {
    return d_window->get(i,j,k);
  }
#endif

  inline T& get(const IntVector& idx) const {
    return d_window->get(idx);
  }

  inline T& get(int i, int j, int k) const {
    return d_window->get(i,j,k);
  }

  inline T& get(int i, int j, int k) {
    return d_window->get(i,j,k);
  }
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
  // These two data members are marked as mutable due to a need for lambdas.
  // When Grid Variables are lambda captured with [=], they are captured as *const*.
  // But we need to let grid variables be modified, and so we set these data members as
  // mutable, which gets around the const.
  mutable Array3Window<T>* d_window{nullptr};
#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  //Array3 variables should never go outside of HostSpace.
  mutable KokkosView3<T, Kokkos::HostSpace> m_view{};
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

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  if (d_window) {
    m_view = d_window->getKokkosView();
  }
#endif

  return no_reallocation_needed;
}

// return true iff no reallocation is needed

//DS 06162020 Added logic to rewindowExact. Ensures the allocated space has exactly same size as the requested. This is needed for D2H copy.
//Check comments in OnDemandDW::allocateAndPut, OnDemandDW::getGridVar, Array3<T>::rewindowExact and UnifiedScheduler::initiateD2H
//TODO: Throwing error if allocated and requested spaces are not same might be a problem for RMCRT. Fix can be to create a temporary
//variable (buffer) in UnifiedScheduler for D2H copy and then copy from buffer to actual variable. But lets try this solution first.
template <class T>
bool Array3<T>::rewindowExact(const IntVector& lowIndex, const IntVector& highIndex) {
  if (!d_window) {
    resize(lowIndex, highIndex);
    return false; // reallocation needed
  }
  bool match = true;
  IntVector relLowIndex = lowIndex - d_window->getOffset();
  IntVector relHighIndex = highIndex - d_window->getOffset();
  IntVector size = d_window->getData()->size();
  for (int i = 0; i < 3; i++) {
    ASSERT(relLowIndex[i] < relHighIndex[i]);
    if ((relLowIndex[i] != 0) || (relHighIndex[i] != size[i])) {
      match = false;
      break;
    }
  }
  Array3Window<T>* oldWindow = d_window;
  bool no_reallocation_needed = false;
  if (match) {
    d_window=scinew Array3Window<T>(oldWindow->getData(), oldWindow->getOffset(), lowIndex, highIndex);
    no_reallocation_needed = true;
  }
  else {
    // will have to re-allocate and copy
    IntVector offset = oldWindow->getOffset();
    IntVector oldHigh = oldWindow->getHighIndex();
    printf("### Error. Allocated size does not exactly match with the requested size. allocated: offset %d %d %d high: %d %d %d requested: low %d %d %d high  %d %d %d\n",
           offset[0], offset[1], offset[2], oldHigh[0], oldHigh[1], oldHigh[2],
           lowIndex[0], lowIndex[1], lowIndex[2], highIndex[0], highIndex[1], highIndex[2]);
  }
  d_window->addReference();
  if(oldWindow->removeReference())
  {
    delete oldWindow;
    oldWindow=0;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  if (d_window) {
    m_view = d_window->getKokkosView();
  }
#endif

  // return true iff no reallocation is needed
  return no_reallocation_needed;
}



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
