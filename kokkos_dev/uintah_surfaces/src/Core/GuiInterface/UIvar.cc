/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  UIvar.cc: Standard operator defintions for UISingle template class
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   September 2004
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/UIvar.h>
using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

//template class UiSingle<string>;
template class UiSingle<double>;
template class UiSingle<int>;


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif


template<class T>
inline UiSingle<T> & UiSingle<T>::operator=(GuiContext *new_context) {
  context_ = new_context;
  if (context_) context_->reset();
  return *this;
}

template<class T>
inline GuiContext & UiSingle<T>::operator->() {
  return *context_;
}

template<class T>
inline bool UiSingle<T>::operator!() {
  get();
  return !value_;
}


template<class T>
inline UiSingle<T> & UiSingle<T>::operator()() {
  if (context_) context_->reset();
  return *this;
}

template<class T>
UiSingle<T> & UiSingle<T>::operator*() {
  if (context_) context_->reset();
  return *this;    
}

template<class T>
const T & UiSingle<T>::operator&() {
  get();
  return value_;    
}

template<class T>
UiSingle<T> & UiSingle<T>::operator=(const T &right) {
  set(right);
  return *this;
}

template<class T>
T UiSingle<T>::operator+(const T & right) {
  get();
  return value_+right;
}

template<class T>
UiSingle<T> & UiSingle<T>::operator+=(const T & right) {
  set(get() + right);
  return *this;
}

template<class T>
T UiSingle<T>::operator-(const T & right) {
  get();
  return value_-right;
}


template<class T>
UiSingle<T> & UiSingle<T>::operator-=(const T & right) {
  set(get() - right);
  return *this;
}


template<class T>
T UiSingle<T>::operator*(const T & right) {
  get();
  return value_*right;
}


template<class T>
UiSingle<T> & UiSingle<T>::operator*=(const T & right) {
  get();
  set(value_ * right);
  return *this;
}


template<class T>
T UiSingle<T>::operator/(const T & right) {
  get();
  return value_ / right;
}



template<class T>
inline UiSingle<T> & UiSingle<T>::operator/=(const T & right) {
  get();
  set(value_ / right);
  return *this;
}



template<class T>
inline bool UiSingle<T>::operator&&(const T & right) {
  get();
  return value_ && right;
}

template<class T>
inline bool UiSingle<T>::operator||(const T & right) {
  get();
  return value_ || right;
}


template<class T>
inline bool UiSingle<T>::operator==(const T & right) {
  get();
  return value_ == right;
}

template<class T>
inline bool UiSingle<T>::operator!=(const T & right) {
  get();
  return value_ != right;
}

template<class T>
inline bool UiSingle<T>::operator<(const T & right) {
  get();
  return value_ < right;
}


template<class T>
inline bool UiSingle<T>::operator>(const T & right) {
  get();
  return value_ > right;
}


template<class T>
inline bool UiSingle<T>::operator<=(const T & right) {
  get();
  return value_ < right;
}


template<class T>
inline bool UiSingle<T>::operator>=(const T & right) {
  get();
  return value_ > right;
}


template<class T>
inline UiSingle<T> UiSingle<T>::operator--(int) {
  get();
  UiSingle<T> rval(context_, value_);
  set(--value_);
  return rval;
}


template<class T>
inline UiSingle<T> UiSingle<T>::operator++(int) {
  get();
  UiSingle<T> rval(context_, value_);
  set(++value_);
  return rval;
}
