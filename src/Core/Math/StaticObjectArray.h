/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


//  
//    File   : StaticObjectArray.h
//    Author : Wayne Witzel
//    Date   : Wed Jun  6 15:40:27 2001


#ifndef Uintah_Core_Math_StaticObjectArray_h
#define Uintah_Core_Math_StaticObjectArray_h

#include <Core/Util/Assert.h>

using namespace SCIRun;

namespace Uintah {

/**************************************

CLASS 
   StaticObjectArray
   
KEYWORDS
   array, static

DESCRIPTION
   
   A very simply array class for arrays that do not change size
   after their creation.  It uses CHECKARRAYBOUNDS to check for
   out of bounds errors (turned off when SCI_ASSERTION_LEVEL < 3
   -- i.e. for optimized code).
 
   Written by:
    Wayne Witzel
    Department of Computer Science
    University of Utah
    Nov. 2000
 
     
 
PATTERNS
   
WARNING
  
****************************************/

template <class T>
class StaticObjectArray
{
public:
  StaticObjectArray(unsigned int size)
    : data_(scinew T[size]), size_(size) {}

  StaticObjectArray(const StaticObjectArray& array)
    : data_(scinew T[array.size_]), size_(array.size_)
  {
    for (unsigned int i = 0; i < size_; i++)
      data_[i] = array.data_[i];
  }

  StaticObjectArray& operator=(const StaticObjectArray& array)
  {
    delete[] data_;
    data_ = scinew T[array.size_];
    size_ = array.size_;
    for (unsigned int i = 0; i < size_; i++)
      data_[i] = array.data_[i];

    return *this;
  }
  
  virtual ~StaticObjectArray()
  { delete[] data_; }

  T& operator[](int index)
  {
    CHECKARRAYBOUNDS(index, 0, (int)size_);
    return data_[index];
  }

  T& operator[](unsigned int index)
  {
    ASSERTL3(index < size_);
    return data_[index];
  }

  const T& operator[](int index) const
  {
    CHECKARRAYBOUNDS(index, 0, (int)size_);
    return data_[index];
  }

  const T& operator[](unsigned int index) const
  {
    ASSERTL3(index < size_);
    return data_[index];
  }

  int size() const
  {
     return (int) size_;
  }

  T* pointer() const
  {
     return data_;
  }

private:
  T* data_;
  unsigned int size_;
};

}

#endif
