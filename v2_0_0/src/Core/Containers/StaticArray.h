//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : StaticArray.h
//    Author : Wayne Witzel
//    Date   : Wed Jun  6 15:40:27 2001


#ifndef SCI_Containers_StaticArray_h
#define SCI_Containers_StaticArray_h

#include <Core/Util/Assert.h>

namespace SCIRun {

/**************************************

CLASS 
   StaticArray
   
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
 
   Copyright (C) 2000 SCI Group
  
 
PATTERNS
   
WARNING
  
****************************************/

template <class T>
class StaticArray
{
public:
  StaticArray(unsigned int size)
    : data_(new T[size]), size_(size) {}

  StaticArray(const StaticArray& array)
    : data_(new T[array.size_]), size_(array.size_)
  {
    for (unsigned int i = 0; i < size_; i++)
      data_[i] = array.data_[i];
  }

  StaticArray& operator=(const StaticArray& array)
  {
    delete[] data_;
    data_ = new T[array.size_];
    size_ = array.size_;
    for (unsigned int i = 0; i < size_; i++)
      data_[i] = array.data_[i];

    return *this;
  }
  
  ~StaticArray()
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
