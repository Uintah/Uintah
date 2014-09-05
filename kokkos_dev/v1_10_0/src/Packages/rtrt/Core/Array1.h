
/*
 *  Array1.h: Interface to dynamic 1D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef Packages_rtrt_Core_Array1_h
#define Packages_rtrt_Core_Array1_h 1

#include <Packages/rtrt/Core/Assert.h>
//#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>

namespace rtrt {
 template<class T> class Array1;
}

namespace rtrt {

using SCIRun::Piostream;

#ifdef __GNUG__
#pragma interface
#endif

template<class T>
class Array1 {
  T* objs;
  int _size;
  int nalloc;
  int default_grow_size;
public:
  Array1(const Array1&);
  Array1(int size=0, int default_grow_size=10, int asize=-1);
  Array1<T>& operator=(const Array1&);
  ~Array1();
  // Accesses the nth element of the array
  inline T& operator[](int n) const {
    ASSERTRANGE(n, 0, _size);
    return objs[n];
  }
  // Returns the size of the array
  inline int size() const{ return _size;}

  // Make the array larger by count elements
  void grow(int count, int grow_size=10);

  // Add one element to the array.  equivalent to:
  //  grow(1)
  //  array[array.size()-1]=data;
  void add(const T&);
  int add2(const T&);

  // Insert one element into the array.  This is very inefficient
  // if you insert anything besides the (new) last element.
  void insert(int, const T&);

  // Remove one element from the array.  This is very inefficient
  // if you remove anything besides the last element.
  void remove(int);

  // Remove all elements in the array.  The array is not freed,
  // and the number of allocated elements remains the same.
  void remove_all();

  // Change the size of the array.
  void resize(int newsize);

  // Changes size, makes exact if currently smaller...
  void setsize(int newsize);

  // Initialize all elements of the array
  void initialize(const T& val);

  // Get the array information
  T* get_objs();
};

} // end namespace rtrt

#include <Packages/rtrt/Core/Array1.cc>

#endif // Packages_rtrt_Core_Array1_h
