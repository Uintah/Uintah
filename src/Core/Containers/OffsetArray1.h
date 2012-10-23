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


/*
 *  OffsetArray1.h: Interface to dynamic 1D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 */

#ifndef SCI_Containers_OffsetArray1_h
#define SCI_Containers_OffsetArray1_h 1

#include <sci_defs/template_defs.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

class Piostream;
class RigorousTest;

template<class T> class OffsetArray1;
template<class T> void Pio(Piostream& stream, OffsetArray1<T>& array);


/**************************************

  CLASS
  OffsetArray1

  KEYWORDS
  OffsetArray1

  DESCRIPTION
  OffsetArray1.h: Interface to dynamic 1D array class

  Written by:
  Steven G. Parker
  Department of Computer Science
  University of Utah
  March 1994

  PATTERNS

  WARNING
  
****************************************/
template<class T> class OffsetArray1 {
  T* objs;
  int _l;
  int _h;
public:

  //////////
  //Copy the array - this can be costly, so try to avoid it.
  OffsetArray1(const OffsetArray1&);

  //////////
  //Make a new array 1. <i>size</i> gives the initial size of the array,
  //<i>default_grow_size</i> indicates the minimum number of objects that
  //should be added to the array at a time.  <i>asize</i> tells how many
  //objects should be allocated initially
  OffsetArray1(int l=0, int h=0);

  //////////
  //Copy over the array - this can be costly, so try to avoid it.
  OffsetArray1<T>& operator=(const OffsetArray1&);

  //////////
  //Deletes the array and frees the associated memory
  ~OffsetArray1();
    
  //////////
  // Accesses the nth element of the array
  inline const T& operator[](int n) const {
    CHECKARRAYBOUNDS(n, _l, _h);
    return objs[n];
  }

  //////////
  // Accesses the nth element of the array
  inline T& operator[](int n) {
    CHECKARRAYBOUNDS(n, _l, _h);
    return objs[n];
  }
    
  //////////
  // Returns the lower bound of the array
  inline int low() const{ return _l;}

  //////////
  // Returns the upper bound of the array
  inline int high() const{ return _h;}


  //////////
  // Change the size of the array.
  void resize(int l, int h);

  //////////
  // Initialize all elements of the array
  void initialize(const T& val);

  //////////
  // Get the array information
  T* get_objs();

  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, OffsetArray1<T>&);
};

template<class T>
OffsetArray1<T>::OffsetArray1(const OffsetArray1<T>& a)
{
  _l=a._l;
  _h=a._h;
  int size = _h-_l;
  ASSERT(size>=0);
  if(size)
    objs=new T[size]-_l;
  else
    objs=0;
  for(int i=this->_l;i<this->_h;i++)objs[i]=a.objs[i];
}

template<class T>
OffsetArray1<T>& OffsetArray1<T>::operator=(const OffsetArray1<T>& copy)
{
  if (objs)delete [] (objs+_l);
  _l=copy._l;
  _h=copy._h;
  int size = _h-_l;
  ASSERT(size>=0);
  if(size)
    objs=new T[size]-_l;
  else
    objs=0;
  for(int i=_l;i<_h;i++)objs[i]=copy.objs[i];
  return(*this);
}

template<class T>
OffsetArray1<T>::OffsetArray1(int l, int h)
  : _l(l), _h(h)
{
  int size=_h-_l;
  ASSERT(size >= 0);
  if(size)
    objs=new T[size]-_l;
  else
    objs=0;
}	

template<class T>
OffsetArray1<T>::~OffsetArray1()
{
  if(objs)delete [] (objs+_l);
}

template<class T>
void OffsetArray1<T>::resize(int l, int h)
{
  int newsize=h-l;
  int cursize=_h-_l;
  if(newsize == cursize){
    objs+=_l-l;
  } else {
    delete[] (objs+_l);
    if(newsize != 0)
      objs = new T[newsize]-l;
    else
      objs = 0;
  }
  _l=l;
  _h=h;
}

template<class T>
void OffsetArray1<T>::initialize(const T& val)
{
  for (int i=_l;i<_h;i++)objs[i]=val;
}

template<class T>
T* OffsetArray1<T>::get_objs()
{
  return objs+_l;
}

#define OFFSETARRAY1_VERSION 1

template<class T>
void Pio(Piostream& stream, OffsetArray1<T>& array)
{
  /* int version= */stream.begin_class("OffsetArray1", OFFSETARRAY1_VERSION);
  int l=array.low();
  int h=array.high();
  Pio(stream, l);
  Pio(stream, h);
  if(stream.reading()){
    array.resize(l, h);
  }
  for(int i=l;i<h;i++)
    Pio(stream, array.objs[i]);
  stream.end_class();
}

template<class T>
void Pio(Piostream& stream, OffsetArray1<T>*& array) {
  if (stream.reading())
    array=new OffsetArray1<T>;
  Pio(stream, *array);
}

} // End namespace SCIRun


#endif /* SCI_Containers_OffsetArray1_h */

