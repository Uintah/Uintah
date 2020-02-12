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

#include <Core/Util/Assert.h>
#include <vector>

namespace Uintah {


class RigorousTest;

template<class T> class OffsetArray1;



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
  T* m_objs;
  int m_l;
  int m_h;
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
    CHECKARRAYBOUNDS(n, m_l, m_h);
    return m_objs[n];
  }

  //////////
  // Accesses the nth element of the array
  inline T& operator[](int n) {
    CHECKARRAYBOUNDS(n, m_l, m_h);
    return m_objs[n];
  }
    
  //////////
  // Returns the lower bound of the array
  inline int low() const{ return m_l;}

  //////////
  // Returns the upper bound of the array
  inline int high() const{ return m_h;}


  //////////
  // Change the size of the array.
  void resize(int l, int h);

  //////////
  // Initialize all elements of the array
  void initialize(const T& val);

  //////////
  // Get the array information
  T* get_objs();
  
  //////////
  // convert offsetArray to a std::vector
  std::vector<T> to_stl_vector();
};

template<class T>
OffsetArray1<T>::OffsetArray1(const OffsetArray1<T>& a)
{
  m_l=a.m_l;
  m_h=a.m_h;
  int size = m_h-m_l;
  ASSERT(size>=0);
  if(size){
    m_objs=new T[size]-m_l;
  }else{
    m_objs=0;
  }
  
  for(int i=this->m_l;i<this->m_h;i++){
    m_objs[i]=a.m_objs[i];
  }
}

template<class T>
OffsetArray1<T>& OffsetArray1<T>::operator=(const OffsetArray1<T>& copy)
{
  if (m_objs){
    delete [] (m_objs+m_l);
  }
  m_l=copy.m_l;
  m_h=copy.m_h;
  int size = m_h-m_l;
  ASSERT(size>=0);
  if(size){
    m_objs=new T[size]-m_l;
  }else{
    m_objs=0;
  }
    
  for(int i=m_l;i<m_h;i++){
    m_objs[i]=copy.m_objs[i];
  }
  return(*this);
}

template<class T>
OffsetArray1<T>::OffsetArray1(int l, int h)
  : m_l(l), m_h(h)
{
  int size=m_h-m_l;
  ASSERT(size >= 0);
  if(size){
    m_objs=new T[size]-m_l;
  }else{
    m_objs=0;
  }
}       

template<class T>
OffsetArray1<T>::~OffsetArray1()
{
  if(m_objs){
    delete [] (m_objs+m_l);
  }
}

template<class T>
void OffsetArray1<T>::resize(int l, int h)
{
  int newsize=h-l;
  int cursize=m_h-m_l;
  if(newsize == cursize){
    m_objs+=m_l-l;
  } else {
  
    delete[] (m_objs+m_l);
    if(newsize != 0){
      m_objs = new T[newsize]-l;
    }else{
      m_objs = 0;
    }
  }
  m_l=l;
  m_h=h;
}

template<class T>
void OffsetArray1<T>::initialize(const T& val)
{
  for (int i=m_l;i<m_h;i++){
    m_objs[i]=val;
  }
}

template<class T>
T* OffsetArray1<T>::get_objs()
{
  return m_objs + m_l;
}


//////////
// to_stl_vector()
template<class T>
std::vector<T> OffsetArray1<T>::to_stl_vector( )
{
  ASSERT( (m_h-m_l) >=0 );
  
  std::vector<T> vec;
  for(int i= m_l;i< m_h;i++){
    vec.push_back(m_objs[i]);
  }
  return vec;
}

#define OFFSETARRAY1_VERSION 1


} // End namespace Uintah


#endif /* SCI_Containers_OffsetArray1m_h */

