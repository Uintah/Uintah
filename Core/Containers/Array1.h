/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#ifndef SCI_Containers_Array1_h
#define SCI_Containers_Array1_h 1

#include <sci_config.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>

#ifdef _WIN32
#pragma warning(disable:4786)
#endif


namespace SCIRun {

class Piostream;
class RigorousTest;

template<class T> class Array1;
template<class T> void Pio(Piostream& stream, Array1<T>& array);


/**************************************

  CLASS
  Array1

  KEYWORDS
  Array1

  DESCRIPTION
  Array1.h: Interface to dynamic 1D array class

  Written by:
  Steven G. Parker
  Department of Computer Science
  University of Utah
  March 1994

  Copyright (C) 1994 SCI Group
  PATTERNS

  WARNING
  
****************************************/
template<class T> class Array1 {T* objs;
  int dm1;
  int nalloc;
  int default_growdm1;
  void allocate();
public:
  //////////
  //Make a new array 1. <i>size</i> gives the initial size of the array,
  //<i>default_growdm1</i> indicates the minimum number of objects that
  //should be added to the array at a time.  <i>asize</i> tells how many
  //objects should be allocated initially
  Array1(int size=0, int default_growdm1=10, int asize=-1);

 //////////
  //Copy the array - this can be costly, so try to avoid it.
  Array1(const Array1&);

  //////////
  //Copy over the array - this can be costly, so try to avoid it.
  Array1<T>& operator=(const Array1&);

  //////////
  //Deletes the array and frees the associated memory
  ~Array1();
    
  //////////
  // Accesses the nth element of the array
  inline const T& operator[](int n) const {
    CHECKARRAYBOUNDS(n, 0, dm1);
    return objs[n];
  }

  //////////
  // Accesses the nth element of the array
  inline T& operator[](int n) {
    CHECKARRAYBOUNDS(n, 0, dm1);
    return objs[n];
  }
    
  //////////
  // Returns the size of the array
  inline int size() const{ return dm1;}


  //////////
  // Make the array larger by count elements
  void grow(int count, int growdm1=10);

  //////////
  // set allocated size 
  void reserve(int n);

  //////////
  // Add one element to the array.  equivalent to:
  //  grow(1)
  //  array[array.size()-1]=data
  void add(const T&);

  //////////
  // Insert one element into the array.  This is very inefficient
  // if you insert anything besides the (new) last element.
  void insert(int, const T&);


  //////////
  // Remove one element from the array.  This is very inefficient
  // if you remove anything besides the last element.
  void remove(int);


  //////////
  // Remove all elements in the array.  The array is not freed,
  // and the number of allocated elements remains the same.
  void remove_all();
   
  //////////
  //Resize Array
  void newsize(int);
    
   //////////
  // Change the size of the array.
  void resize(int newsize);

  //////////
  // Changes size, makes exact if currently smaller...
  void setsize(int newsize);

  //////////
  // Initialize all elements of the array
  void initialize(const T& val);


  //////////
  // Get the array information
  T* get_objs();


  //////////
  //Rigorous Tests
  static void test_rigorous(RigorousTest* __test);

#if defined(_AIX)
  template <typename Type> 
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array1<Type>&);
#else
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array1<T>&);
#endif
};


template<class T>
Array1<T>::Array1(const Array1<T>& a)
{
  dm1=a.dm1;
  nalloc=dm1;
  objs=new T[dm1];
  for(int i=0;i<dm1;i++)objs[i]=a.objs[i];
  nalloc=dm1;
  default_growdm1=a.default_growdm1;
}

template<class T>
Array1<T>& Array1<T>::operator=(const Array1<T>& copy)
{
  if (objs)delete [] objs;
  dm1=copy.dm1;
  nalloc=dm1;
  objs=new T[dm1];
  for(int i=0;i<dm1;i++)objs[i]=copy.objs[i];
  nalloc=dm1;
  default_growdm1=copy.default_growdm1;
  return(*this);
}

template<class T>
Array1<T>::Array1(int size, int gs, int asize)
{
  ASSERT(size >= 0);
  default_growdm1=gs;
  dm1=size;
  if(size){
    if(asize <= size){
      objs=new T[size];
      nalloc=dm1;
    } else {
      objs=new T[asize];
      nalloc=asize;
    }
  } else {
    if(asize > 0){
      objs=new T[asize];
      nalloc=asize;
    } else {
      objs=0;
      nalloc=0;
    }
  }
}	

template<class T>
Array1<T>::~Array1()
{
  if(objs)delete [] objs;
}

template<class T>
void Array1<T>::allocate()
{
  if(dm1 == 0){
    objs=0;
    nalloc=0;
  } else {
    objs=new T[dm1];
    nalloc=dm1;
 }
}

template<class T>
void Array1<T>::grow(int count, int growdm1)
{
  int newsize=dm1+count;
  if(newsize>nalloc){
    // Reallocate...
    int gs1=newsize>>2;
    int gs=gs1>growdm1?gs1:growdm1;
    int newalloc=newsize+gs;
    T* newobjs=new T[newalloc];
    if(objs){
      for(int i=0;i<dm1;i++){
	newobjs[i]=objs[i];
      }
      delete[] objs;
    }
    objs=newobjs;
    nalloc=newalloc;
  }
  dm1=newsize;
}

template<class T>
void Array1<T>::reserve(int n)
{
  if(n>nalloc){
    // Reallocate...
    T* newobjs=new T[n];
    if(objs){
      for(int i=0;i<dm1;i++){
	newobjs[i]=objs[i];
      }
      delete[] objs;
    }
    objs=newobjs;
    nalloc=n;
  }
}

template<class T>
void Array1<T>::add(const T& obj)
{
  grow(1, default_growdm1);
  objs[dm1-1]=obj;
}

template<class T>
void Array1<T>::insert(int idx, const T& obj)
{
  grow(1, default_growdm1);
  for(int i=dm1-1;i>idx;i--)objs[i]=objs[i-1];
  objs[idx]=obj;
}

template<class T>
void Array1<T>::remove(int idx)
{
  dm1--;
  for(int i=idx;i<dm1;i++)objs[i]=objs[i+1];
}

template<class T>
void Array1<T>::remove_all()
{
  dm1=0;
}

template<class T>
void Array1<T>::newsize(int d1)
{
  if(objs && dm1==d1)return;
  dm1=d1;
  if(objs){
    delete[] objs;
  }
  allocate();
}

template<class T>
void Array1<T>::resize(int newsize)
{
  if(newsize > dm1)
    grow(newsize-dm1);
  else
    dm1=newsize;
}

template<class T>
void Array1<T>::setsize(int newsize)
{ 
  if(newsize > nalloc) { // have to reallocate...
    T* newobjs=new T[newsize];     // make it exact!
    if (objs) {
      for(int i=0;i<dm1;i++){
	newobjs[i]=objs[i];
      }
      delete[] objs;
    }		
    objs = newobjs;
    nalloc = newsize;
      
  }
  dm1=newsize;
}



template<class T>
void Array1<T>::initialize(const T& val) {
  for (int i=0;i<dm1;i++)objs[i]=val;
}

template<class T>
T* Array1<T>::get_objs()
{
  return objs;
}

#define ARRAY1_VERSION 2

template<class T>
void Pio(Piostream& stream, Array1<T>& array)
{
  /* int version= */stream.begin_class("Array1", ARRAY1_VERSION);
  int size=array.dm1;
  Pio(stream, size);
  if(stream.reading()){
    array.remove_all();
    array.grow(size);
  }
  for(int i=0;i<size;i++)
    Pio(stream, array.objs[i]);
  stream.end_class();
}

template<class T>
void Pio(Piostream& stream, Array1<T>*& array) {
  if (stream.reading())
    array=new Array1<T>;
  Pio(stream, *array);
}

} // End namespace SCIRun


#endif /* SCI_Containers_Array1_h */

