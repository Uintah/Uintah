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
 *  Array2.h: Interface to dynamic 2D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Array2_h
#define SCI_Containers_Array2_h 1

#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>

#ifdef _WIN32
#pragma warning(disable:4786)
#endif

namespace SCIRun {

class RigorousTest;

template<class T> class Array2;
template<class T> void Pio(Piostream& stream, Array2<T>& data);
template<class T> void Pio(Piostream& stream, Array2<T>*& data);

/**************************************

 CLASS
 Array2

 KEYWORDS
 Array2

 DESCRIPTION
 Array2.h: Interface to dynamic 2D array class

 Written by:
 Steven G. Parker
 Department of Computer Science
 University of Utah
 March 1994

 Copyright (C) 1994 SCI Group
 PATTERNS

 WARNING
  
****************************************/
template<class T>
class Array2 {
  T** objs;
  int dm1;
  int dm2;
  void allocate();

  // The copy constructor and the assignment operator have been
  // privatized on purpose -- no one should use these.  Instead,
  // use the default constructor and the copy method.
  //////////
  //Array2 Copy Constructor
  Array2(const Array2&);
  //////////
  //Assignment Operator
  Array2<T>& operator=(const Array2&);
public:
  //////////
  //Create a 0X0 Array
  Array2();
    
  //////////
  //Create an n by n array
  Array2(int, int);

  //////////
  //Class Destructor
  virtual ~Array2();

  //////////
  //Used for accessing elements in the Array
  inline T& operator()(int d1, int d2) const
  {
    ASSERTL3(d1>=0 && d1<dm1);
    ASSERTL3(d2>=0 && d2<dm2);
    return objs[d1][d2];
  }
    
  //////////
  //Array2 Copy Method
  void copy(const Array2&);
    
  //////////
  //Returns number of rows
  inline int dim1() const {return dm1;}
    
  //////////
  //Returns number of cols
  inline int dim2() const {return dm2;}
    
  //////////
  //Resize Array
  void resize(int, int);
    
  //////////
  //Initialize all values in an array
  void initialize(const T&);

  inline T** get_dataptr() {return objs;}

    
  //////////
  //Rigorous Tests
  static void test_rigorous(RigorousTest* __test);

#if defined(_AIX)
  template <typename Type> 
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array2<Type>&);
  template <typename Type> 
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array2<Type>*&);
#else
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array2<T>&);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array2<T>*&);
#endif
};

template<class T>
Array2<T>::Array2()
{
  dm1=dm2=0;
  objs=0;
}

template<class T>
void Array2<T>::allocate()
{
  if(dm1 == 0 || dm2 == 0){
    objs=0;
  } else {
    objs=new T*[dm1];
    T* p=new T[dm1*dm2];
    for(int i=0;i<dm1;i++){
      objs[i]=p;
      p+=dm2;
    }
  }
}

template<class T>
void Array2<T>::resize(int d1, int d2)
{
  if(objs && dm1==d1 && dm2==d2)return;
  dm1=d1;
  dm2=d2;
  if(objs){
    delete[] objs[0];
    delete[] objs;
  }
  allocate();
}

template<class T>
Array2<T>::Array2(const Array2<T>& a)
  : dm1(a.dm1), dm2(a.dm2)
{
  allocate();
}

template<class T>
Array2<T>::Array2(int dm1, int dm2)
  : dm1(dm1), dm2(dm2)
{
  allocate();
}

template<class T>
Array2<T>::~Array2()
{
  if(objs){
    delete[] objs[0];
    delete[] objs;
  }
}

template<class T>
void Array2<T>::initialize(const T& t)
{
  ASSERT(dm1==0 || dm2==0 || objs != 0);
  for(int i=0;i<dm1;i++){
    for(int j=0;j<dm2;j++){
      objs[i][j]=t;
    }
  }
}

template<class T>
void Array2<T>::copy(const Array2<T> &copy)
{
  resize( copy.dim1(), copy.dim2() );
  for(int i=0;i<dm1;i++)
    for(int j=0;j<dm2;j++)
      objs[i][j] = copy.objs[i][j];
}

#define Array2_VERSION 1

template<class T>
void Pio(Piostream& stream, Array2<T>& data)
{
  stream.begin_class("Array2", Array2_VERSION);
  if(stream.reading()){
    // Allocate the array...
    int d1, d2;
    Pio(stream, d1);
    Pio(stream, d2);
    data.resize(d1, d2);
  } else {
    Pio(stream, data.dm1);
    Pio(stream, data.dm2);
  }
  for(int i=0;i<data.dm1;i++){
    for(int j=0;j<data.dm2;j++){
      Pio(stream, data.objs[i][j]);
    }
  }
  stream.end_class();
}

template<class T>
void Pio(Piostream& stream, Array2<T>*& data) {
  if (stream.reading()) {
    data=new Array2<T>;
  }
  Pio(stream, *data);
}

} // End namespace SCIRun

#endif

