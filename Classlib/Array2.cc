 /*
 *  Array2.cc: Implementation of dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>

#include <Tester/RigorousTest.h>

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
void Array2<T>::newsize(int d1, int d2)
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
	data.newsize(d1, d2);
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

template<class T>
Array2<T>& Array2<T>::operator=(const Array2<T> &copy)
{
  // ok, i did this, but i'm not quite sure it will work...
  
  newsize( copy.dim1(), copy.dim2() );

  for(int i=0;i<dm1;i++)
    for(int j=0;j<dm2;j++)
      objs[i][j] = copy.objs[i][j];
    return( *this );
}

