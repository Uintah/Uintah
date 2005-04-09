/*
 *  LockArray3.cc: Implementation of dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/LockArray3.h>
#include <Malloc/Allocator.h>
#include <Classlib/NotFinished.h>

template<class T>
LockArray3<T>::LockArray3()
{
    objs=0;
}

template<class T>
void LockArray3<T>::allocate()
{
    objs=scinew T**[dm1];
    T** p=scinew T*[dm1*dm2];
    T* pp=scinew T[dm1*dm2*dm3];
    for(int i=0;i<dm1;i++){
	objs[i]=p;
	p+=dm2;
	for(int j=0;j<dm2;j++){
	    objs[i][j]=pp;
	    pp+=dm3;
	}
    }
}

template<class T>
LockArray3<T>* LockArray3<T>::clone() const
{
    return new LockArray3<T>(*this);
}

template<class T>
void LockArray3<T>::newsize(int d1, int d2, int d3)
{
    if(objs && dm1==d2 && dm2==d2 && dm3==d3)return;
    dm1=d1;
    dm2=d2;
    dm3=d3;
    if(objs){
	delete[] objs[0][0];
	delete[] objs[0];
	delete[] objs;
    }
    allocate();
}

template<class T>
LockArray3<T>::LockArray3(const LockArray3<T>& a)
: dm1(a.dm1), dm2(a.dm2), dm3(a.dm3)
{
    allocate();
}

template<class T>
LockArray3<T>::LockArray3(int dm1, int dm2, int dm3)
: dm1(dm1), dm2(dm2),dm3(dm3)
{
    allocate();
}

template<class T>
LockArray3<T>::~LockArray3()
{
    if(objs){
	delete[] objs[0][0];
	delete[] objs[0];
	delete[] objs;
    }
}

template<class T>
void LockArray3<T>::initialize(const T& t)
{
    ASSERT(objs != 0);
    for(int i=0;i<dm1;i++){
	for(int j=0;j<dm2;j++){
	    for(int k=0;k<dm3;k++){
		objs[i][j][k]=t;
	    }
	}
    }
}

template<class T>
T* LockArray3<T>::get_onedim()
{
  int i,j,k, index;
  T* a = scinew T[dm1*dm2*dm3];
  
  for( i=0; i<dm1; i++)
    for( j=0; j<dm2; j++ )
      for( k=0; k<dm3; k++ )
	a[index++] = objs[i][j][k];
}

template<class T>
void
LockArray3<T>::get_onedim_byte( unsigned char *v )
{
  int i,j,k, index;
  index = 0;
  
  for( k=0; k<dm3; k++ )
    for( j=0; j<dm2; j++ )
      for( i=0; i<dm1; i++)
	v[index++] = objs[i][j][k];
}

#define LockArray3_VERSION 1

template<class T>
void Pio(Piostream& stream, LockArray3<T>& data)
{
    /*int version=*/stream.begin_class("LockArray3", LockArray3_VERSION);
    if(stream.reading()){
	// Allocate the array...
	int d1, d2, d3;
	Pio(stream, d1);
	Pio(stream, d2);
	Pio(stream, d3);
	data.newsize(d1, d2, d3);
    } else {
	Pio(stream, data.dm1);
	Pio(stream, data.dm2);
	Pio(stream, data.dm3);
    }
    for(int i=0;i<data.dm1;i++){
	for(int j=0;j<data.dm2;j++){
	    for(int k=0;k<data.dm3;k++){
		Pio(stream, data.objs[i][j][k]);
	    }
	}
    }
    stream.end_class();
}

template<class T>
void Pio(Piostream& stream, LockArray3<T>*& data) {
    if (stream.reading()) {
	data=scinew LockArray3<T>;
    }
    Pio(stream, *data);
}

template<class T>
LockArray3<T>& LockArray3<T>::operator=(const LockArray3<T>&)
{
    NOT_FINISHED("Array2::operator=");
}

template<class T>
void LockArray3<T>::io(Piostream&)
{
  cerr << "Error - not implemented!\n";
}
