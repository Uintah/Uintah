/*
 *  Array2.cc: Implementation of dynamic 2D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include "Array2.h"

namespace rtrt {

template<class T>
Array2<T>::Array2()
{
    objs=0;
    dm1=dm2=0;
}

template<class T>
void Array2<T>::allocate()
{
    if(dm1==0 || dm2==0){
	objs=0;
	refcnt=0;
	return;
    }
    objs=new T*[dm1];
    T* p=new T[dm1*dm2];
    for(int i=0;i<dm1;i++){
        objs[i]=p;
        p+=dm2;
    }
    refcnt=new int;
    *refcnt=1;
}

template<class T>
void Array2<T>::resize(int d1, int d2)
{
    if(objs && dm1==d1 && dm2==d2)return;
    dm1=d1;
    dm2=d2;
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
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
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
}

template<class T>
void Array2<T>::initialize(const T& t)
{
    ASSERT(objs != 0);
    for(int i=0;i<dm1;i++){
        for(int j=0;j<dm2;j++){
	    objs[i][j][k]=t;
        }
    }
}

template<class T>
void Array2<T>::share(const Array2<T>& copy)
{
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
    objs=copy.objs;
    refcnt=copy.refcnt;
    dm1=copy.dm1;
    dm2=copy.dm2;
    (*refcnt)++;
}

} // end namespace rtrt
