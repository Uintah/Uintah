
/*
 *  Array1.cc: Implementation of dynamic 1D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef __GNUG__
#pragma interface
#endif

#include <Classlib/Array1.h>
#include <Classlib/Persistent.h>
#include <Malloc/Allocator.h>

#include <Tester/RigorousTest.h>

template<class T>
Array1<T>::Array1(const Array1<T>& a)
{
    _size=a._size;
    nalloc=_size;
    objs=new T[_size];
    for(int i=0;i<_size;i++)objs[i]=a.objs[i];
    nalloc=_size;
    default_grow_size=a.default_grow_size;
}

template<class T>
Array1<T>& Array1<T>::operator=(const Array1<T>& copy)
{
    if (objs)delete [] objs;
    _size=copy._size;
    nalloc=_size;
    objs=new T[_size];
    for(int i=0;i<_size;i++)objs[i]=copy.objs[i];
    nalloc=_size;
    default_grow_size=copy.default_grow_size;
    return(*this);
}

template<class T>
Array1<T>::Array1(int size, int gs, int asize)
{
    ASSERT(size >= 0);
    default_grow_size=gs;
    if(size){
	if(asize==-1){
	    objs=new T[size];
	    _size=size;
	    nalloc=_size;
	} else {
	    objs=new T[asize];
	    _size=size;
	    nalloc=asize;
	}
    } else {
	if(asize==-1){
	    objs=0;
	    _size=0;
	    nalloc=0;
	} else {
	    objs=new T[asize];
	    _size=0;
	    nalloc=asize;
	}
    }
    nalloc=_size;
}	

template<class T>
Array1<T>::~Array1()
{
    if(objs)delete [] objs;
}

template<class T>
void Array1<T>::grow(int count, int grow_size)
{
    int newsize=_size+count;
    if(newsize>nalloc){
	// Reallocate...
	int gs1=newsize>>2;
	int gs=gs1>grow_size?gs1:grow_size;
	int newalloc=newsize+gs;
	T* newobjs=new T[newalloc];
	if(objs){
	    for(int i=0;i<_size;i++){
		newobjs[i]=objs[i];
	    }
	    delete[] objs;
	}
	objs=newobjs;
	nalloc=newalloc;
    }
    _size=newsize;
}

template<class T>
void Array1<T>::add(const T& obj)
{
    grow(1, default_grow_size);
    objs[_size-1]=obj;
}

template<class T>
void Array1<T>::insert(int idx, const T& obj)
{
    grow(1, default_grow_size);
    for(int i=_size-1;i>idx;i--)objs[i]=objs[i-1];
    objs[idx]=obj;
}

template<class T>
void Array1<T>::remove(int idx)
{
    _size--;
    for(int i=idx;i<_size;i++)objs[i]=objs[i+1];
}

template<class T>
void Array1<T>::remove_all()
{
    _size=0;
}

template<class T>
void Array1<T>::resize(int newsize)
{
    if(newsize > _size)
	grow(newsize-_size);
    else
	_size=newsize;
}

template<class T>
void Array1<T>::setsize(int newsize)
{ 
    if(newsize > nalloc) { // have to reallocate...
      T* newobjs=new T[newsize];     // make it exact!
      if (objs) {
	for(int i=0;i<_size;i++){
	  newobjs[i]=objs[i];
	}
	delete[] objs;
      }		
      objs = newobjs;
      nalloc = newsize;
      
    }
    _size=newsize;
}



template<class T>
void Array1<T>::initialize(const T& val) {
    for (int i=0;i<_size;i++)objs[i]=val;
}

template<class T>
T* Array1<T>::get_objs()
{
  return objs;
}

#define ARRAY1_VERSION 1

template<class T>
void Pio(Piostream& stream, Array1<T>& array)
{
    /* int version= */stream.begin_class("Array1", ARRAY1_VERSION);
    int size=array._size;
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

#include <Classlib/String.h>
#include <iostream.h>



void Array1<int>::test_rigorous(RigorousTest* __test)
{
    Array1<int> my_array;
    TEST(my_array.size()==0);

    for(int i=0;i<1000;i++){
	my_array.grow(1);
	TEST(my_array.size()==i+1);
    }

    for(i=0;i<1000;i++){
	my_array[i]=i*1000;
	TEST(my_array.size()==1000);
    }

    for(i=0;i<1000;i++){
	TEST(my_array[i]==i*1000);
	TEST(my_array.size()==1000);
    }

    Array1<int> my_array2(10);
    TEST(my_array2.size()==10);

    for(i=0;i<10;i++){
	my_array2[i] = i*10;
    }

    for(i=0;i<10;i++){
	TEST(my_array2[i]==i*10);
	TEST(my_array2.size()==10);
    }



    
    Array1<int> x_array;

    for(i=1;i<=1000;i++){
	x_array.setsize(i);
	TEST(x_array.size()==i);
    }


    x_array.setsize(1000);
    TEST(x_array.size()==1000);

    x_array.initialize(11);
    

    for(i=0;i<1000;i++){
	TEST(x_array[i]==11);
    }

    

    

    Array1<clString> string_array;

    
    for(i=1;i<=1000;i++){
	string_array.setsize(i);
	TEST(string_array.size()==i);
    }

    string_array.grow(10000);
    
    for(i=0;i<2996;i+=3){
	string_array[i] = "hi ";
	string_array[i+1] = "there";
	string_array[i+2] = string_array[i]+string_array[i+1];
    }
    
    for(i=0;i<2996;i+=3){
	TEST (string_array[i]=="hi ");
	TEST (string_array[i+1]=="there");
	TEST (string_array[i+2]=="hi there");
    }

    
    string_array.remove_all();
    TEST(string_array.size()==0);



    int c = 0;

    Array1<clString> string_array2;

    for(i=0;i<1000;i++){
	string_array2.grow(1);
	TEST(string_array2.size()==i+1);

    }



    for(i=1000;i>0;i--){
	string_array2.remove(i);
	TEST(string_array2.size()==i-1);
    }


    string_array2.remove_all();
    TEST(string_array2.size()==0);


    for(i=0;i<1000;i++){
	string_array2.add("hi there");
    }


    for(i=0;i<1000;i++)
    {
	TEST(string_array2[i]=="hi there");
    }


    string_array2.initialize("hello");
    
    
    for(i=0;i<1000;i++)
    {
	TEST(string_array2[i]=="hello");
    }
}
















