/*
 *  Array3.cc: Implementation of dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Array3.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <Classlib/NotFinished.h>

#ifdef __GNUG__
#pragma interface
#endif

template<class T>
Array3<T>::Array3()
{
    objs=0;
}

template<class T>
void Array3<T>::allocate()
{
    if(dm1 == 0 || dm2 == 0 || dm3 == 0){
	objs=0;
    } else {
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
}

template<class T>
void Array3<T>::newsize(int d1, int d2, int d3)
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
Array3<T>::Array3(const Array3<T>& a)
: dm1(a.dm1), dm2(a.dm2), dm3(a.dm3)
{
    allocate();
}

template<class T>
Array3<T>::Array3(int dm1, int dm2, int dm3)
: dm1(dm1), dm2(dm2),dm3(dm3)
{
    allocate();
}

template<class T>
Array3<T>::~Array3()
{
    if(objs){
	delete[] objs[0][0];
	delete[] objs[0];
	delete[] objs;
    }
}

template<class T>
void Array3<T>::initialize(const T& t)
{
    ASSERT(dm1 == 0 || dm2 == 0 || dm3 == 0 || objs != 0);
    for(int i=0;i<dm1;i++){
	for(int j=0;j<dm2;j++){
	    for(int k=0;k<dm3;k++){
		objs[i][j][k]=t;
	    }
	}
    }
}

template<class T>
T* Array3<T>::get_onedim()
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
Array3<T>::get_onedim_byte( unsigned char *v )
{
  int i,j,k, index;
  index = 0;
  
  for( k=0; k<dm3; k++ )
    for( j=0; j<dm2; j++ )
      for( i=0; i<dm1; i++)
	v[index++] = objs[i][j][k];
}

#define ARRAY3_VERSION 1

template<class T>
void Pio(Piostream& stream, Array3<T>& data)
{
    /*int version=*/stream.begin_class("Array3", ARRAY3_VERSION);
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
void Pio(Piostream& stream, Array3<T>*& data) {
    if (stream.reading()) {
	data=scinew Array3<T>;
    }
    Pio(stream, *data);
}

template<class T>
Array3<T>& Array3<T>::operator=(const Array3<T>&)
{
    NOT_FINISHED("Array2::operator=");
}

#include <iostream.h>

void Array3<int>::test_rigorous(RigorousTest* __test)
{
    for(int x=0;x<=10;x++){
	for(int y=0;y<=10;y++){
	    for (int z=0;z<10;z++){
		Array3<int> my_array(x,y,z);
		TEST (my_array.dim1()==x);
		TEST (my_array.dim2()==y);
		TEST (my_array.dim3()==z);
	    }
	}
    }
    
    Array3<int> array3(0,0,0);
    TEST (array3.dim1()==0);
    TEST (array3.dim2()==0);
    TEST (array3.dim3()==0);

    
    //A 10X10X10 array is used for the following test, however, it is known to
    //pass when as large as 100X100X100, and has been made smaller for speed
    //purposes


    for(x=0;x<=10;x++){
	for(int y=0;y<10;y++){
	    for(int z=0;z<10;z++){
		array3.newsize(x,y,z);
		TEST (array3.dim1()==x);
		TEST (array3.dim2()==y);
		TEST (array3.dim3()==z);

		array3.initialize(x);

		for(int x1=0;x1<x;x1++){
		    for(int y1=0;y1<y;y1++){
			for(int z1=0;z1<z;z1++){
			    TEST(array3(x1,y1,z1)==x);
			}
		    }
		}
	    }
	}
    }

    
    //The following block of code is known to cause an assertion failure
    
    //#include <Classlib/String.h>
    //
    //for(x=0;x<=10;x++){
    //for(int y=0;y<=10;y++){
    //    for(int z=0;z<=10;z++){
    //	Array3<clString> string_array(x,y,z);
    //	TEST (string_array.dim1()==x);
    //	TEST (string_array.dim2()==y);
    //	TEST (string_array.dim3()==z);
    //
    //	string_array.initialize("hi there");
    //
    //	for(int x1=0;x1<x;x++){
    //	    for(int y1=0;y1<y;y++){
    //		for(int z1=0;z1<z;z++){
    //		    TEST(string_array(x1,y1,z1)=="hi there");
    //		}
    //	    }
    //	}
    //    }
    //}
    //}
    //
}









