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
 *  Array3.h: Interface to dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Array3_h
#define SCI_Containers_Array3_h 1

#include <iostream>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/times.h>

#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>



namespace SCIRun {

class RigorousTest;

template<class T> class Array3;
template<class T> void Pio(Piostream& stream, Array3<T>& array);

/**************************************

 CLASS
 Array3

 KEYWORDS
 Array3

 DESCRIPTION
 Array3.h: Interface to dynamic 3D array class

 Written by:
 Steven G. Parker
 Department of Computer Science
 University of Utah
 March 1994

 Copyright (C) 1994 SCI Group
 PATTERNS

 WARNING
  
****************************************/

template<class T> class Array3 {
  T*** objs;
  int dm1;
  int dm2;
  int dm3;
  void allocate();
public:
  //////////
  //Default Constructor
  Array3();
    
  //////////
  //Copy Constructor
  Array3(const Array3&);

  //////////
  //Constructor
  Array3(int, int, int);
    
  //////////
  //Assignment Operator
  Array3<T>& operator=(const Array3&);
    
  //////////
  //Class Destructor
  virtual ~Array3();
    
  //////////
  //Access the nXnXn element of the array
  inline T& operator()(int d1, int d2, int d3) const
  {
    ASSERTL3(d1>=0 && d1<dm1);
    ASSERTL3(d2>=0 && d2<dm2);
    ASSERTL3(d3>=0 && d3<dm3);
    return objs[d1][d2][d3];
  }
    
  //////////
  //Returns the number of spaces in dim1	    
  inline int dim1() const {return dm1;}
  //////////
  //Returns the number of spaces in dim2
  inline int dim2() const {return dm2;}
  //////////
  //Returns the number of spaces in dim3
  inline int dim3() const {return dm3;}
    
  //////////
  //Re-size the Array
  void newsize(int, int, int);

  //////////
  //Initialize all elements to T
  void initialize(const T&);

  T* get_onedim();
  void get_onedim_byte( unsigned char *v );

  inline T*** get_dataptr() {return objs;}

  //////////
  //read/write from a separate raw file
  int input( const clString& );
  int output( const clString&);

  //////////
  //Rigorous Tests
  static void test_rigorous(RigorousTest* __test);
    
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<T>&);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<T>&, 
					     const clString &);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<T>*&);

};

template<class T>
Array3<T>::Array3()
{
  objs=0;
  dm1 = dm2 = dm3 = 0;
}

template<class T>
void Array3<T>::allocate()
{
  objs=new T**[dm1];
  T** p=new T*[dm1*dm2];
  T* pp=new T[dm1*dm2*dm3];
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
void Array3<T>::newsize(int d1, int d2, int d3)
{
  if(objs && dm1==d1 && dm2==d2 && dm3==d3)return;
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
Array3<T>& Array3<T>::operator=(const Array3<T> &copy)
{
  // ok, i did this, but i'm not quite sure it will work...
  
  newsize( copy.dim1(), copy.dim2(), copy.dim3() );

  for(int i=0;i<dm1;i++)
    for(int j=0;j<dm2;j++)
      for(int k=0;k<dm3;k++)
        objs[i][j][k] = copy.objs[i][j][k];
  return( *this );
}

template<class T>
T* Array3<T>::get_onedim()
{
  int i,j,k, index;
  T* a = new T[dm1*dm2*dm3];
  
  index=0;
  for( i=0; i<dm1; i++)
    for( j=0; j<dm2; j++ )
      for( k=0; k<dm3; k++ )
	a[index++] = objs[i][j][k];
  return a;
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
#ifdef __GNUG__
#else
#endif

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
    data=new Array3<T>;
  }
  Pio(stream, *data);
}

template<class T>
void Pio(Piostream& stream, Array3<T>& data, 
         const clString& filename)
{
#ifdef __GNUG__
#else
#endif

  /*int version=*/stream.begin_class("Array3", ARRAY3_VERSION);
  if(stream.reading()){
    // Allocate the array...
    int d1, d2, d3;
    Pio(stream, d1);
    Pio(stream, d2);
    Pio(stream, d3);
    data.newsize(d1, d2, d3);
    data.input( filename );
  } else {
    Pio(stream, data.dm1);
    Pio(stream, data.dm2);
    Pio(stream, data.dm3);
    data.output( filename );
  }
    
  stream.end_class();
}

template<class T>
int
Array3<T>::input( const clString &filename ) 
{
  std::cerr << "Array3: Split input\n";

  // get raw data
  int file=open( filename(), O_RDONLY, 0666);
  if ( file == -1 ) {
    printf("can not open file %s\n", filename());
    return 0;
  }
  
  int maxiosz=1024*1024;
  long size = dm1*dm2*dm3*long(sizeof(T));
  int n = int(size / maxiosz);
  std::cerr << "grid size = " << size << std::endl;
  char *p = (char *) objs[0][0];

  for ( ; n> 0 ; n--, p+= maxiosz) {
    int i = read( file, p, maxiosz);
    if ( i != maxiosz ) 
      perror( "io read ");
  }
  int i =  read( file, p, size % maxiosz);
  if ( i != (size % maxiosz) ) 
    perror("on last io");
        
  fsync(file);
  close(file);

  return 1;
}

template<class T>
int
Array3<T>::output( const clString &filename ) 
{
  std::cerr << "Array3 output to " << filename << std::endl;
  // get raw data
  //  printf("output [%s] [%s]\n", filename(), rawfile() );
  int file=open( filename(), O_WRONLY|O_CREAT|O_TRUNC, 0666);
  if ( file == -1 ) {
    perror("open file");
    return 0;
  }
  
  int maxiosz=1024*1024;

  int size = dm1*dm2*dm3*sizeof(T);
  int n = size / maxiosz;
  char *p = (char *)objs[0][0];

  printf("Start writing...%d %d %d\n", size, maxiosz, n);

  for ( ; n> 0 ; n--, p+= maxiosz) {
    int l = write( file, p, maxiosz);
    if ( l != maxiosz ) 
      perror("write ");
  }
  int sz = (size % maxiosz );
  int l = write( file, p, sz); 
  if ( l != (size % maxiosz ) ) {
    printf("Error: wrote %d / %d\n", l,(size % maxiosz )); 
    perror("write ");
  }
        
  fsync(file);
  close(file);

  return 1;
} 

} // End namespace SCIRun


#endif


