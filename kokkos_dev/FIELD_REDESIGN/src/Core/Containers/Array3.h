
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

#include <SCICore/Util/Assert.h>
#include <SCICore/Persistent/Persistent.h>

namespace SCICore {

namespace Tester {
  class RigorousTest;
}

namespace Geometry {
  void Pio();  // This is a dummy declaration to get things to compile.
}

namespace Containers {

using SCICore::PersistentSpace::Piostream;
using SCICore::Tester::RigorousTest;

template<class T>
class Array3;
template<class T>
void Pio(Piostream& stream, Array3<T>& array);

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

template<class T>
class Array3 {
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
    ~Array3();
    
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
void Pio(Piostream& stream, Containers::Array3<T>& data)
{
#ifdef __GNUG__
    using namespace SCICore::Geometry;
    using namespace SCICore::PersistentSpace;
    using namespace SCICore::Containers;
#else
    using SCICore::Geometry::Pio;
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
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
void Pio(Piostream& stream, Containers::Array3<T>*& data) {
    if (stream.reading()) {
	data=new Array3<T>;
    }
    Containers::Pio(stream, *data);
}

template<class T>
void Pio(Piostream& stream, Containers::Array3<T>& data, 
         const clString& filename)
{
#ifdef __GNUG__
    using namespace SCICore::Geometry;
    using namespace SCICore::PersistentSpace;
    using namespace SCICore::Containers;
#else
    using SCICore::Geometry::Pio;
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
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

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.12  2000/03/17 08:27:10  sparker
// Made input() and output() be non-virtual functions
//
// Revision 1.11  2000/02/04 01:25:51  dmw
// added std:: before cerr and endl
//
// Revision 1.10  2000/02/04 00:07:51  yarden
// provide methods for writing and reading the array in to/from a seperate
// file in a binary mode.
//
// Revision 1.9  1999/09/16 23:03:49  mcq
// Fixed a few little bugs, hopefully didn't introduce more.  Started ../doc
//
// Revision 1.8  1999/09/08 02:26:45  sparker
// Various #include cleanups
//
// Revision 1.7  1999/09/03 06:07:20  dmw
// added a Makefile.in for Leonid's files
//
// Revision 1.6  1999/08/29 00:46:51  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/28 17:54:34  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/24 06:24:00  dmw
// Added in everything for the DaveW branch
//
// Revision 1.3  1999/08/19 23:18:04  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.2  1999/08/17 06:38:35  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:11  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:35  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:42  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:29  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif


