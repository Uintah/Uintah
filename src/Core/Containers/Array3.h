/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */

#ifndef SCI_Containers_Array3_h
#define SCI_Containers_Array3_h 1

#ifndef SCI_NOPERSISTENT
#include <sci_defs/template_defs.h>
#endif // #ifndef SCI_NOPERSISTENT

#include <iostream>
#include <stdio.h>
#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <Core/Util/Assert.h>

#ifndef SCI_NOPERSISTENT
#include <Core/Persistent/Persistent.h>
#endif // #ifndef SCI_NOPERSISTENT


namespace SCIRun {

class RigorousTest;

template<class T> class Array3;
#ifndef SCI_NOPERSISTENT
template<class T> void Pio(Piostream& stream, Array3<T>& array);
template<class T> void Pio(Piostream& stream, Array3<T>& array, const std::string&);
template<class T> void Pio(Piostream& stream, Array3<T>*& array);
#endif // #ifndef SCI_NOPERSISTENT

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

 PATTERNS

 WARNING
  
****************************************/

template<class T> class Array3 {
  T*** objs;
  int dm1;
  int dm2;
  int dm3;
  void allocate();

  // The copy constructor and the assignment operator have been
  // privatized on purpose -- no one should use these.  Instead,
  // use the default constructor and the copy method.
  //////////
  //Copy Constructor
  Array3(const Array3&);
  //////////
  //Assignment Operator
  Array3<T>& operator=(const Array3&);
public:
  //////////
  //Default Constructor
  Array3();
    
  //////////
  //Constructor
  Array3(int, int, int);
    
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
  //Array2 Copy Method
  void copy(const Array3&);

  //////////
  //Returns the number of spaces in dim1	    
  inline int dim1() const {return dm1;}
  //////////
  //Returns the number of spaces in dim2
  inline int dim2() const {return dm2;}
  //////////
  //Returns the number of spaces in dim3
  inline int dim3() const {return dm3;}
  
  inline long get_datasize() const { return dm1*long(dm2*dm3*sizeof(T)); }
    
  //////////
  //Re-size the Array
  void resize(int, int, int);

  //////////
  //Initialize all elements to T
  void initialize(const T&);

  T* get_onedim();
  void get_onedim_byte( unsigned char *v );

  inline T*** get_dataptr() {return objs;}

  //////////
  //read/write from a separate raw file
  int input( const std::string& );
  int output( const std::string&);

#ifndef SCI_NOPERSISTENT
#if defined(_AIX)
  template <typename type>
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<type>&);
  template <typename type>
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<type>&,
					     const std::string &);
  template <typename type>
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<type>*&);
#else
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<T>&);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<T>&, 
					     const std::string &);
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array3<T>*&);
#endif
#endif // #ifndef SCI_NOPERSISTENT
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
  if( dm1 && dm2 && dm3 ){
    objs=new T**[dm1];
    T** p=new T*[dm1*dm2];
    T* pp=new T[dm1*long(dm2*dm3)];
    for(int i=0;i<dm1;i++){
      objs[i]=p;
      p+=dm2;
      for(int j=0;j<dm2;j++){
	objs[i][j]=pp;
	pp+=dm3;
      }
    }
  } else {
    objs = 0;
  }
}

template<class T>
void Array3<T>::resize(int d1, int d2, int d3)
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
void Array3<T>::copy(const Array3<T> &copy)
{
  resize( copy.dim1(), copy.dim2(), copy.dim3() );
  for(int i=0;i<dm1;i++)
    for(int j=0;j<dm2;j++)
      for(int k=0;k<dm3;k++)
        objs[i][j][k] = copy.objs[i][j][k];
}

template<class T>
T* Array3<T>::get_onedim()
{
  int i,j,k, index;
  T* a = new T[dm1*long(dm2*dm3)];
  
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

#ifndef SCI_NOPERSISTENT

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
    data.resize(d1, d2, d3);
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
         const std::string& filename)
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
    data.resize(d1, d2, d3);
    data.input( filename );
  } else {
    Pio(stream, data.dm1);
    Pio(stream, data.dm2);
    Pio(stream, data.dm3);
    data.output( filename );
  }
    
  stream.end_class();
}
#endif // #ifndef SCI_NOPERSISTENT

template<class T>
int
Array3<T>::input( const std::string &filename ) 
{
  std::cerr << "Array3: Split input\n";

  // get raw data
  int file=open( filename.c_str(), O_RDONLY, 0666);
  if ( file == -1 ) {
    printf("can not open file %s\n", filename.c_str());
    return 0;
  }
  
  int maxiosz=1024*1024;
  long size = dm1*long(dm2*dm3*sizeof(T));
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
Array3<T>::output( const std::string &filename ) 
{
  std::cerr << "Array3 output to " << filename << std::endl;
  // get raw data
  //  printf("output [%s] [%s]\n", filename.c_str(), rawfile() );
  int file=open( filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0666);
  if ( file == -1 ) {
    perror("open file");
    return 0;
  }
  
  const int maxiosz=1024*1024;

  const long size = dm1*long(dm2*dm3*sizeof(T));
  int n = size / maxiosz;
  const char *p = (char *)objs[0][0];

  printf("Start writing...%ld %d %i\n", size, maxiosz, n);

  for ( ; n> 0 ; n--, p+= maxiosz) {
    const int l = write( file, p, maxiosz);
    if ( l != maxiosz ) 
      perror("write ");
  }
  const int sz = (size % maxiosz );
  const int l = write( file, p, sz);
  if ( l != (size % maxiosz ) ) {
    printf("Error: wrote %d / %ld\n", l,(size % maxiosz ));
    perror("write ");
  }
        
  fsync(file);
  close(file);

  return 1;
} 

} // End namespace SCIRun


#endif


