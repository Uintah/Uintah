
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

#include <SCICore/Util/Assert.h>
#include <SCICore/Persistent/Persistent.h>

namespace DaveW {
  namespace Datatypes {
    void Pio();  // This is a dummy declaration to get things to compile.
  }
}

namespace SCICore {

namespace Tester {
  class RigorousTest;
}

namespace GeomSpace {}
namespace Geometry {
  void Pio();  // This is a dummy declaration to get things to compile.
}

namespace Containers {

using SCICore::PersistentSpace::Piostream;
using SCICore::Tester::RigorousTest;

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
public:
    //////////
    //Create a 0X0 Array
    Array2();
    
    //////////
    //Array2 Copy Constructor
    Array2(const Array2&);
    
    //////////
    //Create an n by n array
    Array2(int, int);

    Array2<T>& operator=(const Array2&);
    
    //////////
    //Class Destructor
    ~Array2();

    //////////
    //Used for accessing elements in the Array
    inline T& operator()(int d1, int d2) const
	{
	    ASSERTL3(d1>=0 && d1<dm1);
	    ASSERTL3(d2>=0 && d2<dm2);
	    return objs[d1][d2];
	}
    
    //////////
    //Returns number of rows
    inline int dim1() const {return dm1;}
    
    //////////
    //Returns number of cols
    inline int dim2() const {return dm2;}
    
    //////////
    //Resize Array
    void newsize(int, int);
    
    //////////
    //Initialize all values in an array
    void initialize(const T&);

    inline T** get_dataptr() {return objs;}

    
    //////////
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);

    friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array2<T>&);
    friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array2<T>*&);

};

} // End namespace Containers
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included Array2.cc
//

#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Persistent/Persistent.h>

namespace SCICore {
namespace Containers {

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

#define Array2_VERSION 1

template<class T>
void Pio(Piostream& stream, Containers::Array2<T>& data)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using namespace SCICore::GeomSpace;
    using SCICore::Geometry::Pio;
    using DaveW::Datatypes::Pio;

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
void Pio(Piostream& stream, Containers::Array2<T>*& data) {
    if (stream.reading()) {
	data=new Array2<T>;
    }
    Pio(stream, *data);
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/24 06:24:00  dmw
// Added in everything for the DaveW branch
//
// Revision 1.3  1999/08/19 23:18:04  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.2  1999/08/17 06:38:34  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:11  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:34  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:41  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:28  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//
#endif

