
/*
 *  Array1.h: Interface to dynamic 1D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Array1_h
#define SCI_Containers_Array1_h 1

#include <sci_config.h>

#include <SCICore/Util/Assert.h>
#include <SCICore/Tester/RigorousTest.h>

namespace SCICore {

namespace PersistentSpace {
  class Piostream;
  void Pio();  // This is a dummy declaration to get things to compile.
}
namespace Containers {
  void Pio();  // This is a dummy declaration to get things to compile.
}
 namespace GeomSpace {
  void Pio();  // This is a dummy declaration to get things to compile.
 }
 namespace CoreDatatypes {
  void Pio();  // This is a dummy declaration to get things to compile.
 }
 namespace Geometry {
  void Pio();  // This is a dummy declaration to get things to compile.
 }

namespace Containers {

using SCICore::PersistentSpace::Piostream;
using SCICore::Tester::RigorousTest;

template<class T>

/**************************************

CLASS
   Array1
   
KEYWORDS
   Array1

DESCRIPTION
    Array1.h: Interface to dynamic 1D array class
  
    Written by:
     Steven G. Parker
     Department of Computer Science
     University of Utah
     March 1994
  
    Copyright (C) 1994 SCI Group
PATTERNS
   
WARNING
  
****************************************/
class Array1 {
    T* objs;
    int _size;
    int nalloc;
    int default_grow_size;
public:

    //////////
    //Copy the array - this can be costly, so try to avoid it.
    Array1(const Array1&);

    //////////
    //Make a new array 1. <i>size</i> gives the initial size of the array,
    //<i>default_grow_size</i> indicates the minimum number of objects that
    //should be added to the array at a time.  <i>asize</i> tells how many
    //objects should be allocated initially
    Array1(int size=0, int default_grow_size=10, int asize=-1);

    //////////
    //Copy over the array - this can be costly, so try to avoid it.
    Array1<T>& operator=(const Array1&);

    //////////
    //Deletes the array and frees the associated memory
    ~Array1();
    
    //////////
    // Accesses the nth element of the array
    inline const T& operator[](int n) const {
	ASSERTRANGE(n, 0, _size);
	return objs[n];
    }

    //////////
    // Accesses the nth element of the array
    inline T& operator[](int n) {
	ASSERTRANGE(n, 0, _size);
	return objs[n];
    }
    
    //////////
    // Returns the size of the array
    inline int size() const{ return _size;}


    //////////
    // Make the array larger by count elements
    void grow(int count, int grow_size=10);


    //////////
    // Add one element to the array.  equivalent to:
    //  grow(1)
    //  array[array.size()-1]=data
    void add(const T&);

    //////////
    // Insert one element into the array.  This is very inefficient
    // if you insert anything besides the (new) last element.
    void insert(int, const T&);


    //////////
    // Remove one element from the array.  This is very inefficient
    // if you remove anything besides the last element.
    void remove(int);


    //////////
    // Remove all elements in the array.  The array is not freed,
    // and the number of allocated elements remains the same.
    void remove_all();


    //////////
    // Change the size of the array.
    void resize(int newsize);

    //////////
    // Changes size, makes exact if currently smaller...
    void setsize(int newsize);

    //////////
    // Initialize all elements of the array
    void initialize(const T& val);


    //////////
    // Get the array information
    T* get_objs();


    //////////
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);

    friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Array1<T>&);
};

} // End namespace Containers
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included Array1.cc
//

#include <SCICore/Containers/String.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Malloc/Allocator.h>

#include <SCICore/Tester/RigorousTest.h>

namespace SCICore {
namespace Containers {

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
    _size=size;
    if(size){
	if(asize <= size){
	    objs=new T[size];
	    nalloc=_size;
	} else {
	    objs=new T[asize];
	    nalloc=asize;
	}
    } else {
	if(asize > 0){
	    objs=new T[asize];
	    nalloc=asize;
	} else {
	    objs=0;
	    nalloc=0;
	}
    }
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
#ifdef __GNUG__
  using namespace SCICore::GeomSpace;
  using namespace SCICore::PersistentSpace;
  using namespace SCICore::Geometry;
  using namespace SCICore::Containers;
  using namespace SCICore::CoreDatatypes;
#else
  using SCICore::GeomSpace::Pio;
  using SCICore::PersistentSpace::Pio;
  using SCICore::Geometry::Pio;
  using SCICore::Containers::Pio;
  using SCICore::CoreDatatypes::Pio;
#endif

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
void Pio(Piostream& stream, Containers::Array1<T>*& array) {
    if (stream.reading())
	array=new Array1<T>;
    Pio(stream, *array);
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/08/23 07:06:32  sparker
// Fix IRIX build
//
// Revision 1.5  1999/08/23 06:30:33  sparker
// Linux port
// Added X11 configuration options
// Removed many warnings
//
// Revision 1.4  1999/08/19 05:30:54  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
// Revision 1.3  1999/08/18 21:45:25  sparker
// Array1 const correctness, and subsequent fixes
// Array1 bug fix courtesy Tom Thompson
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

#endif /* SCI_Containers_Array1_h */

