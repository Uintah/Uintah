
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

#ifndef SCI_Classlib_Array1_h
#define SCI_Classlib_Array1_h 1

#include <Classlib/Assert.h>

class Piostream;
class RigorousTest;

#ifdef __GNUG__
#pragma interface
#endif

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
    inline T& operator[](int n) const {
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
    

    friend void Pio(Piostream&, Array1<T>&);
    friend void Pio(Piostream&, Array1<T>*&);
};

template<class T> void Pio(Piostream&, Array1<T>&);
template<class T> void Pio(Piostream&, Array1<T>*&);

#endif /* SCI_Classlib_Array1_h */
