//
//  class BoundedArray.h
//    array with explicit upper and lower bounds -- 
//    can have negative indices.
//
//    Uses templates.
//
//    Features:
//      1.  Build a array with an lower and upper indices.
//      Ex: Build a array of ints starting from 1 to n
//      2.  Uses the array base class
//      3.  Any method from array can be used on BoundedArray.
//
//    Usage:
//      BoundedArray <int> a_array(1,100);  // declare a array of ints
//                                            // with indices from 1->100.
//      BoundedArray <int> a_array(-100,100);  // declare a array of ints
//                                               // with indices from -100->100
//      BoundedArray <int> a_array(0,100,2); // declare a array of ints
//                                             // with indices from 1->100
//                                             // with initial value of 2
//
//
//

#ifndef __BOUNDED_ARRAY_H__
#define __BOUNDED_ARRAY_H__

#include "Array.h"
#include "Array.cc"

template<class T> class BoundedArray : public Array<T> {
protected:
        int lowbound;

public:

  // constructors
       BoundedArray();
       BoundedArray(int lowIndex, int highIndex);
       BoundedArray(int lowIndex, int highIndex, const T &initialValue);
  // copy constructor
       BoundedArray(const BoundedArray &source);

  //  no destructor -- use base class Array destructor 

  //  Assignment operator
       BoundedArray<T> & operator = (const BoundedArray<T> &source);

  // element access
       inline T & operator [] (int index) ;
       inline T operator[] (int index) const;

  // structural information
       int lowerBound() const;
       int upperBound() const;

  // multiplication operator BA * Matrix3
//       template<class T> BoundedArray<T> operator * (const BoundedArray<T> &left,
//                                       const Matrix3 &right);
};

  // inlined functions

template <class T> T &BoundedArray<T>::operator [] (int index)
{
  // Subscript operator for bounded Arrays
  // Subtract off lower bound
  // yielding value between 0 and size of Array
  // then use subscript from parent class

  return Array<T>::operator[](index - lowbound);

}

template <class T> T BoundedArray<T>::operator [] (int index) const
{
  // Subscript operator for bounded Arrays
  // Subtract off lower bound
  // yielding value between 0 and size of Array
  // then use subscript from parent class

  return Array<T>::operator[](index - lowbound);

}


#endif // __BOUNDED_ARRAY_H__


// $Log$
// Revision 1.1  2000/03/14 22:12:42  jas
// Initial creation of the utility directory that has old matrix routines
// that will eventually be replaced by the PSE library.
//
// Revision 1.1  2000/02/24 06:11:53  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:47  sparker
// Stuff may actually work someday...
//
// Revision 1.3  1999/02/25 05:41:42  guilkey
// Inlined some functions for performance.
//
// Revision 1.2  1999/01/25 23:18:00  campbell
// added ident capability.
//
