#ifndef __BoundedArray_cc__
#define __BoundedArray_cc__

//  class BoundedArray 
//    array with explicit upper and lower bounds -- 
//    can have negative indices.
//    Uses templates.
//    Features:
//      1.  Build a array with an lower and upper indices.
//      Ex: Build a array of ints starting from 1 to n
//      2.  Uses the array base class
//      3.  Any method from array can be used on BoundedArray.
//    Usage:
//      BoundedArray <int> a_array(1,100);  // declare a array of ints
//                                            // with indices from 1->100.
//      BoundedArray <int> a_array(-100,100);  // declare a array of ints
//                                               // with indices from -100->100
//      BoundedArray <int> a_array(0,100,2); // declare a array of ints
//                                             // with indices from 1->100
//                                             // with initial value of 2



#include "BoundedArray.h"
#include <Core/Malloc/Allocator.h>


template <class T> BoundedArray<T>::BoundedArray():
  lowbound(0)
{
  // Constructor
  size = 0;
  universe_size = 0;
  data = 0;
}

template <class T> BoundedArray<T>::BoundedArray
	(int lowIndex, int highIndex) : Array<T>(1 + highIndex - lowIndex),
	  lowbound(lowIndex)
{
//  cout << lowIndex << " " <<  highIndex << " lowhigh" << endl;
  // Create and initialize a new bounded Array
  assert(lowIndex <= highIndex);
}

template <class T> BoundedArray<T>::BoundedArray
	(int lowIndex, int highIndex, const T &initialValue)
	: Array<T>(1 + highIndex - lowIndex, initialValue),
	  lowbound(lowIndex)
{
  // Create and initialize a new bounded Array with an initialvalue
  assert(lowIndex <= highIndex);
}

template <class T> BoundedArray<T>::BoundedArray
	(const BoundedArray<T> &source) 
	: Array<T>(source),lowbound(source.lowbound)
{
  // Copy constructor
  // no further initialization necessary
}

template <class T> BoundedArray<T> & BoundedArray<T>::operator = 
(const BoundedArray<T> &source)
{
  // Assignment operator

 if (data != 0)
    delete [] data;

  lowbound = source.lowbound;
  size = source.size;
  universe_size = source.universe_size;

  data = scinew T[universe_size];
  assert(data != 0);
  
  // copy the values

  for (unsigned int i = 0; i<size; i++) {
    data[i] = source.data[i];
  }

  return (*this);
}
							       
//template <class T> T &BoundedArray<T>::operator [] (int index)
//{
  // Subscript operator for bounded Arrays
  // Subtract off lower bound
  // yielding value between 0 and size of Array
  // then use subscript from parent class

//  return Array<T>::operator[](index - lowbound);

//}

//template <class T> T BoundedArray<T>::operator [] (int index) const
//{
  // Subscript operator for bounded Arrays
  // Subtract off lower bound
  // yielding value between 0 and size of Array
  // then use subscript from parent class

//  return Array<T>::operator[](index - lowbound);

//}

template <class T> int BoundedArray<T>::lowerBound() const
{
  // Return smallest legal index for bounded Array

  return lowbound;

}

template <class T> int BoundedArray<T>::upperBound() const
{
  // Return largest legal index for bounded Array

  return lowerBound() + length() - 1;

}
/*
template<class T> BoundedArray<T> operator * (const BoundedArray<T> &left,
                                                const Matrix3 &right)
{
  // Perform vector times Matrix3  multiplication to get another vector.
  // Vectors are represented by BoundedArrays.  The return is a BoundedArray.

  int array_col = left.length();

  // Check that they can be multiplied
  assert(array_col == 3);

  // Create space for the result
  BoundedArray<T> result (1,3,0.0);

  // Fill in the values
  for (int i = 1; i<= mat_col; i++) {
    for (int j = 1; j<= mat_row; j++) {
      result[i] += left[j]*right(j,i);
    }
  }

  // return the result
  return result;

}
*/

#endif

