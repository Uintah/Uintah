//  Array.h 
//  class Array
//    Type safe and access safe Array data type
//    permits dynamic modification of Array length.
//    Uses templates.
//    Features:
//      1.  dynamically resize Arrays - smaller or larger.
//      2.  will allocate a larger pool of memory to reduce
//          dynamic memory allocation when Array changes size
//          frequently.
//      3.  append elements to end of Array
//      4.  delete elements
//      5.  Minimum Array size allocated is 100 elements
//      Usage:
//        Array<int> a_Array(10);     // declare a Array of 10 ints
//        Array<int> a_Array(10,2);   // declare a Array of 10 ints with
//                                      // initial value of 2
//        Array<int> b_Array(a_Array);  // copy the contents of a_Array
//                                         // to b_Array;
//        a_Array.~Array();           //  destroy a Array
//        int a = a_Array[#];          //  access the #th element of a_Array
//        a_Array.length();            //  determine the length of the Array
//        a_Array.univ_size();         //  determine the amount of space 
//                                      //  allocated for a_Array
//        a_Array.setSize(20);         //  dynamically change size of a_Array
//                                      //  to a length of 20
//        a_Array.setSize(20,4);       //  dynamically change size of a_Array
//                                      //  to a length of 20 with initial 
//                                      //  value of 4
//        a_Array.append(5);           //  append element with value of 4 to
//                                      //  a_Array
//        a_Array.del();               //  delete the last element in a_Array
//        a_Array.del(#);              //  delete #th element from a_Array

#ifndef __ARRAY_H__
#define __ARRAY_H__



#include <assert.h>
#define MINIMUM_INCREMENT 1


template <class T> class Array {
 protected:
  // data areas
       T *data;
       unsigned int size;
       unsigned int universe_size;

 public:
  // constructors and destructor
       Array();
       Array(unsigned int numberElements);
       Array(unsigned int numberElements,const T &initialValue);
  // copy constructor
       Array(const Array<T> &source); 
  // destructor 
       inline virtual ~Array();             

  // access to elements via subscript
       inline T & operator[](unsigned int index);
       inline T operator[](unsigned int index) const;

  // assignment operator
       Array<T>  & operator = (const Array<T> &source);
       

  // length of Array
       unsigned int length() const;

  // universe size of Array -- amount of space allocated
       unsigned int univ_size() const;

  // dynamically change size
       unsigned int  setSize(unsigned int numberOfElements);
       unsigned int  setSize(unsigned int numberOfElements,
			     const T &intialValue);

 //  append an element to the Array

       void append(const T &initialValue);

 //  delete an element from the Array - last element only

       void del();

 //  delete index number from Array

       void del(unsigned int indx);

 //  find the largest element in the Array
       T largest();

 //  find the element in the Array with the largest magnitude
       T largest_mag();

 //  find the smallest element in the Array
       T smallest();

  // find the element in the Array with the smallest magnitude
      T smallest_mag();

 // find the sum of all the elements
      T sum();

  // Set each value in Array to T

      void set(const T &value);

};

  // inlined functions

template<class T> T & Array<T>::operator[] (unsigned int index)
{
  // subscript a Array value
  // check that the index is valid

  assert(index < size);

  // return requested element

  return data[index];

}

template<class T> T Array<T>::operator[] (unsigned int index) const
{
  // subscript an Array value
  // check that the index is valid

  assert(index < size);

  // return request element

  return data[index];
}

template<class T> Array<T>::~Array()
{

  // destructor

  delete [] data;
  data = 0;
  size = 0;
  universe_size = 0;

}

#include "Array.cc"

#endif  // __ARRAY_H__ 

