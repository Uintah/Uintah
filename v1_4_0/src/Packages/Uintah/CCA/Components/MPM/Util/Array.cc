#ifndef __Array_cc__
#define __Array_cc__

//  Array.cc
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



#include "Array.h"

					    
template<class T> Array<T>::Array():size(0),universe_size(0)
{
  // constructor
  // no further initialization

  data = 0;
}

template<class T> Array<T>::Array(unsigned int numberElements)
   : size(numberElements)
{
   // Create and initialize a new Array
   // allocate the space for the elements

   				 
   if (size <= MINIMUM_INCREMENT) {
         universe_size = MINIMUM_INCREMENT;
	 
   }
   else {
         universe_size = numberElements;
	 
   }

   data = new T[universe_size];

   // check that allocation was successful

   assert(data != 0);

}

template<class T> Array<T>::Array(unsigned int numberElements,
				   const T &initialValue)
  : size(numberElements)
{
  // Create and initialize a new Array
  // allocate the space for the elements and
  // set each element to initialValue

  if (size <= MINIMUM_INCREMENT ) {
    universe_size = MINIMUM_INCREMENT;
   
  }
  else {
    universe_size = numberElements;
  }
				    

  data = new T[universe_size];

  // check that allocation was successful

  assert(data != 0);

  // set each element to initialValue

  for (unsigned int i = 0; i<size; i++) {
    data[i] = initialValue;
  }

}

template<class T> Array<T>::Array(const Array<T> &source)
  : size(source.size)
{
  // copy constructor for the Array class

  // create and initialize a new Array
  // allocate the space for the lements

  data = new T[size];
  assert(data != 0);

  // copy values from old Array

  for (unsigned int i = 0; i < size; i++) {
    data[i] = source.data[i];
  }

  universe_size = source.universe_size;

}

//template<class T> Array<T>::~Array()
//{

  // destructor

//  delete [] data;
//  data = 0;
//  size = 0;
//  universe_size = 0;

//}

template<class T> Array<T>  & Array<T>:: operator = (const Array<T> &source)
{
  // Assignment operator
 
    
  if (data != 0)
    delete [] data;

  size = source.size;
  universe_size = source.universe_size;

  data = new T[universe_size];
  assert(data != 0);
  
  // copy the values

  for (unsigned int i = 0; i<size; i++) {
    data[i] = source.data[i];
  }

  return (*this);
}

  

//template<class T> T & Array<T>::operator[] (unsigned int index)
//{
  // subscript a Array value
  // check that the index is valid

//  assert(index < size);

  // return requested element

//  return data[index];

//}

//template<class T> T Array<T>::operator[] (unsigned int index) const
//{
  // subscript an Array value
  // check that the index is valid

//  assert(index < size);

  // return request element

//  return data[index];
//}

template<class T> unsigned int Array<T>::length() const
{
  // return the number of elements in the Array

  return size;

}

template<class T> unsigned int Array<T>::univ_size() const
{

  // return the universe size of the Array

  return universe_size;
}

template<class T> 
unsigned int Array<T>::setSize(unsigned int numberOfElements,
				const T &initialValue)
{
  // dynamically change the size of the Array

  
  if (numberOfElements <= size) {
    // data area is shrinking - copy initialValue to data space
         for (unsigned int i = 0; i < numberOfElements; i++){
        data[i] = initialValue;
    }
    // Set size to numberOfElements
	 size = numberOfElements;
  }
    else {
      // data area is growing

      // Check if numberOfElements greater than universe_size;

      if (numberOfElements > universe_size) {

      // Allocate space 
         T * newData = new T[numberOfElements];
         assert(newData != 0);

      //  copy old values
	 for (unsigned int i = 0; i < size; i++) {
	      newData[i] = data[i];
	 }

        // Initialize new values;
         for (unsigned int i = size; i < numberOfElements; i++) {
	   newData[i] = initialValue;
         }

	 // Update data member fields;
	 delete [] data;
	 data = newData;
         size = numberOfElements;
         universe_size = numberOfElements;
      }
      else {
	for (unsigned int i = size; i < numberOfElements; i++) {
	  data[i] = initialValue;
	}
	size = numberOfElements;
      }
    }
    
   
  return size;
}


template<class T> 
unsigned int Array<T>::setSize(unsigned int numberOfElements)
{
  // dynamically change the size of the Array

  
  if (numberOfElements <= size) {
    // data area is shrinking -
    // Set size to numberOfElements
    size = numberOfElements;
  }
  else {
    // data area is growing
    // Check if numberOfElements greater than universe_size;
    if (numberOfElements > universe_size) {

      // Allocate space 
      T * newData = new T[numberOfElements];
      assert(newData != 0);

      //  copy old values
      for (unsigned int i = 0; i < size; i++) {
	newData[i] = data[i];
      }

	 
      // Update data member fields;
      delete [] data;
      data = newData;
      size = numberOfElements;
      universe_size = numberOfElements;
    }
    else {
      size = numberOfElements;
    }
  }
    
    return size;
}


template<class T>
void Array<T>::append(const T &initialValue)
{

  // append an element to the Array

  //  check to see if there is space available

    if (size < universe_size) {
      // append the new value
       data[size] = initialValue;
      // update the data member field
       size++;
    }
    else {
      T *newData = new T[universe_size + MINIMUM_INCREMENT];
      assert(newData != 0);
      // copy the old values to the new data space
      for (unsigned int i = 0; i < size; i++) {
	newData[i] = data[i];
      }
      // append the new value to the new data space
      newData[size] = initialValue;
      // delete the old data space
      delete [] data;

     // update the data member fields and 
     // move new data space to old data space
      data = newData;
      size++;
      universe_size += MINIMUM_INCREMENT;
    }
}

template <class T>
void Array<T>::del()
{
  // Delete the last element from the Array

  assert(size > 0);

  size--;
}
      
template <class T>
void Array<T>::del(unsigned int indx)
{
  // Delete the entry associated with indx # from the Array.
  // Shift values down

  assert(size > 0);

  for (unsigned int i = indx; i < size - 1 ; i++) {
    data[i] = data[i+1];
  }
  size--;

}


template <class T> T Array<T>::largest()
{
  // Find the largest element in the Array

  T big = data[0];
  T temp;

  for (unsigned int i = 0; i < size; i++) {
    temp = data[i];
    if (temp > big) {
      big = temp;
    }
  }

  return big;
}

template <class T> T Array<T>::largest_mag()
{
  // Find the element in the Array with the largest absolute value

  T big = data[0];
  T temp,n_temp;
 // Find the absolute value of the first element assigned to big
  if (big >= 0)
    temp = big;
  else
    temp = -(big);

  big = temp;

  for (unsigned int i = 0; i < size; i++) {
    // Find the absolute value of the element
    temp = data[i];
    if (temp >= 0) {
      n_temp = temp;
    }
    else {
      n_temp = -(temp);
    }
    if (n_temp > big) {
      big = n_temp;
    }
  }

  return big;
}

template <class T> T Array<T>::smallest()
{
  // Find the smallest element in the Array
  T small = data[0];
  T temp;

  for (unsigned int i = 0; i < size; i++) {
    temp = data[i];
    if (temp < small) {
      small = temp;
    }
  }

  return small;
}

template <class T> T Array<T>::smallest_mag()
{
  // Find the element in the Array with the smallest magnitude

  T small = data[0];
  T temp,n_temp;
  // Find the absolute value of the first element assigned to small
  if (small >= 0)
    temp = small;
  else
    temp = -(small);

  small = temp;

  for (unsigned int i = 0; i < size; i++) {
    // Find the absolute value of the element
    temp = data[i];
    if (temp >= 0) {
      n_temp = temp;
    }
    else {
      n_temp = -(temp);
    }
    if (n_temp < small) {
      small = n_temp;
    }
  }

  return small;
}

template <class T> T Array<T>::sum()
{
  // Find the sum of all the elements in the Array

  T total = 0;

  for (unsigned int i = 0; i< size; i++) {
    total += data[i];
  }

  // Return the result

  return total;
}

template <class T> void Array<T>::set(const T &value)
{
  // Set each element in Array to value

  for (unsigned int i = 0; i< size; i++) {
    data[i] = value;
  }

}

#endif

