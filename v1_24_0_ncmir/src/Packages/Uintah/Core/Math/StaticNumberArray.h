#ifndef UINTAH_STATIC_NUMBER_ARRAY_H
#define UINTAH_STATIC_NUMBER_ARRAY_H

// Headers
#include <Core/Util/Assert.h>

// Base Class
#include <Packages/Uintah/Core/Math/StaticObjectArray.h>

namespace Uintah {

  /*! \class StaticNumberArray
   *  \brief Templated base class for arrays of number types.  The
   *         number type must have the following mathematical operators
   *         defined. \n
   *         (1) {+=, -=, *=, /=} : MUST return references to this\n
   *         (3) must allow assignment.
   *  \author  Biswajit Banerjee, 
   *  \author  C-SAFE and Department of Mechanical Engineering,
   *  \author  University of Utah.
   *  \author  Copyright (C) 2003 Container Dynamics Group

   Based on Tahoe implementation of nArrayT.h v.1.20.
  */

  template <class T>
    class StaticNumberArray : public StaticObjectArray<T>
    {
    public:

      /** Construct an array of the specified length. The values
       * in the array are not initialized.
       * \param size length of allocated space */
      StaticNumberArray(unsigned int size);

      /** Copy constructor */
      StaticNumberArray(const StaticNumberArray& array);

      /** Destructor */
      ~StaticNumberArray();

      /** Assignment operator */
      StaticNumberArray<T>& operator=(const StaticNumberArray& array); 

      /** Element-by-element addition with RHS */
      StaticNumberArray<T>& operator+=(const StaticNumberArray& array); 

      /** Element-by-element subtraction with RHS */
      StaticNumberArray<T>& operator-=(const StaticNumberArray& array); 

      /** Element-by-element multiplication by RHS */
      StaticNumberArray<T>& operator*=(const StaticNumberArray& array); 

      /** Element-by-element division by RHS 
          No checks for division by zero or indeterminate forms */
      StaticNumberArray<T>& operator/=(const StaticNumberArray& array);

    private:
      StaticNumberArray();

    };

// Implementation of StaticNumberArray

// Standard constructor : Create array of given size 
template <class T>
StaticNumberArray<T>::StaticNumberArray(unsigned int size): 
  StaticObjectArray<T>(size) 
{ 
}

// Copy constructor : Create array from given array
template <class T>
StaticNumberArray<T>::StaticNumberArray(const StaticNumberArray& array):
  StaticObjectArray<T>(array) 
{ 
}

// Destructor
template <class T>
StaticNumberArray<T>::~StaticNumberArray()
{
}
  
// Assignment operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator=(const StaticNumberArray& array)
{
  StaticObjectArray<T>::operator=(array);
  return *this;   
}

// Addition operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator+=(const StaticNumberArray& array)
{
  ASSERT(size() != array.size());
  T* pthis = pointer();
  T* parray = array.pointer();
  for (int i = 0; i < size(); i++) *pthis++ += *parray++;
  return *this ;
}

// Subtraction operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator-=(const StaticNumberArray& array)
{
  ASSERT(size() != array.size());
  T* pthis = pointer();
  T* parray = array.pointer();
  for (int i = 0; i < size(); i++) *pthis++ -= *parray++;
  return *this ;
}

// Multiplication operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator*=(const StaticNumberArray& array)
{
  ASSERT(size() != array.size());
  T* pthis = pointer();
  T* parray = array.pointer();
  for (int i = 0; i < size(); i++) *pthis++ *= *parray++;
  return *this ;
}

// Division operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator/=(const StaticNumberArray& array)
{
  ASSERT(size() != array.size());
  T* pthis = pointer();
  T* parray = array.pointer();
  for (int i = 0; i < size(); i++) *pthis++ *= *parray++;
  return *this;
}

} // namespace Uintah

#endif // UINTAH_STATIC_NUMBER_ARRAY
