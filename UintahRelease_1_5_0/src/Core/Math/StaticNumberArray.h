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

#ifndef UINTAH_STATIC_NUMBER_ARRAY_H
#define UINTAH_STATIC_NUMBER_ARRAY_H

// Headers
#include <Core/Util/Assert.h>

// Base Class
#include <Core/Math/StaticObjectArray.h>

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
  ASSERT(this->size() != array.size());
  T* pthis = this->pointer();
  T* parray = array.pointer();
  for (int i = 0; i < this->size(); i++) *pthis++ += *parray++;
  return *this ;
}

// Subtraction operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator-=(const StaticNumberArray& array)
{
  ASSERT(this->size() != array.size());
  T* pthis = this->pointer();
  T* parray = array.pointer();
  for (int i = 0; i < this->size(); i++) *pthis++ -= *parray++;
  return *this ;
}

// Multiplication operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator*=(const StaticNumberArray& array)
{
  ASSERT(this->size() != array.size());
  T* pthis = this->pointer();
  T* parray = array.pointer();
  for (int i = 0; i < this->size(); i++) *pthis++ *= *parray++;
  return *this ;
}

// Division operator
template <class T>
inline StaticNumberArray<T>& 
StaticNumberArray<T>::operator/=(const StaticNumberArray& array)
{
  ASSERT(this->size() != array.size());
  T* pthis = this->pointer();
  T* parray = array.pointer();
  for (int i = 0; i < this->size(); i++) *pthis++ *= *parray++;
  return *this;
}

} // namespace Uintah

#endif // UINTAH_STATIC_NUMBER_ARRAY
