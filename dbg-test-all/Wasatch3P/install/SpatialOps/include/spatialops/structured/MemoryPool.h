/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/**
 * \file MemoryPool.h
 */

#ifndef UT_MemoryPool_h
#define UT_MemoryPool_h

#include <stack>
#include <map>

#include <spatialops/structured/MemoryTypes.h>

namespace SpatialOps {

template<typename T>
class Pool{
  typedef std::stack<T*>              FieldQueue;
  typedef std::map<size_t,FieldQueue> FQSizeMap;
  typedef std::map<T*,size_t>         FieldSizeMap;

  static bool destroyed_;
  bool pinned_;

  FQSizeMap cpufqm_, gpufqm_;
  FieldSizeMap fsm_;
  size_t pad_;
  size_t cpuhighWater_, gpuhighWater_;
  Pool();
  ~Pool();
  Pool(const Pool&);
  Pool& operator=(const Pool&);

  static Pool& self();

  const unsigned short int deviceIndex_;

 public:

  /**
   * @brief Obtain a pointer to a block of memory with the requested size.
   *        If the requested size is zero, this returns a NULL pointer.
   * @param deviceLocation the location where the memory is requested
   * @param n the size of the block of memory
   *
   * Example: return a pointer to a block of memory for 5 doubles
   * \code
   * double* my5Doubles = Pool<double>::get( CPU_INDEX, 5 );
   * \endcode
   */
  static T* get( const short int deviceLocation, const size_t n );

  /**
   * @brief Return the requested block of memory to the pool. This should only
   *        be done for memory requested from the pool to begin with!
   * @param deviceLocation the location to return it to
   * @param t the pointer to the block of memory.
   */
  static void put( const short int deviceLocation, T* t );

  /**
   * @return the number of active fields in the pool on CPU
   */
  static size_t active();

  /**
   * @return the maximum number of active fields in the pool on CPU
   */
  static size_t total();
};

template<typename T> bool Pool<T>::destroyed_ = false;

} //namespace SpatialOps

#endif
