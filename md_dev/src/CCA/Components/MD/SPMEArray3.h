/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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
 *  @class MD
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper (after Steve Parker)
 *  @date   May, 2013
 *
 *  @brief Interface to streamlined dynamic 3D array class.
 *
 *  @param
 */

#ifndef UINTAH_MD_ELECTROSTATICS_SPME_ARRAY3_H
#define UINTAH_MD_ELECTROSTATICS_SPME_ARRAY3_H

#include <Core/Util/Assert.h>

#include <iostream>
#include <stdio.h>

#include <sci_defs/fftw_defs.h>

namespace Uintah {

using namespace SCIRun;

typedef std::complex<double> dblcomplex;

template<class T> class SPMEArray3 {

  public:

    /**
     * @brief Default constructor
     * @param none
     */
    SPMEArray3();

    /**
     * @brief 3 argument constructor
     * @param dim1 The first dimension of this SPMEArray3
     * @param dim2 The second dimension of this SPMEArray3
     * @param dim3 The third dimension of this SPMEArray3
     */
    SPMEArray3(int dim1,
               int dim2,
               int dim3);

    /**
     * @brief Destructor
     * @param None
     */
    ~SPMEArray3();

    /**
     * @brief Access the nXnXn element of the linearized 3D array
     * @param d1 The first coordinate dimension
     * @param d2 The second coordinate dimension
     * @param d3 The third coordinate dimension
     * @return T& A reference to the nXnXn element
     */
    inline T& operator()(int d1,
                         int d2,
                         int d3) const
    {
      ASSERTL3(d1>=0 && d1<dm1);
      ASSERTL3(d2>=0 && d2<dm2);
      ASSERTL3(d3>=0 && d3<dm3);
      int idx = (d1) + ((d2) * dm1) + ((d3) * dm2 * dm3);
      return objs[idx];
    }

    /**
     * @brief SPMEArray3 copy method
     * @param copy The SPMEArray3 source object to copy from
     * @return None
     */
    void copy(const SPMEArray3& copy);

    /**
     * @brief Returns the number of elements in dimension 1
     * @param None
     * @return int The number of elements in dimension 1
     */
    inline int dim1() const
    {
      return dm1;
    }

    /**
     * @brief Returns the number of elements in dimension 2
     * @param None
     * @return int The number of elements in dimension 2
     */
    inline int dim2() const
    {
      return dm2;
    }

    /**
     * @brief Returns the number of elements in dimension 3
     * @param None
     * @return int The number of elements in dimension 3
     */
    inline int dim3() const
    {
      return dm3;
    }

    /**
     * @brief Returns the size in bytes of this SPMEArray3
     * @param None
     * @return The size in bytes of this SPMEArray3
     */
    inline long get_datasize() const
    {
      return dm1 * long(dm2 * dm3 * sizeof(T));
    }

    /**
     * @brief Resize the linearized 3D objects array
     * @param dim1 The first dimension of the new SPMEArray3
     * @param dim2 The second dimension of the new SPMEArray3
     * @param dim3 The third dimension of the new SPMEArray3
     */
    void resize(int dim1,
                int dim2,
                int dim3);

    /**
     * @brief Initialize all objects elements to T
     * @param T The value to initialize all objects elements to
     * @return None
     */
    void initialize(const T&);

    /**
     * @brief Returns a pointer to the linearized 3D objects array
     * @param None
     * @return A pointer to the linearized 3D objects array
     */
    inline T* get_dataptr()
    {
      return objs;
    }

  private:

    T* objs;
    int dm1;
    int dm2;
    int dm3;
    void allocate();

    // The copy constructor and the assignment operator have been
    // privatized on purpose -- no one should use these.  Instead,
    // use the default constructor and the copy method.

    /**
     * @brief Copy Constructor
     * @param copy The SPMEArray3 object to copy from
     */
    SPMEArray3(const SPMEArray3& copy);

    /**
     * @brief Assignment operator
     * @param other The assignee of this assignment operation
     * @return A reference to the new SPMEArray3 object after assignment
     */
    SPMEArray3<T>& operator=(const SPMEArray3& other);

};

}  // End namespace Uintah

#endif

