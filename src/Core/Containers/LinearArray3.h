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

/**************************************

 CLASS
 LinearArray3

 GENERAL INFORMATION

 LinearArray3.h

 Alan Humphrey, after Steven G. Parker
 Department of Computer Science
 University of Utah

 KEYWORDS
 LInearArray3

 DESCRIPTION
 Interface to streamlined dynamic linearized 3D array class.

 WARNING

 ****************************************/

#ifndef SCI_CONTAINERS_LINEARARRAY3_h
#define SCI_CONTAINERS_LINEARARRAY3_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Util/Assert.h>

#include <complex>
#include <string>

namespace SCIRun {

using namespace Uintah;

template<class T> class LinearArray3 {

  public:

    /**
     * @brief Default constructor
     * @param none
     */
    LinearArray3();

    /**
     * @brief 3 argument constructor
     * @param dim1 The first dimension of this LinearArray3
     * @param dim2 The second dimension of this LinearArray3
     * @param dim3 The third dimension of this LinearArray3
     */
    LinearArray3(int dim1,
                 int dim2,
                 int dim3);

    /**
     * @brief Copy constructor
     * @param copy The LinearArray3 object to copy from
     */
    LinearArray3(const LinearArray3& copy);

    /**
     * @brief Destructor
     * @param None
     */
    ~LinearArray3();

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
      ASSERTL3(d1>=0 && d1<dm1);ASSERTL3(d2>=0 && d2<dm2);ASSERTL3(d3>=0 && d3<dm3);
      int idx = (d1) + ((d2) * dm1) + ((d3) * dm1 * dm2);
      return objs[idx];
    }

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
     * @brief Returns the size in bytes of this LinearArray3
     * @param None
     * @return The size in bytes of this LinearArray3
     */
    inline long get_datasize() const
    {
      return dm1 * long(dm2 * dm3 * sizeof(T));
    }

    /**
     * @brief Resize the linearized 3D objects array
     * @param dim1 The first dimension of the new LinearArray3
     * @param dim2 The second dimension of the new LinearArray3
     * @param dim3 The third dimension of the new LinearArray3
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

    /**
     * @brief Assignment operator
     * @param other The assignee of this assignment operation
     * @return A reference to the new LinearArray3 object after assignment
     */
    inline LinearArray3<T>& operator=(const LinearArray3& other)
    {
      resize(other.dim1(), other.dim2(), other.dim3());
      for (int i = 0; i < dm1; i++) {
        for (int j = 0; j < dm2; j++) {
          for (int k = 0; k < dm3; k++) {
            int idx = (i) + ((j) * dm1) + ((k) * dm1 * dm2);
            objs[idx] = other.objs[idx];
          }
        }
      }
      return *this;
    }

    //! support dynamic compilation
    static const string& get_h_file_path();

    SCISHARE const TypeDescription* get_type_description(LinearArray3<T>*);

  private:

    T* objs;
    int dm1;
    int dm2;
    int dm3;

    void allocate();

};

}  // End namespace SCIRun

#endif

