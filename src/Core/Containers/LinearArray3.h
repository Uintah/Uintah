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
 LinearArray3

 DESCRIPTION
 Interface to streamlined dynamic linearized 3D array class.

 WARNING

 ****************************************/

#ifndef SCI_CONTAINERS_LINEARARRAY3_h
#define SCI_CONTAINERS_LINEARARRAY3_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Assert.h>

#include <complex>
#include <string>

namespace SCIRun {

// Added for compatibility with core types
class TypeDescription;

using namespace Uintah;

typedef std::complex<double> dblcomplex;

template<class T> class LinearArray3 {

  public:

    /**
     * @brief Default constructor
     * @param none
     */
    LinearArray3();

    /**
     * @brief 3 argument constructor.
     * @param dim1 The first dimension of this LinearArray3.
     * @param dim2 The second dimension of this LinearArray3.
     * @param dim3 The third dimension of this LinearArray3.
     */
    LinearArray3(int dim1,
                 int dim2,
                 int dim3);

    /**
     * @brief 4 argument constructor, with default value.
     * @param dim1 The first dimension of this LinearArray3.
     * @param dim2 The second dimension of this LinearArray3.
     * @param dim3 The third dimension of this LinearArray3.
     * @param value The single, default value to initialize every element to.
     */
    LinearArray3(int dim1,
                 int dim2,
                 int dim3,
                 T value);

    /**
     * @brief Copy constructor.
     * @param copy The LinearArray3 object to copy from.
     */
    LinearArray3(const LinearArray3& copy);

    /**
     * @brief Default destructor.
     * @param None
     */
    ~LinearArray3();

    /**
     * @brief Access the nXnXn element of the linearized 3D array.
     * @param d1 The first coordinate dimension.
     * @param d2 The second coordinate dimension.
     * @param d3 The third coordinate dimension.
     * @return T& A reference to the nXnXn element.
     */
    inline T& operator()(int d1,
                         int d2,
                         int d3) const
    {
      ASSERTL3(d1>=0 && d1<dm1);
      ASSERTL3(d2>=0 && d2<dm2);
      ASSERTL3(d3>=0 && d3<dm3);
      int idx = (d1) + ((d2) * dm1) + ((d3) * dm1 * dm2);
      return objs[idx];
    }

    /**
     * @brief Returns the number of elements in dimension 1.
     * @param None
     * @return int The number of elements in dimension 1.
     */
    inline int dim1() const
    {
      return dm1;
    }

    /**
     * @brief Returns the number of elements in dimension 2.
     * @param None
     * @return int The number of elements in dimension 2.
     */
    inline int dim2() const
    {
      return dm2;
    }

    /**
     * @brief Returns the number of elements in dimension 3.
     * @param None
     * @return int The number of elements in dimension 3.
     */
    inline int dim3() const
    {
      return dm3;
    }

    /**
     * @brief Returns the size in bytes of this LinearArray3.
     * @param None
     * @return The size in bytes of this LinearArray3.
     */
    inline long getDataSize() const
    {
      return dm1 * long(dm2 * dm3 * sizeof(T));
    }

    /**
     * @brief Returns the number of elements in this LinearArray3.
     * @param None
     * @return The number of elements in this LinearArray3.
     */
    inline long getSize() const
    {
      return long(dm1 * dm2 * dm3 );
    }

    /**
     * @brief Resize the linearized 3D objects array.
     * @param dim1 The first dimension of the new LinearArray3.
     * @param dim2 The second dimension of the new LinearArray3.
     * @param dim3 The third dimension of the new LinearArray3.
     */
    void resize(int dim1,
                int dim2,
                int dim3);

    void copyData(const LinearArray3<T>& copy);

    /**
     * @brief Initialize all objects elements to T.
     * @param T The value to initialize all objects elements to.
     * @return None
     */
    void initialize(const T&);

    /**
     * @brief Returns a pointer to the linearized 3D objects array.
     * @param None
     * @return A pointer to the linearized 3D objects array.
     */
    inline T* get_dataptr()
    {
      return objs;
    }

    /**
     * @brief Assignment operator.
     * @param other The assignee of this assignment operation.
     * @return A reference to the new LinearArray3 object after assignment.
     */
    inline LinearArray3<T>& operator=(const LinearArray3& other)
    {
      resize(other.dim1(), other.dim2(), other.dim3());
      long int size = other.getSize();
      for (long int idx = 0; idx < size; idx++) {
        objs[idx] = other.objs[idx];
      }
      return *this;
    }

    /**
     * @brief Addition of LinearArray3; check size and extents.
     * @param addend The addend.
     * @return LinearArray3<T> The result of the addition.
     */
    inline LinearArray3<T> operator+(const LinearArray3<T> &addend) const
    {
      // Add a LinearArray3 to a LinearArray3
      long int size = getSize();
      ASSERTEQ(size, addend.getSize());ASSERTL3(dm1==addend.dm1 && dm2==addend.dm2 && dm3==addend.dm3);
      LinearArray3<T> la3(dm1, dm2, dm3);
      for (long int idx = 0; idx < size; idx++) {
        la3.objs[idx] = objs[idx] + addend.objs[idx];
      }
      return la3;
    }

    /**
     * @brief In place LinearArray3 element by element accumulation.
     * @param arrayIn The addend.
     * @return LinearArray3<T>& The result of the addition on this LinearArray3.
     */
    inline LinearArray3<T>& operator+=(const LinearArray3<T>& arrayIn)
    {
      long int size = getSize();
      for (long int idx = 0; idx < size; idx++) {
        objs[idx] += arrayIn.objs[idx];
      }
      return *this;
    }

    /**
     * @brief Check for LinearArray3 equality; element for element.
     * @param None
     * @return bool Ture if the specified LinearArray3 has the same values for all "obj" elements, false otherwise.
     */
    inline bool operator==(const LinearArray3<T> &other) const
    {
      long int size = getSize();
      ASSERTEQ(size, other.getSize());ASSERTL3(dm1==other.dm1 && dm2==other.dm2 && dm3==other.dm3);
      LinearArray3<T> la3(dm1, dm2, dm3);
      for (long int idx = 0; idx < size; idx++) {
        if (other.objs[idx] != objs[idx]) {
          return false;
        }
      }
      return true;
    }

    //! support dynamic compilation
    static const std::string& get_h_file_path();

  private:

    T* objs;
    int dm1;
    int dm2;
    int dm3;

    void allocate();

    friend std::ostream& operator <<(std::ostream& out_file,
                                     const LinearArray3<dblcomplex> &la3);

};
// end class LinearArray3

std::ostream& operator <<(std::ostream& os,
                          const LinearArray3<dblcomplex>& la3);

void swapbytes(LinearArray3<dblcomplex>& array);

const std::string find_type_name(LinearArray3<std::complex<double> >*);

const TypeDescription* get_type_description(LinearArray3<std::complex<double> >*);

}  // End namespace SCIRun

namespace Uintah {

const TypeDescription* fun_getTypeDescription(SCIRun::LinearArray3<std::complex<double> >*);

}

#endif

