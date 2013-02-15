/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
//---------------------------------------------------------------------------------------------------------------------------------
#ifndef UINTAH_MD_SIMPLEGRID_H
#define UINTAH_MD_SIMPLEGRID_H

#include <Core/Geometry/IntVector.h>
#include <Core/Containers/Array3.h>
#include <Core/Grid/Patch.h>
#include <Core/Util/Assert.h>

#include <sci_defs/fftw_defs.h>

namespace Uintah {

typedef std::complex<double> dblcomplex;

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::IntVector;

class Patch;

/**
 *  @class SimpleGrid
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   December, 2012
 *
 *  @brief A lightweight representation of the local grid and supporting operations involved in the SPME routine.
 *
 *  @param T The data type for this SimpleGrid. Should be double, std::complex<double>> or a std::vector of these
 */
template<typename T> class SimpleGrid {

  public:

    /**
     * @brief Default Constructor.
     */
    SimpleGrid();

    /**
     * @brief Destructor
     */
    virtual ~SimpleGrid();

    /**
     * @brief 3 argument constructor.
     * @param extents The extents, or total number of cells this SimpleGrid will contain.
     * @param offset The offset for the first point in the patch in reference to the global grid.
     * @param numGhostCells The number of ghost cells this SimpleGrid has.
     */
    SimpleGrid(const SCIRun::IntVector& extents,
               const SCIRun::IntVector& offset,
               const int numGhostCells);
    /**
     * @brief Constructor to build grid from 3 linear arrays
     * @param XArray:  The populated array in the X direction
     * @param YArray:  The populated array in the Y direction
     * @param ZArray:  The populated array in the Z direction
     * @param offset:  The offset for the first point in the patch in reference to the global grid.
     * @param numGhostCells:  The number of ghost cells for this SimpleGrid.
     */
    SimpleGrid(const std::vector<T>& XArray,
               const std::vector<T>& YArray,
               const std::vector<T>& ZArray,
               const SCIRun::IntVector& offset,
               const int numGhostCells);
    /**
     * @brief
     * @param
     */
    inline void initialize(T val)
    {
      this->charges.initialize(val);
    }

    /**
     * @brief Copy constructor.
     * @param copy A reference to the SimpleGrid to copy.
     */
    SimpleGrid(const SimpleGrid& copy);

    /**
     *
     */
    inline T*** getCharges() const
    {
      return this->charges.get_dataptr();
    }

    /**
     * @brief Returns the extents of this SimpleGrid.
     * @param None
     * @return IntVector The extents, or total number of cells in this SimpleGrid.
     */
    inline IntVector getExtents() const
    {
      return this->gridExtents;
    }

    /**
     * @brief Returns the offset for the first point in this SimpleGrid in reference to the global grid.
     * @param None
     * @return The offset of this SimpleGrid.
     */
    inline IntVector getOffset() const
    {
      return this->gridOffset;
    }

    /**
     * @brief Returns prime (non-ghost only) extents of this SimpleGrid.
     * @param None
     * @return IntVector The prime (non-ghost only) extents of this SimpleGrid.
     */
    inline IntVector getNonGhostExtent() const
    {
      return this->gridExtents - this->numGhostCells;
    }

    /**The offset
     * @brief Returns prime (non-ghost only) offset for the first point in this
     *        SimpleGrid in reference to the global grid.
     * @param None
     * @return IntVector The prime (non-ghost only) offset of this SImpleGrid.
     */
    inline IntVector getNonGhostOffset() const
    {
      return this->gridOffset - this->numGhostCells;
    }

    /**
     * @brief Pass through indexing of Value array.
     * @param x The x component of the 3D cell index.
     * @param y The y component of the 3D cell index.
     * @param z The z component of the 3D cell index.
     * @return T& A reference to the value at index [x,y,z] in this SimpleGrid.
     */
    inline T& operator()(const int& x,
                         const int& y,
                         const int& z)
    {
      return this->charges(x, y, z);
    }

    /**
     * @brief Pass through indexing of Value array.
     * @param x The x component of the 3D cell index.
     * @param y The y component of the 3D cell index.
     * @param z The z component of the 3D cell index.
     * @return T The value at index [x,y,z] in this SimpleGrid.
     */
    inline T operator()(const int& x,
                        const int& y,
                        const int& z) const
    {
      return this->charges(x, y, z);
    }

    /**
     * @brief Index a cell value by IntVector.
     * @param idx The 3-component index vector.
     * @return T& A reference to the value at index idx.
     */
    inline T& operator()(const IntVector& idx)
    {
      return this->charges(idx.x(), idx.y(), idx.z());
    }

    /**
     * @brief Index a cell value by IntVector.
     * @param idx The 3-component index vector.
     * @return T The value at index idx.
     */
    inline T operator()(const IntVector& idx) const
    {
      return this->charges(idx.x(), idx.y(), idx.z());
    }

    /**
     * @brief Checks to make sure grid1 and grid2 have same Extent/Offset/Ghost Regions.
     *        Note that in general, gridIn doesn't have to have the same data type as (this) object does.
     * @param gridIn A reference to the SimpleGrid to compare against this SimpleGrid.
     * @return bool Returns true if this SimpleGrid has the same extents, offeset and number of ghost cells as
     *              as the specified SimpleGrid, flase otherwise.
     */
    bool verifyRegistration(SimpleGrid<T>& gridIn);

    /**
     * @brief Transforms 'Q' from real to fourier space
     * @param
     * @return
     */
    void inPlaceFFT_RealToFourier();

    /**
     * @brief Transforms 'Q' from fourier to real space
     * @param
     * @return
     */
    void inPlaceFFT_FourierToReal();

    //-------------------------------------------------------------------------------------
    // Beware high expense temporary creation; meta-template.
    // and/or re-couch in functions like .MultiplyInPlace(GridIn)?

    /**
     * @brief Multiplication of grids; be sure to check for same.
     * @param gridIn The multiplicand.
     * @return SimpleGrid<T> The result of the multiplication.
     */
    SimpleGrid<T> operator*(const SimpleGrid<T>& gridIn);

    /**
     * @brief Addition of grids; check extent/offset registration.
     * @param gridIn The addend.
     * @return SimpleGrid<T> The result of the addition.
     */
    SimpleGrid<T> operator+(const SimpleGrid<T>& gridIn);

    /**
     * @brief Subtraction of grids; check extent/offset registration.
     * @param gridIn The subtrahend.
     * @return SimpleGrid<T> The result of the subtraction.
     */
    SimpleGrid<T> operator-(const SimpleGrid<T>& gridIn);

    /**
     * @brief In place grid point by point accumulation.
     * @param gridIn The addend.
     * @return SimpleGrid<T>& The result of the addition on this SimpleGrid.
     */
    inline SimpleGrid<T>& operator+=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] += gridIn.charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place grid point by point subtraction.
     * @param gridIn The subtrahend.
     * @return SimpleGrid<T>& The result of the subtraction on this SimpleGrid (*this).
     */
    inline SimpleGrid<T>& operator-=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] -= gridIn.charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place grid point by point multiplication.
     * @param gridIn The multiplicand.
     * @return SimpleGrid<T>& The result of the multiplication on this SimpleGrid (*this).
     */
    inline SimpleGrid<T>& operator*=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] *= gridIn.charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place grid point by point division.
     * @param gridIn The divisor.
     * @return SimpleGrid<T>& The result of the division on this SimpleGrid (*this).
     */
    inline SimpleGrid<T>& operator/=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] /= gridIn.charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value addition.
     * @param gridIn The addend.
     * @return SimpleGrid<T>& The result of the addition on this SimpleGrid (*this).
     */
    inline SimpleGrid<T>& operator+=(const T& valueIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] += valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value subtraction.
     * @param gridIn The subtrahend.
     * @return SimpleGrid<T>& The result of the subtraction on this SimpleGrid (*this).
     */
    inline SimpleGrid<T>& operator-=(const T& valueIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] -= valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value multiplication.
     * @param valueIn The value multiplier.
     * @return SimpleGrid<T>& The result of the multiplication on this SimpleGrid with the specified value.
     */
    inline SimpleGrid<T>& operator*=(const T& valueIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] *= valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value division.
     * @param valueIn The value multiplier.
     * @return SimpleGrid<T>& The result of the division on this SimpleGrid by the specified value.
     */
    inline SimpleGrid<T>& operator/=(const T& valueIn)
    {
      for (unsigned int x = 0; x < gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < gridExtents.z(); ++z) {
            charges[x][y][z] /= valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief Unitary negation. For operators we should make sure the sub-operators are defined for the data types.
     *        What would a SimpleGrid<char> do with unitary negation?
     * @param None
     * @return SimpleGrid<T>& The result of the unitary negation of this SimpleGrid.
     */
    SimpleGrid<T>& operator-();

    /**
     * @brief Assignment operator. Gracefully handles self assignment.
     * @param sg The assignee.
     * @return SimpleGrid<T>& The result of the assignment (*this).
     */
    inline SimpleGrid<T>& operator=(const SimpleGrid<T>& copy)
    {
      charges.copy(copy.getCharges());          // SCIRun::Array3 assignment operator is private, use copy() method
      gridExtents = copy.gridExtents;
      gridOffset = copy.gridOffset;
      numGhostCells = copy.numGhostCells;
      return *this;
    }

//    friend std::ostream& operator<<(std::ostream& out,
//                                    const Uintah::SimpleGrid<T>& sg);
    /**
     * @brief A way to print this SimpleGrid. Avoids friending std::ostream.
     * @param out The std::ostream to output to.
     * @return std::ostream& A reference to the populated std::ostream.
     */
    std::ostream& print(std::ostream& out) const;

  private:

    SCIRun::Array3<T> charges;  //!< Grid cell values - can be double or std::complex<double>>
    IntVector gridExtents;      //!< Stores the number of total grid points in this grid
    IntVector gridOffset;       //!< Stores the offset pointer for the first point in this grid in reference to the global grid
    int numGhostCells;          //!< The number of ghost cells for the patch the associated points are on

    // NOTE:  We need to decide how to deal with ghost cells.
    // Extent/Offset of total grid doesn't tell us how much is "real" and how much is "ghost"

};

}  // End namespace Uintah

#endif
