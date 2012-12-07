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

#ifndef UINTAH_MD_SIMPLE_GRID_H
#define UINTAH_MD_SIMPLE_GRID_H

#include <Core/Geometry/IntVector.h>
#include <Core/Containers/Array3.h>
#include <Core/Grid/Patch.h>
#include <Core/Util/Assert.h>

namespace Uintah {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::IntVector;

class Patch;

template<class T> class SimpleGrid {

  public:

    /**
     * @brief Default Constructor.
     */
    SimpleGrid();

    /**
     * @brief 3 argument constructor.
     * @param
     * @param
     * @param
     */
    SimpleGrid(const IntVector& extents,
               const IntVector& offset,
               const int numGhostCells);

    /**
     * @brief Destructor
     */
    virtual ~SimpleGrid();

    /**
     * @brief Returns the extents of this SimpleGrid.
     */
    inline IntVector getExtents() const
    {
      return this->d_gridExtents;
    }

    /**
     * @brief Returns the offset of this SimpleGrid.
     */
    inline IntVector getOffset() const
    {
      return this->d_gridOffset;
    }

    /**
     * @brief Returns prime (non-ghost only) extent of this SimpleGrid.
     */
    inline IntVector getNonGhostExtent() const
    {
      return this->d_gridExtents - this->d_numGhostCells;
    }

    /**
     * @brief Returns prime (non-ghost only) extent.
     */
    inline IntVector getNonGhostOffset() const
    {
      return this->d_gridOffset - this->d_numGhostCells;
    }

    /**
     * @brief Pass through indexing of Value array.
     */
    inline T& operator()(const int& i,
                         const int& j,
                         const int& k)
    {
      return this->d_charges(i, j, k);
    }

    /**
     * @brief Pass through indexing of Value array.
     * @param
     * @param
     * @param
     */
    inline T operator()(const int& i,
                        const int& j,
                        const int& k) const
    {
      return this->d_charges(i, j, k);
    }

    /**
     * @brief Index by IntVector.
     * @param
     */
    inline T& operator()(const IntVector& idx)
    {
      return this->d_charges(idx.x(), idx.y(), idx.z());
    }

    /**
     * @brief Index by IntVector.
     * @param
     */
    inline T operator()(const IntVector& idx) const
    {
      return this->d_charges(idx.x(), idx.y(), idx.z());
    }

    /**
     * @brief Checks to make sure grid1 and grid2 have same Extent/Offset/Ghost Regions.
     *        Note that in general, gridIn doesn't have to have the same data type as (this) object does.
     * @param
     */
    bool verifyRegistration(SimpleGrid<T>& gridIn);

    // Beware high expense temporary creation; meta-template.
    // and/or re-couch in functions like .MultiplyInPlace(GridIn)?

    /**
     * @brief Multiplication of grids; be sure to check for same.
     * @param
     */
    SimpleGrid<T> operator*(const SimpleGrid<T>& gridIn);

    /**
     * @brief Addition of grids; check extent/offset registration.
     * @param
     */
    SimpleGrid<T> operator+(const SimpleGrid<T>& gridIn);

    /**
     * @brief Subtraction of grids; check extent/offset registration.
     * @param
     */
    SimpleGrid<T> operator-(const SimpleGrid<T>& gridIn);

    /**
     * @brief In place grid point by point accumulation.
     * @param
     */
    inline SimpleGrid<T>& operator+=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] += gridIn.d_charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place grid point by point subtraction.
     * @param
     */
    inline SimpleGrid<T>& operator-=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] -= gridIn.d_charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place grid point by point multiplication.
     * @param
     */
    inline SimpleGrid<T>& operator*=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] *= gridIn.d_charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place grid point by point division.
     * @param
     */
    inline SimpleGrid<T>& operator/=(const SimpleGrid<T>& gridIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] /= gridIn.d_charges[x][y][z];
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value addition.
     * @param
     */
    inline SimpleGrid<T>& operator+=(const T& valueIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] += valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value subtraction.
     * @param
     */
    inline SimpleGrid<T>& operator-=(const T& valueIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] -= valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value multiplication.
     * @param
     */
    inline SimpleGrid<T>& operator*=(const T& valueIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] *= valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief In place value division.
     * @param
     */
    inline SimpleGrid<T>& operator/=(const T& valueIn)
    {
      for (unsigned int x = 0; x < d_gridExtents.x(); ++x) {
        for (unsigned int y = 0; y < d_gridExtents.y(); ++y) {
          for (unsigned int z = 0; z < d_gridExtents.z(); ++z) {
            d_charges[x][y][z] /= valueIn;
          }
        }
      }
      return *this;
    }

    /**
     * @brief Unitary negation. For operators we should make sure the sub-operators are defined for the data types.
     *        What would a SimpleGrid<char> do with unitary negation?
     */
    SimpleGrid<T>& operator-();

    /**
     * @brief Assignment operator
     * @param
     */
    inline SimpleGrid<T>& operator=(const SimpleGrid<T>& sg)
    {
      d_charges = sg.d_charges;
      d_gridExtents = sg.d_gridExtents;
      d_gridOffset = sg.d_gridOffset;
      d_numGhostCells = sg.d_numGhostCells;

      return *this;
    }


//    friend std::ostream& operator<<(std::ostream& out,
//                                    const Uintah::SimpleGrid<T>& sg);
    /**
     * @brief
     * @param
     */
    std::ostream& print(std::ostream& out) const;

  private:
    SCIRun::Array3<T> d_charges;  // Grid cell values - can be double or std::complex<double>>
    IntVector d_gridExtents;      // Stores the number of total grid points in this grid;
    IntVector d_gridOffset;       // Stores the offset pointer for the first point in this grid in reference to the global grid
    int d_numGhostCells;          // The number of ghost cells from the patch we

    // NOTE:  We need to decide how to deal with ghost cells.
    // Extent/Offset of total grid doesn't tell us how much is "real" and how much is "ghost"

};

}  // End namespace Uintah

#endif
