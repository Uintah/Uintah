/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_MD_SIMPLEGRID_H
#define UINTAH_MD_SIMPLEGRID_H

#include <Core/Containers/LinearArray3.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Patch.h>
#include <Core/Util/Assert.h>

#include <complex>

namespace Uintah {

  typedef std::complex<double> dblcomplex;

  using namespace SCIRun;

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
      ~SimpleGrid();

      /**
       * @brief 3 argument constructor.
       * @param extents The extents, or total number of cells this SimpleGrid will contain.
       * @param offset The offset for the first point in the patch in reference to the global grid.
       * @param numGhostCells The number of ghost cells this SimpleGrid has.
       */
      SimpleGrid(const IntVector&   extents,
                 const IntVector&   offset,
                 const IntVector&   origin,
                 const int          numGhostCells);

      /**
       * @brief
       * @param
       */
      inline void initialize(const T val)
      {
        d_values.initialize(val);
      }

      /**
       * @brief Copy constructor.
       * @param copy A reference to the SimpleGrid to copy.
       */
      SimpleGrid(const SimpleGrid& copy);

      /**
       * @brief Returns the extents of this SimpleGrid.
       * @param None
       * @return IntVector The extents, or total number of cells in this SimpleGrid.
       */
      inline IntVector getExtents() const
      {
        return d_internalExtents;
      }

      /**
       * @brief Returns the offset for the first point in this SimpleGrid in reference to the global grid.
       * @param None
       * @return The offset of this SimpleGrid.
       */
      inline IntVector getOffset() const
      {
        return d_gridOffset;
      }

      inline void setOffset(const SCIRun::IntVector& _IV_In) {
        d_gridOffset = _IV_In;
        return;
      }

      inline IntVector getOrigin() const
      {
        return d_internalOrigin;
      }

      inline void setOrigin(const SCIRun::IntVector& _IV_In) {
        d_internalOrigin = _IV_In;
        return;
      }

      /**
       * @brief Returns prime (non-ghost only) extents of this SimpleGrid.
       * @param None
       * @return IntVector The prime (non-ghost only) extents of this SimpleGrid.
       */
      inline IntVector getExtentWithGhost() const
      {
        return d_internalExtents + IntVector(d_numGhostCells, d_numGhostCells, d_numGhostCells);
      }

      /**The offset
       * @brief Returns prime (non-ghost only) offset for the first point in this
       *        SimpleGrid in reference to the global grid.
       * @param None
       * @return IntVector The prime (non-ghost only) offset of this SimpleGrid.
       */
      inline IntVector getOffsetWithGhost() const
      {
        return d_gridOffset - d_internalOrigin;
      }

      /**
       * @brief Returns a pointer to the linearized, 1D array of values contained within this SimpleGrid.
       * @param None
       * @return T* A pointer to the linearized, 1D array of values contained within this SimpleGrid.
       */
      inline T* getDataPtr()
      {
        return d_values.get_dataptr();
      }

      inline LinearArray3<T>* getDataArray()
      {
        return &d_values;
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
        return d_values(x, y, z);
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
        return d_values(x, y, z);
      }

      /**
       * @brief Index a cell value by IntVector.
       * @param idx The 3-component index vector.
       * @return T& A reference to the value at index idx.
       */
      inline T& operator()(const IntVector& idx)
      {
        return d_values(idx.x(), idx.y(), idx.z());
      }

      /**
       * @brief Index a cell value by IntVector.
       * @param idx The 3-component index vector.
       * @return T The value at index idx.
       */
      inline T operator()(const IntVector& idx) const
      {
        return d_values(idx.x(), idx.y(), idx.z());
      }

      /**
       * @brief Checks to make sure grid1 and grid2 have same Extent/Offset/Ghost Regions.
       *        Note that in general, gridIn doesn't have to have the same data type as (this) object does.
       * @param other A reference to the SimpleGrid to compare against this SimpleGrid.
       * @return bool Returns true if this SimpleGrid has the same extents, offset and number of ghost cells as
       *              as the specified SimpleGrid, false otherwise.
       */
      bool verifyRegistration(const SimpleGrid<T>& other);

      /**
       * @brief Fill the grid with the given value.
       * @param fillValue A reference to the value to fill the grid with.
       * @return None
       */
      inline void fill(const T& fillValue)
      {
    	 for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
           for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
             for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
            	 d_values(x, y, z) = fillValue;
             }
           }
    	 }
      }

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
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
              d_values(x, y, z) += gridIn.d_values(x, y, z);
            }
          }
        }
        return *this;
      }

      /**
       * @brief In place grid point by point subtraction.
       * @param gridIn The subtrahend.
       * @return SimpleGrid<T>& The result of the subtraction on this SimpleGrid.
       */
      inline SimpleGrid<T>& operator-=(const SimpleGrid<T>& gridIn)
      {
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
              d_values(x, y, z) -= gridIn.d_values(x, y, z);
            }
          }
        }
        return *this;
      }

      /**
       * @brief In place grid point by point multiplication.
       * @param gridIn The multiplicand.
       * @return SimpleGrid<T>& The result of the multiplication on this SimpleGrid.
       */
      inline SimpleGrid<T>& operator*=(const SimpleGrid<T>& gridIn)
      {
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
              d_values(x, y, z) *= gridIn.d_values(x, y, z);
            }
          }
        }
        return *this;
      }

      /**
       * @brief In place grid point by point division.
       * @param gridIn The divisor.
       * @return SimpleGrid<T>& The result of the division on this SimpleGrid.
       */
      inline SimpleGrid<T>& operator/=(const SimpleGrid<T>& gridIn)
      {
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
            // FIXME / is not well defined for vector/vector
            //d_values(x, y, z) /= gridIn.d_values(x, y, z);
            }
          }
        }
        return *this;
      }

      /**
       * @brief In place value addition.
       * @param valueIn The addend.
       * @return SimpleGrid<T>& The result of the addition on this SimpleGrid.
       */
      inline SimpleGrid<T>& operator+=(const T& valueIn)
      {
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
              d_values(x, y, z) += valueIn;
            }
          }
        }
        return *this;
      }

      /**
       * @brief In place value subtraction.
       * @param gridIn The subtrahend.
       * @return SimpleGrid<T>& The result of the subtraction on this SimpleGrid.
       */
      inline SimpleGrid<T>& operator-=(const T& valueIn)
      {
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
              d_values(x, y, z) -= valueIn;
            }
          }
        }
        return *this;
      }

      /**
       * @brief In place value multiplication.
       * @param value The value multiplier.
       * @return SimpleGrid<T>& The result of the multiplication on this SimpleGrid with the specified value.
       */
      inline SimpleGrid<T>& operator*=(const double value)
      {
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
              d_values(x, y, z) *= value;
            }
          }
        }
        return *this;
      }

      /**
       * @brief In place value division.
       * @param value The value multiplier.
       * @return SimpleGrid<T>& The result of the division on this SimpleGrid by the specified value.
       */
      inline SimpleGrid<T>& operator/=(const double value)
      {
        for (int x = 0; x < d_internalExtents.x() + d_numGhostCells; ++x) {
          for (int y = 0; y < d_internalExtents.y() + d_numGhostCells; ++y) {
            for (int z = 0; z < d_internalExtents.z() + d_numGhostCells; ++z) {
              d_values(x, y, z) /= value;
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
       * @param copy The assignee.
       * @return SimpleGrid<T>& The result of the assignment.
       */
      inline SimpleGrid<T>& operator=(const SimpleGrid<T>& copy)
      {
        d_values = copy.d_values;
        d_internalExtents = copy.d_internalExtents;
        d_internalOrigin = copy.d_internalOrigin;
        d_gridOffset = copy.d_gridOffset;
        d_numGhostCells = copy.d_numGhostCells;
        return *this;
      }

      /**
       * @brief A way to print this SimpleGrid. Avoids friending std::ostream.
       * @param out The std::ostream to output to.
       * @return std::ostream& A reference to the populated std::ostream.
       */
      std::ostream& print(std::ostream& out) const;

    private:

      LinearArray3<T> d_values;     //!< Grid cell values
      IntVector d_internalExtents;  //!< Stores the number of total grid points
                                    //   in this grid
      IntVector d_gridOffset;       //!< Stores the offset of the current grid
                                    //   in relation to the global origin
      IntVector d_internalOrigin;   //!< IntVector which stores the internal
                                    //   x/y/z/ components of the first non-ghost
                                    //   point within the simple grid
      int d_numGhostCells;          //!< The total number of ghost cells in
                                    //   each direction in the grid

  };

}  // End namespace Uintah

#endif
