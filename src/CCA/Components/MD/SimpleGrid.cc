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

#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Math/Matrix3.h>
#include <Core/Geometry/Vector.h>

#include <sci_values.h>

#include <complex>

namespace Uintah {

  template<typename T>
  SimpleGrid<T>::SimpleGrid()
  {

  }

  template<typename T>
  SimpleGrid<T>::~SimpleGrid()
  {

  }

  template<typename T>
  SimpleGrid<T>::SimpleGrid(const IntVector& extents,
                            const IntVector& offset,
                            const IntVector& origin,
                            const int numGhostCells) :
      d_internalExtents(extents), d_gridOffset(offset), d_internalOrigin(origin), d_numGhostCells(numGhostCells)
  {
    d_values.resize(extents.x()+numGhostCells, extents.y()+numGhostCells, extents.z()+numGhostCells);
  }

  template<typename T>
  SimpleGrid<T>::SimpleGrid(const SimpleGrid& copy)
  {
    d_values = copy.d_values;
    d_internalExtents = copy.d_internalExtents;
    d_gridOffset = copy.d_gridOffset;
    d_numGhostCells = copy.d_numGhostCells;
  }

  template<typename T>
  bool SimpleGrid<T>::verifyRegistration(const SimpleGrid<T>& gridIn)
  {
    if ((d_internalExtents != gridIn.d_internalExtents) || (d_gridOffset != gridIn.d_gridOffset)
        || (d_numGhostCells != gridIn.d_numGhostCells) || (d_internalOrigin != gridIn.d_internalOrigin)) {
      std::ostringstream ostr;
      ostr << "Uintah::MD SimpleGrids differ in extent, offset or number of ghost cells.";
      throw SCIRun::InternalError(ostr.str(), __FILE__, __LINE__);
    } else {
      return true;
    }
  }

  template<typename T>
  SimpleGrid<T> SimpleGrid<T>::operator+(const SimpleGrid<T>& gridIn)
  {
    if (this->verifyRegistration(gridIn)) {
      SimpleGrid<T> newGrid(*this);
      newGrid += gridIn;
      return newGrid;
    } else {
      std::ostringstream ostr;
      ostr << "SimpleGrid operator+ error:  Grids are not registered with one another.";
      throw SCIRun::InternalError(ostr.str(), __FILE__, __LINE__);
    }
  }

  template<typename T>
  SimpleGrid<T> SimpleGrid<T>::operator-(const SimpleGrid<T>& gridIn)
  {
    if (this->verifyRegistration(gridIn)) {
      SimpleGrid<T> newGrid(*this);
      newGrid -= gridIn;
      return newGrid;
    } else {
      std::ostringstream ostr;
      ostr << "SimpleGrid operator- error:  Grids are not registered with one another.";
      throw SCIRun::InternalError(ostr.str(), __FILE__, __LINE__);
    }
  }

  template<typename T>
  SimpleGrid<T> SimpleGrid<T>::operator*(const SimpleGrid<T>& gridIn)
  {
    if (this->verifyRegistration(gridIn)) {
      SimpleGrid<T> newGrid(*this);
      newGrid *= gridIn;
      return newGrid;
    } else {
      std::ostringstream ostr;
      ostr << "SimpleGrid operator* error:  Grids are not registered with one another.";
      throw SCIRun::InternalError(ostr.str(), __FILE__, __LINE__);
    }
  }

  template<typename T>
  std::ostream& SimpleGrid<T>::print(std::ostream& out) const
  {
    out << "Extent, [x,y,z]: " << d_internalExtents;
    out << "Offset, [x,y,z]: " << d_gridOffset;
    out << "Origin, [x,y,z]: " << d_internalOrigin;
    out << "GhostCells, [x,y,z]: " << d_internalExtents;
    return out;
  }

  template<typename T>
  std::ostream& operator<<(std::ostream& out,
                           const Uintah::SimpleGrid<T>& sg)
  {
    return sg.print(out);
  }

// Explicit template instantiations:
  template class SimpleGrid<dblcomplex> ;
  template class SimpleGrid<Uintah::Matrix3> ;
  template class SimpleGrid<double> ;
  template class SimpleGrid<SCIRun::Vector> ;

}  // end namespace Uintah
