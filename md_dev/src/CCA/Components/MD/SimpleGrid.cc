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

#include <complex>

#include <sci_values.h>

using namespace SCIRun;

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
                          const int numGhostCells) :
    d_gridExtents(extents), d_gridOffset(offset), d_numGhostCells_(numGhostCells)
{
  d_charges.copy(Array3<T>(extents.x(), extents.y(), extents.z()));
}

template<typename T>
SimpleGrid<T>::SimpleGrid(const SimpleGrid& copy)
{
  int dm1 = d_gridExtents.x();
  int dm2 = d_gridExtents.y();
  int dm3 = d_gridExtents.z();
  for (int i = 0; i < dm1; i++) {
    for (int j = 0; j < dm2; j++) {
      for (int k = 0; k < dm3; k++) {
        d_charges(i, j, k) = copy(i, j, k);
      }
    }
  }
  d_gridExtents = copy.d_gridExtents;
  d_gridOffset = copy.d_gridOffset;
  d_numGhostCells_ = copy.d_numGhostCells_;
}

template<typename T>
bool SimpleGrid<T>::verifyRegistration(SimpleGrid<T>& gridIn)
{
  if ((d_gridExtents != gridIn.d_gridExtents) || (d_gridOffset != gridIn.d_gridOffset)
      || (d_numGhostCells_ != gridIn.d_numGhostCells_)) {
    ostringstream ostr;
    ostr << "MD SimpleGrids differ in extent, offset or number of ghost cells.";
    throw SCIRun::InternalError(ostr.str(), __FILE__, __LINE__);
  } else {
    return true;
  }
}

template<typename T>
void SimpleGrid<T>::inPlaceFFT_RealToFourier()
{

}

template<typename T>
void SimpleGrid<T>::inPlaceFFT_FourierToReal()
{

}

template<typename T>
SimpleGrid<T> SimpleGrid<T>::operator*(const SimpleGrid<T>& gridIn)
{
  int xdim = d_gridExtents.x();
  int ydim = d_gridExtents.y();
  int zdim = d_gridExtents.z();
  for (int x = 0; x < xdim; ++x) {
    for (int y = 0; y < ydim; ++y) {
      for (int z = 0; z < zdim; ++z) {
        // FIXME d_charges(x,y,z) *= gridIn.d_charges(x,y,z);
      }
    }
  }
  return *this;
}

template<typename T>
SimpleGrid<T> SimpleGrid<T>::operator+(const SimpleGrid<T>& gridIn)
{
  int xdim = d_gridExtents.x();
  int ydim = d_gridExtents.y();
  int zdim = d_gridExtents.z();
  for (int x = 0; x < xdim; ++x) {
    for (int y = 0; y < ydim; ++y) {
      for (int z = 0; z < zdim; ++z) {
        d_charges(x, y, z) -= gridIn.d_charges(x, y, z);
      }
    }
  }
  return *this;
}

template<typename T>
SimpleGrid<T> SimpleGrid<T>::operator-(const SimpleGrid<T>& gridIn)
{
  int xdim = d_gridExtents.x();
  int ydim = d_gridExtents.y();
  int zdim = d_gridExtents.z();
  for (int x = 0; x < xdim; ++x) {
    for (int y = 0; y < ydim; ++y) {
      for (int z = 0; z < zdim; ++z) {
        d_charges(x, y, z) -= gridIn.d_charges(x, y, z);
      }
    }
  }
  return *this;
}

template<typename T>
std::ostream& SimpleGrid<T>::print(std::ostream& out) const
{
  out << "Extent, [x,y,z]: " << d_gridExtents;
  out << "Offset, [x,y,z]: " << d_gridOffset;
  out << "GhostCells, [x,y,z]: " << d_gridExtents;
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
