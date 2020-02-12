/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

/*
 *  utils.cc: Utility functions for the interface between Uintah and VisIt.
 *
 *  Written by:
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2018
 *
 */

#include <VisIt/interfaces/utils.h>

using namespace Uintah;

namespace Uintah {

/////////////////////////////////////////////////////////////////////
// Utility functions for copying data from Uintah structures into
// simple arrays.

  void copyIntVector(int to[3], const IntVector &from)
{
  to[0]=from[0];  to[1]=from[1];  to[2]=from[2];
}

void copyVector(double to[3], const Vector &from)
{
  to[0]=from[0];  to[1]=from[1];  to[2]=from[2];
}

void copyVector(double to[3], const Point &from)
{
  to[0]=from.x();  to[1]=from.y();  to[2]=from.z();
}


template <>
int numComponents<Vector>()
{
  return 3;
}

template <>
int numComponents<IntVector>()
{
  return 3;
}

template <>
int numComponents<Stencil7>()
{
  return 7;
}

template <>
int numComponents<Stencil4>()
{
  return 4;
}

template <>
int numComponents<Point>()
{
  return 3;
}

template <>
int numComponents<Matrix3>()
{
  return 9;
}

template <>
void copyComponents(double *dest, const int &src)
{
  (*dest) = (double) src;
}

template <>
void copyComponents(double *dest, const float &src)
{
  (*dest) = (double) src;
}

template <>
void copyComponents(double *dest, const double &src)
{
  (*dest) = (double) src;
}

template <>
void copyComponents<Vector>(double *dest, const Vector &src)
{
  dest[0] = (double)src[0];
  dest[1] = (double)src[1];
  dest[2] = (double)src[2];
}

template <>
void copyComponents<IntVector>(double *dest, const IntVector &src)
{
  dest[0] = (double)src[0];
  dest[1] = (double)src[1];
  dest[2] = (double)src[2];
}

template <>
void copyComponents<Stencil7>(double *dest, const Stencil7 &src)
{
  dest[0] = (double)src[0];
  dest[1] = (double)src[1];
  dest[2] = (double)src[2];
  dest[3] = (double)src[3];
  dest[4] = (double)src[4];
  dest[5] = (double)src[5];
  dest[6] = (double)src[6];
}

template <>
void copyComponents<Stencil4>(double *dest, const Stencil4 &src)
{
  dest[0] = (double)src[0];
  dest[1] = (double)src[1];
  dest[2] = (double)src[2];
  dest[3] = (double)src[3];
}

template <>
void copyComponents<Point>(double *dest, const Point &src)
{
  dest[0] = (double)src.x();
  dest[1] = (double)src.y();
  dest[2] = (double)src.z();
}

template <>
void copyComponents<Matrix3>(double *dest, const Matrix3 &src)
{
  for (int j=0; j<3; ++j) {
    for (int i=0; i<3; ++i) {
      dest[j*3+i] = (double)src(j,i);
    }
  }
}

}
