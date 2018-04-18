/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_VISIT_INTERFACES_UTILS_H
#define UINTAH_VISIT_INTERFACES_UTILS_H

#include <Core/Grid/Variables/Variable.h>

#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Math/Matrix3.h>

using namespace Uintah;

namespace Uintah {

/////////////////////////////////////////////////////////////////////
// Utility functions for copying data from Uintah structures into
// simple arrays.
void copyIntVector(int to[3], const IntVector &from);
void copyVector(double to[3], const Vector &from);
void copyVector(double to[3], const Point &from);

/////////////////////////////////////////////////////////////////////
// Utility functions for serializing Uintah data structures into
// a simple array for visit.
template <typename T>
int numComponents()
{
  return 1;
}

template <typename T>
void copyComponents(double *dest, const T &src)
{
  (*dest) = 0;
}

template <>
int numComponents<Vector>();

template <>
int numComponents<IntVector>();

template <>
int numComponents<Stencil7>();

template <>
int numComponents<Stencil4>();

template <>
int numComponents<Point>();

template <>
int numComponents<Matrix3>();

template <>
void copyComponents(double *dest, const int &src);

template <>
void copyComponents(double *dest, const float &src);

template <>
void copyComponents(double *dest, const double &src);

template <>
void copyComponents<Vector>(double *dest, const Vector &src);

template <>
void copyComponents<IntVector>(double *dest, const IntVector &src);

template <>
void copyComponents<Stencil7>(double *dest, const Stencil7 &src);

template <>
void copyComponents<Stencil4>(double *dest, const Stencil4 &src);

template <>
void copyComponents<Point>(double *dest, const Point &src);

template <>
void copyComponents<Matrix3>(double *dest, const Matrix3 &src);

}

#endif
