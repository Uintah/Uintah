/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef SpatialOps_structured_StencilBuilder_h
#define SpatialOps_structured_StencilBuilder_h

/**
 *  \file StencilBuilder.h
 */

namespace SpatialOps{

  class OperatorDatabase; // forward declaration
  class Grid;             // forward declaration

  /**
   *  \ingroup optypes
   *
   *  \fn void build_stencils( const unsigned int,
   *                           const unsigned int,
   *                           const unsigned int,
   *                           const double,
   *                           const double,
   *                           const double,
   *                           OperatorDatabase& );
   *
   *  \brief builds commonly used stencil operators
   *
   *  \param nx number of points in the x-direction
   *  \param ny number of points in the y-direction
   *  \param nz number of points in the z-direction
   *  \param Lx length in x-direction
   *  \param Ly length in y-direction
   *  \param Lz length in z-direction
   *  \param opdb the OperatorDatabase to register the operators on
   */
  void build_stencils( const unsigned int nx,
                       const unsigned int ny,
                       const unsigned int nz,
                       const double Lx,
                       const double Ly,
                       const double Lz,
                       OperatorDatabase& opdb );

  /**
   * \ingroup optypes
   *
   * \fn void build_stencils( const Grid&, OperatorDatabase& )
   *
   * \brief builds commonly used stencil operators
   *
   * \param grid the grid to build the stencils on
   * \param opDB the OperatorDatabase to store the stencils in
   */
  void build_stencils( const Grid& grid, OperatorDatabase& opDB );

} // namespace SpatialOps

#endif // SpatialOps_structured_StencilBuilder_h
