/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#ifndef BoundaryConditionBase_Expr_h
#define BoundaryConditionBase_Expr_h

#include <expression/Expression.h>
/**
 *  \class 	BoundaryConditionBase
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Provides an expression to set basic Dirichlet and Neumann boundary
 *  conditions. Given a BCValue, we set the ghost value such that
 *  \f$ f[g] = \alpha f[i] + \beta F \f$ where \f$ g \f$ is the extra cell, \f$ i \f$ is the interior cell,
 and \f$F\f$ is the desired boundary condition for \f$ f \f$. Note that \f$ \alpha \f$ and \f$ \beta \f$
 are interpolation coefficients used to deduce the extra cell value that will reproduce the boundary condition 
 up to second order accuracy.
 *
 *  \tparam FieldT - The type of field for the expression on which this bc applies.
 */

template< typename FieldT >
class BoundaryConditionBase
: public Expr::Expression<FieldT>
{
public:

  /**
   *  \brief Specify whether this boundary condition applies to a staggered field that is staggered in a
   direction normal to the boundary. For exampl, XVol and on x-boundaries, YVol on y-boundaries, and ZVol
   on z-boundaries.
   *  \param staggered Boolean that specifies whether this field in staggered and normal (see documentation above).
   *
   */
  void set_staggered( const bool staggered )
  {
    isStaggered_ = staggered;
  }

  /**
   *  \brief Set value of \f$\alpha\f$ (the interior coefficient) that is used in the bc interpolation (see documentation above).
   *  \param coef Value of the interior coefficient.
   *
   */
  void set_interior_coef( const double coef )
  {
    ci_ = coef;
  }

  /**
   *  \brief Set value of \f$\beta\f$ (the extra cell coefficient) that is used in the bc interpolation (see documentation above).
   *  \param coef Value of the exterior coefficient.
   *
   */
  void set_ghost_coef( const double coef )
  {
    cg_ = coef;
  }
  
  /**
   *  \brief Set interior points. This is a vector of locally indexed ijk interior points. Interior
   points correspond to the interior cells adjacent to a boundary. For staggered fields that are normal
   to a boundary, the interior points correspond to the boundary faces instead of the cells.
   *  \param vecInteriorPoints Pointer to a stl vector of ijk triplets of interior cells adjacent
   to the boundary.
   *
   */
  void set_interior_points( const std::vector<SpatialOps::structured::IntVec>* vecInteriorPoints )
  {
    vecInteriorPts_  = vecInteriorPoints;
  }

  /**
   *  \brief Set extra cells. This is a vector of locally indexed ijk extra-cell points. Extra-cell
   points correspond to the extra cells adjacent to a boundary (outside). For staggered fields that 
   are normal to a boundary, the extra points correspond to the outside faces instead of the cells.
   *  \param vecGhostPoints Pointer to a stl vector of ijk triplets of extra cells adjacent
   to the boundary.
   *
   */
  void set_ghost_points( const std::vector<SpatialOps::structured::IntVec>* vecGhostPoints ){
    vecGhostPts_ = vecGhostPoints;
  }

  /**
   *  \brief Set interior edge cells. This is a vector of locally indexed ijk edge interior points.
   The edges correspond to the edges of the compuational domain.
   *  \param interiorEdgePoints Pointer to a stl vector of ijk triplets of interior edge cells.
   *
   */
  void set_interior_edge_points( const std::vector<SpatialOps::structured::IntVec>* interiorEdgePoints ) {
    interiorEdgePoints_  = interiorEdgePoints;
  }

  /**
   *  \brief Set the patch cell offset. This is the global ijk of the lowest cell on this patch.
   *
   */
  void set_patch_cell_offset( const SpatialOps::structured::IntVec& patchCellOffset ){
    patchCellOffset_ = patchCellOffset;
  }

  /**
   *  \brief Set the boundary unit normal. This is (1,0,0) fro x+, (-1,0,0) for x-, etc...
   *
   */
  void set_boundary_normal( const SpatialOps::structured::IntVec& bndNormal )
  {
    bndNormal_ = bndNormal;
  }

  BoundaryConditionBase()
  {
    isStaggered_ = false;
    ci_ = 0.0;
    cg_ = 0.0;
    patchCellOffset_ = SpatialOps::structured::IntVec(0,0,0);
    bndNormal_       = SpatialOps::structured::IntVec(0,0,0);
    vecInteriorPts_ = NULL;
    vecGhostPts_ = NULL;
    interiorEdgePoints_ = NULL;
  }
  virtual ~BoundaryConditionBase(){}

protected:
  bool isStaggered_;
  double ci_, cg_;
  SpatialOps::structured::IntVec patchCellOffset_;
  SpatialOps::structured::IntVec bndNormal_;
  const std::vector<SpatialOps::structured::IntVec>* vecInteriorPts_;
  const std::vector<SpatialOps::structured::IntVec>* vecGhostPts_;
  const std::vector<SpatialOps::structured::IntVec>* interiorEdgePoints_;
};

#endif // BoundaryConditionBase_Expr_h
