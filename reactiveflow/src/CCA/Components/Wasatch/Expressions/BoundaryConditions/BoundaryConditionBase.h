/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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
#include <CCA/Components/Wasatch/BCHelper.h>

#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

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

  void bind_operators( const SpatialOps::OperatorDatabase& opdb )
  {
    switch (bcTypeEnum_) {
      case Wasatch::DIRICHLET:
      {
        switch (faceTypeEnum_) {
          case Uintah::Patch::xminus:
          case Uintah::Patch::xplus:
          {
            diriXOp_ = opdb.retrieve_operator<DiriXOpT>();
            break;
          }
          case Uintah::Patch::yminus:
          case Uintah::Patch::yplus:
          {
            diriYOp_ = opdb.retrieve_operator<DiriYOpT>();
            break;
          }
          case Uintah::Patch::zminus:
          case Uintah::Patch::zplus:
          {
            diriZOp_ = opdb.retrieve_operator<DiriZOpT>();
            break;
          }
          default:
          {
            std::ostringstream msg;
            msg << "ERROR: An invalid uintah face has been specified when tyring to apply boundary conditions.";
            break;
          }
        }
        break; // DIRICHLET
      }
        
      case Wasatch::NEUMANN:
      {
        switch (faceTypeEnum_) {
          case Uintah::Patch::xminus:
          case Uintah::Patch::xplus:
          {
            neumXOp_ = opdb.retrieve_operator<NeumXOpT>();
            break;
          }
          case Uintah::Patch::yminus:
          case Uintah::Patch::yplus:
          {
            neumYOp_ = opdb.retrieve_operator<NeumYOpT>();
            break;
          }
          case Uintah::Patch::zminus:
          case Uintah::Patch::zplus:
          {
            neumZOp_ = opdb.retrieve_operator<NeumZOpT>();
            break;
          }
          default:
          {
            std::ostringstream msg;
            msg << "ERROR: An invalid uintah face has been specified when tyring to apply boundary conditions.\n";
            break;
          }
        }
        break; // NEUMANN
      }
        
      default:
      {
        std::ostringstream msg;
        msg << "ERROR: It looks like you have specified an UNSUPPORTED boundary condition type!"
        << "Basic boundary types can only be either DIRICHLET or NEUMANN. Please revise your input file." << std::endl;
        break;
      }
    }
  }
  
  /**
   *  \brief Specify whether this boundary condition applies to a staggered field that is staggered in a
   direction normal to the boundary. For example, XVol and on x-boundaries, YVol on y-boundaries, and ZVol
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
  void set_interior_points( const std::vector<SpatialOps::IntVec>* vecInteriorPoints )
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
  void set_ghost_points( const std::vector<SpatialOps::IntVec>* vecGhostPoints ){
    vecGhostPts_ = vecGhostPoints;
  }

  /**
   *  \brief Set nebo-mask-friendly interior points. This is a vector of locally indexed ijk interior points. Interior
   points correspond to the interior cells adjacent to a boundary. For staggered fields that are normal
   to a boundary, the interior points correspond to the boundary faces instead of the cells.
   *  \param vecInteriorPoints Pointer to a stl vector of ijk triplets of interior cells adjacent
   to the boundary. These are indexed zero-based on the first interior cell of the patch.
   *
   */
  void set_nebo_interior_points( const std::vector<SpatialOps::IntVec>* vecInteriorPoints )
  {
    neboInteriorPts_  = vecInteriorPoints;
  }
  
  /**
   *  \brief Set nebo-mask-friendly extra cells. This is a vector of locally indexed ijk extra-cell points. Extra-cell
   points correspond to the extra cells adjacent to a boundary (outside). For staggered fields that
   are normal to a boundary, the extra points correspond to the outside faces instead of the cells.
   *  \param vecGhostPoints Pointer to a stl vector of ijk triplets of extra cells adjacent
   to the boundary. These are indexed zero-based on the first interior cell of the patch.
   *
   */
  void set_nebo_ghost_points( const std::vector<SpatialOps::IntVec>* vecGhostPoints ){
    neboGhostPts_ = vecGhostPoints;
  }

  /**
   *  \brief Set nebo-mask-friendly interior points. This is a vector of locally indexed ijk interior points. Interior
   points correspond to the interior cells adjacent to a boundary. For staggered fields that are normal
   to a boundary, the interior points correspond to the boundary faces instead of the cells.
   *  \param boundaryParticles
   *
   */
  void set_boundary_particles( const std::vector<int>* boundaryParticles )
  {
    boundaryParticles_ = boundaryParticles;
  }

  /**
   *  \brief Set interior edge cells. This is a vector of locally indexed ijk edge interior points.
   The edges correspond to the edges of the compuational domain.
   *  \param interiorEdgePoints Pointer to a stl vector of ijk triplets of interior edge cells.
   *
   */
  void set_interior_edge_points( const std::vector<SpatialOps::IntVec>* interiorEdgePoints ) {
    interiorEdgePoints_  = interiorEdgePoints;
  }

  /**
   *  \brief Set the patch cell offset. This is the global ijk of the lowest cell on this patch.
   *
   */
  void set_patch_cell_offset( const SpatialOps::IntVec& patchCellOffset )
  {
    patchCellOffset_ = patchCellOffset;
  }

  /**
   *  \brief Set the type of this bc: Dirichlet or Neumann.
   *
   */
  void set_bc_type ( Wasatch::BndCondTypeEnum bcTypeEnum)
  {
    bcTypeEnum_ = bcTypeEnum;
  }

  /**
   *  \brief Set the face type of this bc: xminus, xplus,...
   *
   */
  void set_face_type ( Uintah::Patch::FaceType faceType)
  {
    faceTypeEnum_ = faceType;
    switch (faceTypeEnum_) {
      case Uintah::Patch::xminus:
      case Uintah::Patch::yminus:
      case Uintah::Patch::zminus:
      {
        isMinusFace_ = true;
        break;
      }
      case Uintah::Patch::xplus:
      case Uintah::Patch::yplus:
      case Uintah::Patch::zplus:
      {
        isMinusFace_ = false;
        break;
      }
      default:
      {
        break;
      }
    }
  }

  /**
   *  \brief Set the boundary unit normal. This is (1,0,0) fro x+, (-1,0,0) for x-, etc...
   *
   */
  void set_boundary_normal( const SpatialOps::IntVec& bndNormal )
  {
    bndNormal_ = bndNormal;
  }

  /**
   *  \brief Specify whether this boundary condition applies in extra cells directly or uses operator inversion.
   *
   */
  void set_extra_only( const bool setExtraOnly )
  {
    setInExtraCellsOnly_ = setExtraOnly;
  }


  BoundaryConditionBase()
  {
    isStaggered_ = false;
    isMinusFace_ = false;
    setInExtraCellsOnly_ = false;
    ci_ = 0.0;
    cg_ = 0.0;
    patchCellOffset_ = SpatialOps::IntVec(0,0,0);
    bndNormal_       = SpatialOps::IntVec(0,0,0);
    bcTypeEnum_      = Wasatch::UNSUPPORTED;
    faceTypeEnum_    = Uintah::Patch::xminus;

    diriXOp_ = NULL;
    diriYOp_ = NULL;
    diriZOp_ = NULL;
    neumXOp_ = NULL;
    neumYOp_ = NULL;
    neumZOp_ = NULL;

    vecInteriorPts_     = NULL;
    vecGhostPts_        = NULL;
    neboInteriorPts_    = NULL;
    neboGhostPts_       = NULL;
    interiorEdgePoints_ = NULL;
    boundaryParticles_  = NULL;
  }
  virtual ~BoundaryConditionBase(){}

protected:
  bool isStaggered_, isMinusFace_, setInExtraCellsOnly_;
  double ci_, cg_;

  SpatialOps::IntVec patchCellOffset_;
  SpatialOps::IntVec bndNormal_;
  
  Wasatch::BndCondTypeEnum bcTypeEnum_; // DIRICHLET, NEUMANN, UNSUPPORTED
  Uintah::Patch::FaceType faceTypeEnum_;    // xminus, xplus...
  
  
  // operators
  typedef Wasatch::BCOpTypeSelector<FieldT> OpT;
  
  typedef SpatialOps::NeboBoundaryConditionBuilder<typename OpT::DirichletX> DiriXOpT;
  typedef SpatialOps::NeboBoundaryConditionBuilder<typename OpT::DirichletY> DiriYOpT;
  typedef SpatialOps::NeboBoundaryConditionBuilder<typename OpT::DirichletZ> DiriZOpT;
  
  typedef SpatialOps::NeboBoundaryConditionBuilder<typename OpT::NeumannX> NeumXOpT;
  typedef SpatialOps::NeboBoundaryConditionBuilder<typename OpT::NeumannY> NeumYOpT;
  typedef SpatialOps::NeboBoundaryConditionBuilder<typename OpT::NeumannZ> NeumZOpT;
  
  const DiriXOpT* diriXOp_;
  const DiriYOpT* diriYOp_;
  const DiriZOpT* diriZOp_;

  const NeumXOpT* neumXOp_;
  const NeumYOpT* neumYOp_;
  const NeumZOpT* neumZOp_;
  
  const std::vector<SpatialOps::IntVec>* vecInteriorPts_;
  const std::vector<SpatialOps::IntVec>* vecGhostPts_;
  
  const std::vector<SpatialOps::IntVec>* neboInteriorPts_;
  const std::vector<SpatialOps::IntVec>* neboGhostPts_;

  const std::vector<SpatialOps::IntVec>* interiorEdgePoints_;
  
  const std::vector<int>* boundaryParticles_; // vector of indices of particles on this boundary
  
  void build_mask_points(std::vector<SpatialOps::IntVec>& maskPoints)
  {
    if(isStaggered_ && bcTypeEnum_ != Wasatch::NEUMANN) {
      maskPoints = *neboInteriorPts_;
      maskPoints.insert(maskPoints.end(), neboGhostPts_->begin(), neboGhostPts_->end());
    }
    else if(isStaggered_ && bcTypeEnum_ == Wasatch::NEUMANN) {
      maskPoints = isMinusFace_ ? *neboInteriorPts_ : *neboInteriorPts_;
    }
    else {
      if (setInExtraCellsOnly_) {
        maskPoints = *neboGhostPts_;
      } else {
        switch (this->faceTypeEnum_) {
          case Uintah::Patch::xminus:
          case Uintah::Patch::yminus:
          case Uintah::Patch::zminus:
          {
            maskPoints = *neboInteriorPts_;
            break;
          }
          case Uintah::Patch::xplus:
          case Uintah::Patch::yplus:
          case Uintah::Patch::zplus:
          {
            maskPoints = *neboGhostPts_;
            break;
          }
          default:
          {
            break;
          }
        }
      }
    }
  }
};

#endif // BoundaryConditionBase_Expr_h
