/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#include <CCA/Components/Wasatch/WasatchBCHelper.h>

namespace WasatchCore{

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

    inline void bind_operators( const SpatialOps::OperatorDatabase& opdb )
    {
      // operators
      typedef WasatchCore::BCOpTypeSelector<FieldT> OpT;

      switch (faceTypeEnum_) {
        case Uintah::Patch::xminus:
        case Uintah::Patch::xplus:
        {
          interpXOp_ = opdb.retrieve_operator<typename OpT::DirichletX>();
          interpNeuXOp_ = opdb.retrieve_operator<typename OpT::InterpX>();
          break;
        }
        case Uintah::Patch::yminus:
        case Uintah::Patch::yplus:
        {
          interpYOp_ = opdb.retrieve_operator<typename OpT::DirichletY>();
          interpNeuYOp_ = opdb.retrieve_operator<typename OpT::InterpY>();
          break;
        }
        case Uintah::Patch::zminus:
        case Uintah::Patch::zplus:
        {
          interpZOp_ = opdb.retrieve_operator<typename OpT::DirichletZ>();
          interpNeuZOp_ = opdb.retrieve_operator<typename OpT::InterpZ>();
          break;
        }
        default:
        {
          std::ostringstream msg;
          msg << "ERROR: It looks like you have specified an UNSUPPORTED boundary condition type!"
              << "Basic boundary types can only be either DIRICHLET or NEUMANN. Please revise your input file." << std::endl;
          break;
        }
      }

      switch (bcTypeEnum_) {
        case WasatchCore::DIRICHLET:
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

        case WasatchCore::NEUMANN:
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
     *  \brief Specify whether this boundary condition applies to a staggered
     *         field that is staggered in a direction normal to the boundary.
     *         For example, XVol and on x-boundaries, YVol on y-boundaries,
     *         and ZVol on z-boundaries.
     *  \param staggeredNormal Boolean that specifies whether this field in
     *         staggered and normal (see documentation above).
     *
     */
    inline void set_staggered_normal( const bool staggeredNormal )
    {
      isStaggeredNormal_ = staggeredNormal;
    }

    /**
     *  \brief Set value of \f$\alpha\f$ (the interior coefficient) that is used in the bc interpolation (see documentation above).
     *  \param coef Value of the interior coefficient.
     *
     */
    inline void set_interior_coef( const double coef )
    {
      ci_ = coef;
    }

    /**
     *  \brief Set value of \f$\beta\f$ (the extra cell coefficient) that is used in the bc interpolation (see documentation above).
     *  \param coef Value of the exterior coefficient.
     *
     */
    inline void set_ghost_coef( const double coef )
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
    inline void set_interior_points( const std::vector<SpatialOps::IntVec>* vecInteriorPoints )
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
    inline void set_ghost_points( const std::vector<SpatialOps::IntVec>* vecGhostPoints ){
      vecGhostPts_ = vecGhostPoints;
    }

    /**
     *  \brief Set extra cell spatial mask.
     *
     */
    inline void set_spatial_mask( const SpatialOps::SpatialMask<FieldT>* spatMask )
    {
      spatialMask_ = spatMask;
    }

    inline void set_svol_spatial_mask( const SpatialOps::SpatialMask<SVolField>* spatMask )
    {
      svolSpatialMask_ = spatMask;
    }

    inline void set_interior_svol_spatial_mask( const SpatialOps::SpatialMask<SVolField>* spatMask )
    {
      interiorSvolSpatialMask_ = spatMask;
    }

    /**
     *  \brief Set nebo-mask-friendly interior points. This is a vector of locally indexed ijk interior points. Interior
     points correspond to the interior cells adjacent to a boundary. For staggered fields that are normal
     to a boundary, the interior points correspond to the boundary faces instead of the cells.
     *  \param boundaryParticles
     *
     */
    inline void set_boundary_particles( const std::vector<int>* boundaryParticles )
    {
      boundaryParticles_ = boundaryParticles;
    }

    /**
     *  \brief Set interior edge cells. This is a vector of locally indexed ijk edge interior points.
     The edges correspond to the edges of the compuational domain.
     *  \param interiorEdgePoints Pointer to a stl vector of ijk triplets of interior edge cells.
     *
     */
    inline void set_interior_edge_points( const std::vector<SpatialOps::IntVec>* interiorEdgePoints ) {
      interiorEdgePoints_  = interiorEdgePoints;
    }

    /**
     *  \brief Set the patch cell offset. This is the global ijk of the lowest cell on this patch.
     *
     */
    inline void set_patch_cell_offset( const SpatialOps::IntVec& patchCellOffset )
    {
      patchCellOffset_ = patchCellOffset;
    }

    /**
     *  \brief Set the type of this bc: Dirichlet or Neumann.
     *
     */
    inline void set_bc_type ( WasatchCore::BndCondTypeEnum bcTypeEnum)
    {
      bcTypeEnum_ = bcTypeEnum;
    }

    /**
     *  \brief Set the type of this boundary: INLET, VELOCITY, etc...
     *
     */
    inline void set_bnd_type ( WasatchCore::BndTypeEnum bndTypeEnum)
    {
      bndTypeEnum_ = bndTypeEnum;
    }

    /**
     *  \brief Set the face type of this bc: xminus, xplus,...
     *
     */
    inline void set_face_type ( Uintah::Patch::FaceType faceType)
    {
      faceTypeEnum_ = faceType;
      switch (faceTypeEnum_) {
        case Uintah::Patch::xminus:
        case Uintah::Patch::yminus:
        case Uintah::Patch::zminus:
        {
          isMinusFace_ = true;
          bcSide_ = SpatialOps::MINUS_SIDE;
          shiftSide_ = SpatialOps::PLUS_SIDE;
          break;
        }
        case Uintah::Patch::xplus:
        case Uintah::Patch::yplus:
        case Uintah::Patch::zplus:
        {
          isMinusFace_ = false;
          bcSide_ = SpatialOps::PLUS_SIDE;
          shiftSide_ = SpatialOps::MINUS_SIDE;
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
    inline void set_boundary_normal( const SpatialOps::IntVec& bndNormal )
    {
      bndNormal_ = bndNormal;
    }

    /**
     *  \brief Specify whether this boundary condition applies in extra cells directly or uses operator inversion.
     *
     */
    inline void set_extra_only( const bool setExtraOnly )
    {
      setInExtraCellsOnly_ = setExtraOnly;
    }


    BoundaryConditionBase()
    {
      this->set_gpu_runnable(false);
      isStaggeredNormal_ = false;
      isMinusFace_ = false;
      setInExtraCellsOnly_ = false;
      ci_ = 0.0;
      cg_ = 0.0;
      patchCellOffset_ = SpatialOps::IntVec(0,0,0);
      bndNormal_       = SpatialOps::IntVec(0,0,0);
      bcTypeEnum_      = WasatchCore::UNSUPPORTED;
      faceTypeEnum_    = Uintah::Patch::xminus;

      diriXOp_ = nullptr;
      diriYOp_ = nullptr;
      diriZOp_ = nullptr;
      neumXOp_ = nullptr;
      neumYOp_ = nullptr;
      neumZOp_ = nullptr;

      vecInteriorPts_     = nullptr;
      vecGhostPts_        = nullptr;
      interiorEdgePoints_ = nullptr;
      boundaryParticles_  = nullptr;
    }
    virtual ~BoundaryConditionBase(){}

  protected:
    bool isStaggeredNormal_, isMinusFace_, setInExtraCellsOnly_;
    double ci_, cg_;

    SpatialOps::IntVec patchCellOffset_;
    SpatialOps::IntVec bndNormal_;

    WasatchCore::BndTypeEnum bndTypeEnum_;
    WasatchCore::BndCondTypeEnum bcTypeEnum_; // DIRICHLET, NEUMANN, UNSUPPORTED
    Uintah::Patch::FaceType faceTypeEnum_;    // xminus, xplus...
    SpatialOps::BCSide bcSide_, shiftSide_;

    // operators
    typedef WasatchCore::BCOpTypeSelector<FieldT> OpT;

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

    // interpolants that will be used for independent variables when used for complex boundary conditions (i.e. a*x + b -> a*interp(x) + b)
    const typename OpT::DirichletX* interpXOp_;
    const typename OpT::DirichletY* interpYOp_;
    const typename OpT::DirichletZ* interpZOp_;

    const typename OpT::InterpX* interpNeuXOp_;
    const typename OpT::InterpY* interpNeuYOp_;
    const typename OpT::InterpZ* interpNeuZOp_;

    const std::vector<SpatialOps::IntVec>* vecInteriorPts_;
    const std::vector<SpatialOps::IntVec>* vecGhostPts_;

    const std::vector<SpatialOps::IntVec>* interiorEdgePoints_;

    const std::vector<int>* boundaryParticles_; // vector of indices of particles on this boundary

    const SpatialOps::SpatialMask<SVolField>* svolSpatialMask_;
    const SpatialOps::SpatialMask<SVolField>* interiorSvolSpatialMask_;
    const SpatialOps::SpatialMask<FieldT>*    spatialMask_;
  };

} // namespace WasatchCore

#endif // BoundaryConditionBase_Expr_h
