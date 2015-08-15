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

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <spatialops/structured/SpatialMask.h>
#include <spatialops/NeboStencilBuilder.h>

/* 
 \brief The STAGGERED_MASK macro returns a mask that consists of the two points on each side of the svol extra cell mask.
 This mask is usually used for direct assignment on staggered fields on normal boundaries (XVol on x boundaries, YVol on y boundaries, and ZVol on z boundaries).
 The reason that we create this mask is to set values in the staggered extra cells for appropriate visualization
*/
#define STAGGERED_MASK \
  convert<FieldT>( *(this->svolSpatialMask_),SpatialOps::MINUS_SIDE, SpatialOps::PLUS_SIDE)

/* 
 \brief The APPLY_COMPLEX_BC macro allows the application of complex boundary conditions. The general formulation is as follows:

 APPLY_COMPLEX_BC(f, my_complex_bc)
 
 where f is the computed field on which the BC is being applied and my_complex_bc is ANY nebo expression of the same type as the field that is
 computed by the expression. So, if you want to apply a BC on a field of type FieldT, then my_complex_bc is of type FieldT.
 
 Examples:
 APPLY_COMPLEX_BC(f, a*x + b);
 APPLY_COMPLEX_BC(f, a_ * (x - x0_)*(x - x0_) + b_ * (x - x0_) + c_);
 APPLY_COMPLEX_BC(f, phic_ * pow( 1.0 - abs(x - x0_) / R_ , 1.0/n_ ));
 
 If your independent variable is NOT of the same type as the computed field, then simply use an interpolant
 to move things to the correct location.
 
 e.g.
 
 APPLY_COMPLEX_BC(f, a * interOp(y) + b); where interpOp interpolates from yType to fType
 */
#define APPLY_COMPLEX_BC(f, BCVALUE)                                                               \
{                                                                                               \
  if( this->isStaggeredNormal_ ){                                                                     \
    masked_assign(STAGGERED_MASK, f, BCVALUE);                                                  \
  } else {                                                                                      \
    if (this->setInExtraCellsOnly_)                                                             \
    {                                                                                           \
      masked_assign( *(this->spatialMask_), f, BCVALUE);                                        \
    } else {                                                                                    \
      typedef Wasatch::BCOpTypeSelector<FieldT> OpT;                                            \
      switch (this->bcTypeEnum_) {                                                              \
        case Wasatch::DIRICHLET:                                                                \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletX::DestFieldType DesT;                             \
              (*this->diriXOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpXOp_)(BCVALUE), this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletY::DestFieldType DesT;                             \
              (*this->diriYOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpYOp_)(BCVALUE), this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletZ::DestFieldType DesT;                             \
              (*this->diriZOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpZOp_)(BCVALUE), this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        case Wasatch::NEUMANN:                                                                  \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannX::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumXOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, (*this->interpNeuXOp_)(BCVALUE), this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumXOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpNeuXOp_)(BCVALUE), this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannY::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumYOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, (*this->interpNeuYOp_)(BCVALUE), this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumYOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpNeuYOp_)(BCVALUE), this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannZ::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumZOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, (*this->interpNeuZOp_)(BCVALUE), this->isMinusFace_);  \
              } else {                                                                          \
                (*this->neumZOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, (*this->interpNeuZOp_)(BCVALUE), this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        default:                                                                                \
        {                                                                                       \
          std::ostringstream msg;                                                               \
          msg << "ERROR: It looks like you have specified an UNSUPPORTED boundary condition type!"  \
          << "Basic boundary types can only be either DIRICHLET or NEUMANN. Please revise your input file." << std::endl; \
          break;                                                                                \
          }                                                                                     \
        }                                                                                       \
      }                                                                                         \
  }                                                                                             \
}

/* 
 \brief The APPLY_CONSTANT_BC macro applies a constant boundary condition.
 Here are the rules:
 
 Staggered field on a boundary perpendicular to the staggered direction (we call this staggered normal):
 ======================================================================================================
 This occurs when setting an inlet x-velocity for example at an x boundary. In general, XVol and X-boundaries,
 YVol and y-boundaries, ZVol and z-boundaries fall into this category. For this case, the domain's boundary
 coincides with the first interior cell for the staggered field. Also, for visualization purposes, we choose
 to set the same bc value in the staggered extra cell. That's why you see the use of STAGGERED_MASK
 in the masked_assign below.
 
 setInExtraCellsOnly - or Direct Assignment to the extra cell:
 =============================================================
 This will be visited by Scalar BCs only. As the name designates, this will set the bcvalue in the 
 extra cell (or spatialMask_) only
 
 Dirichlet Boundary Conditions:
 ==============================
 Applies standard operator inversion
 
 Neumann Boundary Condition:
 ===========================
 Applies standard operator inversion. When the field is staggered-normal, use the svolmask instead
 of the staggered mask.
 */
#define APPLY_CONSTANT_BC(f, BCVALUE)                                                              \
{                                                                                               \
  if( this->isStaggeredNormal_ && this->bcTypeEnum_ != Wasatch::NEUMANN ){                            \
    masked_assign ( STAGGERED_MASK, f, bcValue_ );                                              \
  } else {                                                                                      \
    if (this->setInExtraCellsOnly_)                                                             \
    {                                                                                           \
      masked_assign( *(this->spatialMask_), f, BCVALUE);                                        \
    } else {                                                                                    \
      typedef Wasatch::BCOpTypeSelector<FieldT> OpT;                                            \
      switch (this->bcTypeEnum_) {                                                              \
        case Wasatch::DIRICHLET:                                                                \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletX::DestFieldType DesT;                             \
              (*this->diriXOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletY::DestFieldType DesT;                             \
              (*this->diriYOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::DirichletZ::DestFieldType DesT;                             \
              (*this->diriZOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        case Wasatch::NEUMANN:                                                                  \
        {                                                                                       \
          switch (this->faceTypeEnum_) {                                                        \
            case Uintah::Patch::xminus:                                                         \
            case Uintah::Patch::xplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannX::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumXOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumXOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::yminus:                                                         \
            case Uintah::Patch::yplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannY::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumYOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              } else {                                                                                                          \
                (*this->neumYOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            case Uintah::Patch::zminus:                                                         \
            case Uintah::Patch::zplus:                                                          \
            {                                                                                   \
              typedef typename OpT::NeumannZ::DestFieldType DesT;                               \
              if (this->isStaggeredNormal_)                                                           \
              {                                                                                 \
                (*this->neumZOp_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);  \
              } else {                                                                          \
                (*this->neumZOp_)(convert<DesT>( *(this->spatialMask_),this->shiftSide_), f, BCVALUE, this->isMinusFace_);      \
              }                                                                                 \
              break;                                                                            \
            }                                                                                   \
            default:                                                                            \
            {                                                                                   \
              break;                                                                            \
            }                                                                                   \
          }                                                                                     \
          break;                                                                                \
        }                                                                                       \
        default:                                                                                \
        {                                                                                       \
          std::ostringstream msg;                                                               \
          msg << "ERROR: It looks like you have specified an UNSUPPORTED boundary condition type!"  \
          << "Basic boundary types can only be either DIRICHLET or NEUMANN. Please revise your input file." << std::endl; \
          break;                                                                                \
          }                                                                                     \
        }                                                                                       \
      }                                                                                         \
  }                                                                                           \
}
// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
ConstantBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  if (this->spatialMask_) {
    FieldT& lhs = this->value();
    APPLY_CONSTANT_BC(lhs, bcValue_);
  }
}

// ###################################################################
// a necessary specialization for particle fields because the BCHelper automatically creates
// ConstantBC for Dirichlet boundary conditions specified in the input file.
template<>
void
ConstantBC<ParticleField>::
evaluate()
{}

// ###################################################################

template< typename FieldT >
void
OneSidedDirichletBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;

  FieldT& f = this->value();

  if (this->spatialMask_) {
    FieldT& lhs =  this->value();
    masked_assign(*this->spatialMask_, lhs, bcValue_);
    masked_assign(convert<FieldT>(*this->spatialMask_, this->shiftSide_), lhs, bcValue_);
  }
}

// ###################################################################

template< typename FieldT >
void
LinearBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  if( this->spatialMask_ ) {
    FieldT& lhs =  this->value();
    const FieldT& x = x_->field_ref();
    APPLY_COMPLEX_BC(lhs, a_ * x + b_);
  }
}

// ###################################################################

template< typename FieldT >
void
ParabolicBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  if( this->spatialMask_ ) {
    FieldT& lhs =  this->value();
    const FieldT& x = x_->field_ref();
    APPLY_COMPLEX_BC(lhs, a_ * (x - x0_)*(x - x0_) + b_ * (x - x0_) + c_);
  }
}

// ###################################################################

template< typename FieldT >
void
PowerLawBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  if( this->spatialMask_ ) {
    FieldT& lhs =  this->value();
    const FieldT& x = x_->field_ref();
    APPLY_COMPLEX_BC(lhs, phic_ * pow( 1.0 - abs(x - x0_) / R_ , 1.0/n_ ));
  }
}

// ###################################################################

template< typename FieldT >
void
BCCopier<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& lhs =  this->value();
  const FieldT& src = src_->field_ref();
  if( this->spatialMask_ ){
    if( this->isStaggeredNormal_ ){
      masked_assign(STAGGERED_MASK, lhs, src);
    } else {
      masked_assign(*this->spatialMask_, lhs, src);
    }
  }  
}

// ###################################################################
template< typename FieldT >
void
BCPrimVar<FieldT>::
bind_operators(const SpatialOps::OperatorDatabase& opdb)
{
  BoundaryConditionBase<FieldT>::bind_operators(opdb);
  if (hasDensity_) {
    rhoInterpOp_ = opdb.retrieve_operator<DenInterpT>();
  }
  neux_ = opdb.retrieve_operator<Neum2XOpT>();
  neuy_ = opdb.retrieve_operator<Neum2YOpT>();
  neuz_ = opdb.retrieve_operator<Neum2ZOpT>();
}

//---------------------------------------------------------------------
template< typename FieldT >
void
BCPrimVar<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& lhs =  this->value();  
  if ( this->spatialMask_ ) {
    const FieldT& src = src_->field_ref();
    if (this->isStaggeredNormal_) {
      if (!hasDensity_) {
        masked_assign(STAGGERED_MASK, lhs, src);
      } else {
        const SVolField& rho = rho_->field_ref();
        masked_assign( convert<FieldT>( *(this->svolSpatialMask_),this->shiftSide_), lhs, (*rhoInterpOp_)(rho) * src);
        /*
         FOR the extra-cell on the staggered-normal field, use a gradient operator (XVol to XSurfX)
         and set the value to zero. Essentially copying the value from the first interior staggered cell.
         */
        switch (this->faceTypeEnum_) {
          case Uintah::Patch::xminus:
          case Uintah::Patch::xplus:
          {
            typedef typename GradX::DestFieldType DesT;
            (*this->neux_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), lhs, 0.0, this->isMinusFace_);
            break;
          }                                                                                   
          case Uintah::Patch::yminus:                                                         
          case Uintah::Patch::yplus:                                                          
          {                                                                                   
            typedef typename GradY::DestFieldType DesT;
            (*this->neuy_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), lhs, 0.0, this->isMinusFace_);
            break;
          }                                                                                   
          case Uintah::Patch::zminus:                                                         
          case Uintah::Patch::zplus:                                                          
          {                                                                                   
            typedef typename GradZ::DestFieldType DesT;
            (*this->neuz_)(convert<DesT>( *(this->svolSpatialMask_),this->shiftSide_), lhs, 0.0, this->isMinusFace_);
            break;
          }                                                                                   
          default:                                                                            
          {                                                                                   
            break;                                                                            
          }                                                                                   
        }
        // tsaad: STAGGERED_MASK doesnt' work here since it results in a runtime segfault. The Interpolant
        // for the density is invalid in the extra-cells for the staggered-normal field.
        // so we use two masked assign, for each side of the svol cell (see previous masked_assign)
        // masked_assign(STAGGERED_MASK, lhs, (*rhoInterpOp_)(rho) * src);
      }
    } else {
      if (!hasDensity_) {
        masked_assign(*this->spatialMask_, lhs, src);
      } else {
        const SVolField& rho = rho_->field_ref();
        masked_assign(*this->spatialMask_, lhs, (*rhoInterpOp_)(rho) * src);
      }
    }
  }
}

// ###################################################################
// EXPLICIT INSTANTIATION
#include <CCA/Components/Wasatch/FieldTypes.h>
#define INSTANTIATE_BC_PROFILES(VOLT)       \
template class ConstantBC         <VOLT>;   \
template class OneSidedDirichletBC<VOLT>;   \
template class LinearBC           <VOLT>;   \
template class ParabolicBC        <VOLT>;   \
template class PowerLawBC         <VOLT>;   \
template class BCCopier<VOLT>;

INSTANTIATE_BC_PROFILES(SVolField)
INSTANTIATE_BC_PROFILES(XVolField)
INSTANTIATE_BC_PROFILES(YVolField)
INSTANTIATE_BC_PROFILES(ZVolField)

template class BCPrimVar<XVolField>;
template class BCPrimVar<YVolField>;
template class BCPrimVar<ZVolField>;