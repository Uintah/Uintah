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

#define STAGGERED_MASK \
  convert<FieldT>( *(this->svolSpatialMask_),SpatialOps::MINUS_SIDE, SpatialOps::PLUS_SIDE)

#define APPLY_COMPLEX_BC(BCVALUE)                                                               \
{                                                                                               \
  if( this->isStaggered_ ){                                                                     \
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
              if (this->isStaggered_)                                                           \
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
              if (this->isStaggered_)                                                           \
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
              if (this->isStaggered_)                                                           \
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

#define APPLY_CONSTANT_BC(BCVALUE)                                                              \
{                                                                                               \
  if( this->isStaggered_ && this->bcTypeEnum_ != Wasatch::NEUMANN ){                            \
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
              if (this->isStaggered_)                                                           \
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
              if (this->isStaggered_)                                                           \
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
              if (this->isStaggered_)                                                           \
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
    FieldT& f = this->value();
    APPLY_CONSTANT_BC(bcValue_);
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
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    std::vector<SpatialOps::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost local ijk index
    std::vector<SpatialOps::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior local ijk index
    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      f(*ii) = bcValue_ ;
      f(*ig) = bcValue_ ;
    }
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
    FieldT& f = this->value();
    const FieldT& x = x_->field_ref();
    APPLY_COMPLEX_BC(a_ * x + b_);
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
    FieldT& f = this->value();
    const FieldT& x = x_->field_ref();
    APPLY_COMPLEX_BC(a_ * (x - x0_)*(x - x0_) + b_ * (x - x0_) + c_);
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
    FieldT& f = this->value();
    const FieldT& x = x_->field_ref();
    APPLY_COMPLEX_BC(phic_ * pow( 1.0 - abs(x - x0_) / R_ , 1.0/n_ ));
  }
}

// ###################################################################

template< typename FieldT >
void
BCCopier<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const FieldT& src = src_->field_ref();
  if( this->spatialMask_ ){
    if( this->isStaggered_ ){
      masked_assign(STAGGERED_MASK, f, src);
    } else {
      masked_assign(*this->spatialMask_, f, src);
    }
  }  
}

// ###################################################################

template< typename FieldT >
void
BCPrimVar<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    const FieldT& src = src_->field_ref();

    std::vector<SpatialOps::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      if (!hasDensity_) {
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = src(*ig);
          f(*ii) = src(*ii);
        }
      } else {
        const SVolField& rho = rho_->field_ref();
        SpatialOps::IntVec offset = this->isMinusFace_ ? IntVec(0,0,0) : this->bndNormal_;
        for( ; ii != (this->vecInteriorPts_)->end(); ++ig, ++ii ){
          const double avrho = ( rho(*ig - offset) + rho(*ii - offset) )/2.0;
          f(*ig) = avrho * src(*ig);
          f(*ii) = avrho * src(*ii);
        }
      }
    } else {
      if (hasDensity_) {
        const SVolField& rho = rho_->field_ref();
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = rho(*ig) * src(*ig);
        }
      } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        //const double avrho = ((*rho_)(*ig) + (*rho_)(*ig + offset))/2.0;
        f(*ig) = src(*ig);
      }
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
template class BCCopier<VOLT>;\
template class BCPrimVar<VOLT>;
INSTANTIATE_BC_PROFILES(SVolField)
INSTANTIATE_BC_PROFILES(XVolField)
INSTANTIATE_BC_PROFILES(YVolField)
INSTANTIATE_BC_PROFILES(ZVolField)
