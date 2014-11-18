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

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <spatialops/structured/SpatialMask.h>

#define APPLY_BC(BCVALUE)                                                                   \
{                                                                                           \
if (this->setInExtraCellsOnly_)                                                             \
{                                                                                           \
  SpatialMask<FieldT> mask(f, maskPoints);                                              \
  f <<= cond( mask, BCVALUE )                                                              \
            ( f              );                                                             \
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
          SpatialMask<DesT> mask = SpatialMask<DesT>::build(f, maskPoints);         \
          (*this->diriXOp_)(mask, f, BCVALUE, this->isMinusFace_);                          \
          break;                                                                            \
        }                                                                                   \
        case Uintah::Patch::yminus:                                                         \
        case Uintah::Patch::yplus:                                                          \
        {                                                                                   \
          typedef typename OpT::DirichletY::DestFieldType DesT;                             \
          SpatialMask<DesT> mask = SpatialMask<DesT>::build(f, maskPoints);         \
          (*this->diriYOp_)(mask, f, BCVALUE, this->isMinusFace_);                          \
          break;                                                                            \
        }                                                                                   \
        case Uintah::Patch::zminus:                                                         \
        case Uintah::Patch::zplus:                                                          \
        {                                                                                   \
          typedef typename OpT::DirichletZ::DestFieldType DesT;                             \
          SpatialMask<DesT> mask = SpatialMask<DesT>::build(f, maskPoints);         \
          (*this->diriZOp_)(mask, f, BCVALUE, this->isMinusFace_);                          \
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
          SpatialMask<DesT> mask = SpatialMask<DesT>::build(f, maskPoints);         \
          (*this->neumXOp_)(mask, f, BCVALUE, this->isMinusFace_);                          \
          break;                                                                            \
        }                                                                                   \
        case Uintah::Patch::yminus:                                                         \
        case Uintah::Patch::yplus:                                                          \
        {                                                                                   \
          typedef typename OpT::NeumannY::DestFieldType DesT;                               \
          SpatialMask<DesT> mask = SpatialMask<DesT>::build(f, maskPoints);         \
          (*this->neumYOp_)(mask, f, BCVALUE, this->isMinusFace_);                          \
          break;                                                                            \
        }                                                                                   \
        case Uintah::Patch::zminus:                                                         \
        case Uintah::Patch::zplus:                                                          \
        {                                                                                   \
          typedef typename OpT::NeumannZ::DestFieldType DesT;                               \
          SpatialMask<DesT> mask = SpatialMask<DesT>::build(f, maskPoints);         \
          (*this->neumZOp_)(mask, f, BCVALUE, this->isMinusFace_);                          \
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
  FieldT& f = this->value();
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    
    std::vector<IntVec> maskPoints;
    this->build_mask_points(maskPoints);
    
    if( this->isStaggered_ && this->bcTypeEnum_ != Wasatch::NEUMANN ){
      SpatialMask<FieldT> mask(f, maskPoints);
      masked_assign( mask, f, bcValue_ );
    }
    else{
      APPLY_BC(bcValue_);
    }
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
  
  FieldT& f = this->value();
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    if( this->isStaggered_ ){
      std::vector<IntVec> maskPoints;
      this->build_mask_points(maskPoints);
      SpatialMask<FieldT> mask(f, maskPoints);
      f <<= cond( mask, a_ * *x_ + b_ )
                ( f                   );
    }
    else{
      const double ci = this->ci_;
      const double cg = this->cg_;
      std::vector<SpatialOps::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost local ijk index
      std::vector<IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior local ijk index
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        f(*ig) = ( ( a_ * (*x_)(*ig) + b_ ) - ci*f(*ii) ) / cg;
      }
    }
  }
}

// ###################################################################

template< typename FieldT >
void
ParabolicBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    if( this->isStaggered_ ){
      std::vector<IntVec> maskPoints;
      this->build_mask_points(maskPoints);
      SpatialMask<FieldT> mask(f, maskPoints);
      f <<= cond( mask, a_ * (*x_ - x0_)*(*x_ - x0_) + b_ * (*x_ - x0_) + c_ )
                ( f );
    }
    else{
      const double ci = this->ci_;
      const double cg = this->cg_;
      std::vector<IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost local ijk index
      std::vector<IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior local ijk index
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double x = (*x_)(*ig) - x0_;
        f(*ig) = ( (a_ * x*x + b_ * x + c_) - ci*f(*ii) ) / cg;
      }
    }
  }
}

// ###################################################################

template< typename FieldT >
void
PowerLawBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    if(this->isStaggered_) {
      std::vector<IntVec> maskPoints;
      this->build_mask_points(maskPoints);
      SpatialMask<FieldT> mask(f, maskPoints);
      f <<= cond( mask, phic_ * pow( 1.0 - abs(*x_ - x0_) / R_ , 1.0/n_ ) )
                ( f );
    }
    else{
      const double ci = this->ci_;
      const double cg = this->cg_;
      std::vector<IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost local ijk index
      std::vector<IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior local ijk index
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double bcVal = phic_ * std::pow( 1.0 - std::fabs( (*x_)(*ig) - x0_ ) / R_ , 1.0/n_ );
        f(*ig) = ( bcVal - ci*f(*ii) ) / cg;
      }
    }
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
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    std::vector<IntVec> maskPoints;
    this->build_mask_points(maskPoints);
    if( this->isStaggered_ ){
      SpatialMask<FieldT> mask(f, maskPoints);
      f <<= cond( mask, *src_)
                ( f          );
    }
    else{
      SpatialMask<FieldT> mask(f, * this->neboGhostPts_);
      f <<= cond( mask, *src_)
                ( f          );
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
template class BCCopier           <VOLT>;

INSTANTIATE_BC_PROFILES(SVolField)
INSTANTIATE_BC_PROFILES(XVolField)
INSTANTIATE_BC_PROFILES(YVolField)
INSTANTIATE_BC_PROFILES(ZVolField)
