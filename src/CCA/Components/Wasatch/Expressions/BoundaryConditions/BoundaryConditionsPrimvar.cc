/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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
 *  NOTE: we have split the implementations into several files to reduce the size
 *        of object files when compiling with nvcc, since we were crashing the
 *        linker in some cases.
 */

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <spatialops/structured/SpatialMask.h>

namespace WasatchCore{

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

#define INSTANTIATE_BC_PROFILES(VOLT)   \
    template class BCPrimVar  <VOLT>;

  INSTANTIATE_BC_PROFILES(SVolField)
  INSTANTIATE_BC_PROFILES(XVolField)
  INSTANTIATE_BC_PROFILES(YVolField)
  INSTANTIATE_BC_PROFILES(ZVolField)

} // namespace WasatchCore
