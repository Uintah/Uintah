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

#include <CCA/Components/Wasatch/Expressions/TargetValueSource.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT >
TargetValueSource<FieldT>::
TargetValueSource( const Expr::Tag& phiTag,
                  const Expr::Tag& phiRHSTag,
                  const Expr::Tag& volFracTag,
                  const Expr::Tag& targetPhiTag,
                  const double targetPhiValue)
  : Expr::Expression<FieldT>(),
    constValTarget_(targetPhiTag == Expr::Tag()),
    targetphivalue_(targetPhiValue)
{
  this->set_gpu_runnable( true );
  
  const WasatchCore::TagNames& tagNames = WasatchCore::TagNames::self();
  dt_ = this->template create_field_request<TimeField>(tagNames.dt);
  if (targetPhiTag != Expr::Tag()) targetphi_ = this->template create_field_request<FieldT>(targetPhiTag);
  phi_     = this->template create_field_request<FieldT>(phiTag);
  phiRHS_  = this->template create_field_request<FieldT>(phiRHSTag);
  volFrac_ = this->template create_field_request<FieldT>(volFracTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
TargetValueSource<FieldT>::
~TargetValueSource()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
TargetValueSource<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TargetValueSource<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  if (constValTarget_)
    result <<= (1.0/dt_->field_ref()) * ( targetphivalue_ - phi_->field_ref()) + phiRHS_->field_ref() ;
  else
    result <<= (1.0/dt_->field_ref()) * ( targetphi_->field_ref() - phi_->field_ref()) + phiRHS_->field_ref() ;
  
  result <<= volFrac_->field_ref()*result;

}

//--------------------------------------------------------------------

template< typename FieldT >
TargetValueSource<FieldT>::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& phiTag,
                 const Expr::Tag& phiRHSTag,
                 const Expr::Tag& volFracTag,
                 const Expr::Tag& targetPhiTag,
                 const double targetPhiValue )
  : ExpressionBuilder(result         ),
    phit_           ( phiTag         ),
    phirhst_        ( phiRHSTag      ),
    volfract_       ( volFracTag     ),
    targetphit_     ( targetPhiTag   ),
    targetphivalue_ ( targetPhiValue )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TargetValueSource<FieldT>::Builder::build() const
{
  return new TargetValueSource<FieldT>( phit_, phirhst_, volfract_, targetphit_, targetphivalue_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class TargetValueSource< SpatialOps::SVolField >;
//==================================================================
