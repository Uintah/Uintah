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

#include <CCA/Components/Wasatch/Expressions/FanModel.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT >
FanModel<FieldT>::
FanModel( const Expr::Tag& rhoTag,
          const Expr::Tag& momTag,
          const Expr::Tag& momRHSTag,
          const Expr::Tag& fanSrcOldTag,
          const Expr::Tag& volFracTag,
          const double targetVelocity)
  : Expr::Expression<FieldT>(), targetVel_(targetVelocity)
{
  this->set_gpu_runnable( true );
  rho_     = this->template create_field_request<SVolField>(rhoTag);
  
  const WasatchCore::TagNames& tagNames = WasatchCore::TagNames::self();
  dt_ = this->template create_field_request<TimeField>(tagNames.dt);
  
  mom_     = this->template create_field_request<FieldT>(momTag);
  momRHS_  = this->template create_field_request<FieldT>(momRHSTag);
  volFrac_ = this->template create_field_request<FieldT>(volFracTag);
  fanSourceOld_ = this->template create_field_request<FieldT>(fanSrcOldTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
FanModel<FieldT>::
~FanModel()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
FanModel<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  densityInterpOp_ = opDB.retrieve_operator<DensityInterpT>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
FanModel<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  // Do not delete this next line plz. THis is another form of the Fan Model - and for some reason it just works!
  // result <<= volFrac_->field_ref()*(1.0/dt_->field_ref()) * ( (*densityInterpOp_)(rho_->field_ref()) * targetVel_ - mom_->field_ref());
  result <<= (1.0/dt_->field_ref()) * ( (*densityInterpOp_)(rho_->field_ref()) * targetVel_ - mom_->field_ref()) - ( momRHS_->field_ref() - fanSourceOld_->field_ref() );
  result <<= volFrac_->field_ref()*result;

}

//--------------------------------------------------------------------

template< typename FieldT >
FanModel<FieldT>::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& rhoTag,
                 const Expr::Tag& momTag,
                 const Expr::Tag& momRHSTag,
                 const Expr::Tag& volFracTag,
                 const double targetVelocity )
  : ExpressionBuilder(result         ),
    rhot_           ( rhoTag         ),
    momt_           ( momTag         ),
    momrhst_        ( momRHSTag      ),
    volfract_       ( volFracTag     ),
    fansrcoldt_     ( Expr::Tag(result.name() + "_old", Expr::STATE_NONE) ),
    targetvelocity_ ( targetVelocity )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
FanModel<FieldT>::Builder::build() const
{
  return new FanModel<FieldT>( rhot_, momt_, momrhst_, fansrcoldt_, volfract_, targetvelocity_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class FanModel< SpatialOps::SVolField >;
template class FanModel< SpatialOps::XVolField >;
template class FanModel< SpatialOps::YVolField >;
template class FanModel< SpatialOps::ZVolField >;
//==================================================================
