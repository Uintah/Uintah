/*
 * The MIT License
 *
 * Copyright (c) 2014-2026 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/TimeAdvanceMomentum.h>
#include <CCA/Components/Wasatch/TagNames.h>

#include <spatialops/Nebo.h>


template< typename FieldT >
TimeAdvanceMomentum<FieldT>::
TimeAdvanceMomentum( const std::string& solnVarName,
                       const Expr::Tag& momHatTag,
                       const Expr::Tag& rhsTag,
                       const WasatchCore::TimeIntegrator timeIntInfo )
: Expr::Expression<FieldT>(),
  dtt_        ( WasatchCore::TagNames::self().dt      ),
  timeIntInfo_( timeIntInfo                           )
{
  this->set_gpu_runnable( true );
  
   momHat_ = this->template create_field_request<FieldT>(momHatTag);
   gradP_ = this->template create_field_request<FieldT>(rhsTag);
   dt_ = this->template create_field_request<SingleValue>(WasatchCore::TagNames::self().dt);
}

//--------------------------------------------------------------------

template< typename FieldT >
TimeAdvanceMomentum<FieldT>::
~TimeAdvanceMomentum()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
TimeAdvanceMomentum<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();

  const SingleValue& dt = dt_->field_ref();
  const FieldT& momHat = momHat_->field_ref();
  const FieldT& gradP = gradP_->field_ref();

  phi <<= momHat + dt*gradP;
}

//--------------------------------------------------------------------

template< typename FieldT >
TimeAdvanceMomentum<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& momHatTag,
                  const Expr::Tag& rhsTag,
                  const WasatchCore::TimeIntegrator timeIntInfo )
  : ExpressionBuilder(result),
    solnVarName_( result.name() ),
    momHatt_( momHatTag ),
    rhst_ ( rhsTag ),
    timeIntInfo_( timeIntInfo )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TimeAdvanceMomentum<FieldT>::Builder::build() const
{
  return new TimeAdvanceMomentum<FieldT>( solnVarName_, momHatt_, rhst_, timeIntInfo_ );
}

//--------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class TimeAdvanceMomentum< SpatialOps::SVolField >;
template class TimeAdvanceMomentum< SpatialOps::XVolField >;
template class TimeAdvanceMomentum< SpatialOps::YVolField >;
template class TimeAdvanceMomentum< SpatialOps::ZVolField >;
template class TimeAdvanceMomentum< SpatialOps::Particle::ParticleField >;
//==========================================================================
