/*
 * The MIT License
 *
 * Copyright (c) 2014-2017 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/TimeAdvance.h>
#include <CCA/Components/Wasatch/TagNames.h>

#include <spatialops/Nebo.h>


template< typename FieldT >
TimeAdvance<FieldT>::
TimeAdvance( const std::string& solnVarName,
             const Expr::Tag& phiOldTag,
             const Expr::Tag& rhsTag,
             const WasatchCore::TimeIntegrator timeIntInfo )
: Expr::Expression<FieldT>(),
  phiOldt_    ( phiOldTag                         ),
  rhst_       ( rhsTag                            ),
  dtt_        ( WasatchCore::TagNames::self().dt      ),
  rkstaget_   ( WasatchCore::TagNames::self().rkstage ),
  timeIntInfo_( timeIntInfo                       )
{
  this->set_gpu_runnable( true );
  
   phiOld_ = this->template create_field_request<FieldT>(phiOldTag);
   rhs_ = this->template create_field_request<FieldT>(rhsTag);
   dt_ = this->template create_field_request<SingleValue>(WasatchCore::TagNames::self().dt);
   rkStage_ = this->template create_field_request<SingleValue>(WasatchCore::TagNames::self().rkstage);
}

//--------------------------------------------------------------------

template< typename FieldT >
TimeAdvance<FieldT>::
~TimeAdvance()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
TimeAdvance<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();

  const double a2 = timeIntInfo_.alpha[1];
  const double a3 = timeIntInfo_.alpha[2];

  const double b2 = timeIntInfo_.beta[1];
  const double b3 = timeIntInfo_.beta[2];

  const SingleValue& rkStage = rkStage_->field_ref();
  const SingleValue& dt = dt_->field_ref();
  const FieldT& phiOld = phiOld_->field_ref();
  const FieldT& rhs = rhs_->field_ref();
  // Since rkStage_ is a SpatialField, we cannot dereference it and use "if"
  // statements since that will break GPU execution. Therefore, we use cond here
  // to allow GPU execution of this expression, despite the fact that it causes
  // branching on the inner loop.  However, the same branch is followed for all
  // points in the loop, which shouldn't degrade GPU execution and branch
  // prediction on CPU should limit performance degradation.
  phi <<= cond( rkStage == 1.0,      phiOld +              dt * rhs )
              ( rkStage == 2.0, a2 * phiOld + b2 * ( phi + dt * rhs ) )
              ( rkStage == 3.0, a3 * phiOld + b3 * ( phi + dt * rhs ) )
              ( 0.0 ); // should never get here.
}

//--------------------------------------------------------------------

template< typename FieldT >
TimeAdvance<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& rhsTag,
                  const WasatchCore::TimeIntegrator timeIntInfo )
  : ExpressionBuilder(result),
    solnVarName_( result.name() ),
    phiOldt_( Expr::Tag(solnVarName_, Expr::STATE_N) ),
    rhst_( rhsTag ),
    timeIntInfo_( timeIntInfo )
{}

//--------------------------------------------------------------------

template< typename FieldT >
TimeAdvance<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& phiOldTag,
                  const Expr::Tag& rhsTag,
                  const WasatchCore::TimeIntegrator timeIntInfo )
: ExpressionBuilder( result ),
  solnVarName_( result.name() ),
  phiOldt_( phiOldTag ),
  rhst_( rhsTag ),
  timeIntInfo_( timeIntInfo )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TimeAdvance<FieldT>::Builder::build() const
{
  return new TimeAdvance<FieldT>( solnVarName_, phiOldt_, rhst_, timeIntInfo_ );
}

//--------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class TimeAdvance< SpatialOps::SVolField >;
template class TimeAdvance< SpatialOps::XVolField >;
template class TimeAdvance< SpatialOps::YVolField >;
template class TimeAdvance< SpatialOps::ZVolField >;
template class TimeAdvance< SpatialOps::Particle::ParticleField >;
//==========================================================================
