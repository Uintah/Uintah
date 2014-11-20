/*
 * The MIT License
 *
 * Copyright (c) 2014 The University of Utah
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

/**
 *  \file   SimpleEmission.cc
 *  \date   Oct 13, 2014
 *  \author "James C. Sutherland"
 */

#include <CCA/Components/Wasatch/Expressions/SimpleEmission.h>

//--------------------------------------------------------------------

template< typename FieldT >
SimpleEmission<FieldT>::
SimpleEmission( const Expr::Tag& temperatureTag,
                const Expr::Tag& envTempTag,
                const double envTemp,
                const Expr::Tag& absCoefTag )
  : Expr::Expression<FieldT>(),
    temperatureTag_( temperatureTag ),
    envTempTag_    ( envTempTag     ),
    absCoefTag_    ( absCoefTag     ),
    envTempValue_  ( envTemp        ),
    hasAbsCoef_     ( absCoefTag != Expr::Tag() ),
    hasConstEnvTemp_( envTempTag == Expr::Tag() )
{}

//--------------------------------------------------------------------

template< typename FieldT >
SimpleEmission<FieldT>::
~SimpleEmission()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
SimpleEmission<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( temperatureTag_ );
  if(!hasConstEnvTemp_ ) exprDeps.requires_expression( envTempTag_ );
  if( hasAbsCoef_      ) exprDeps.requires_expression( absCoefTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
SimpleEmission<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  temperature_ = &fm.field_ref( temperatureTag_ );
  if(!hasConstEnvTemp_ ) envTemp_ = &fm.field_ref( envTempTag_ );
  if( hasAbsCoef_      ) absCoef_ = &fm.field_ref( absCoefTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
SimpleEmission<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& divQ = this->value();
  const double sigma = 5.67037321e-8; // Stefan-Boltzmann constant, W/(m^2 K^4)
  if( hasConstEnvTemp_ ){
    if( hasAbsCoef_ ) divQ <<= *absCoef_ * sigma * ( pow( *temperature_, 4 ) - pow( envTempValue_, 4 ) );
    else              divQ <<=             sigma * ( pow( *temperature_, 4 ) - pow( envTempValue_, 4 ) );
  }
  else{
    if( hasAbsCoef_ ) divQ <<= *absCoef_ * sigma * ( pow( *temperature_, 4 ) - pow( *envTemp_, 4 ) );
    else              divQ <<=             sigma * ( pow( *temperature_, 4 ) - pow( *envTemp_, 4 ) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
SimpleEmission<FieldT>::
Builder::Builder( const Expr::Tag divQTag,
                  const Expr::Tag temperatureTag,
                  const Expr::Tag envTempTag,
                  const Expr::Tag absCoefTag )
  : ExpressionBuilder( divQTag ),
    temperatureTag_( temperatureTag ),
    envTempTag_( envTempTag ),
    absCoefTag_( absCoefTag ),
    envTemp_   ( 0.0        )
{}

template< typename FieldT >
SimpleEmission<FieldT>::
Builder::Builder( const Expr::Tag divQTag,
                  const Expr::Tag temperatureTag,
                  const double envTemperature,
                  const Expr::Tag absCoefTag )
  : ExpressionBuilder( divQTag ),
    temperatureTag_( temperatureTag ),
    envTempTag_    (                ),
    absCoefTag_    ( absCoefTag     ),
    envTemp_       ( envTemperature )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
SimpleEmission<FieldT>::
Builder::build() const
{
  return new SimpleEmission<FieldT>( temperatureTag_,envTempTag_,envTemp_,absCoefTag_ );
}

//--------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class SimpleEmission<SVolField>;
