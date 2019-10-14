/*
 * The MIT License
 *
 * Copyright (c) 2014-2018 The University of Utah
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
    envTempValue_   ( envTemp ),
    hasAbsCoef_     ( absCoefTag != Expr::Tag() ),
    hasConstEnvTemp_( envTempTag == Expr::Tag() )
{
  temperature_ = this->template create_field_request<FieldT>(temperatureTag);
  if(!hasConstEnvTemp_)  envTemp_ = this->template create_field_request<FieldT>(envTempTag);
  if( hasAbsCoef_     )  absCoef_ = this->template create_field_request<FieldT>(absCoefTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
SimpleEmission<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& divQ = this->value();
  const FieldT& temperature = temperature_->field_ref();
  
  const double sigma = 5.67037321e-8; // Stefan-Boltzmann constant, W/(m^2 K^4)
  if( hasConstEnvTemp_ ){
    const double envTerm = pow( envTempValue_, 4 );
    if( hasAbsCoef_ ) divQ <<= absCoef_->field_ref() * sigma * ( pow( temperature, 4 ) - envTerm );
    else              divQ <<=                         sigma * ( pow( temperature, 4 ) - envTerm );
  }
  else{
    const FieldT& envTemp = envTemp_->field_ref();
    if( hasAbsCoef_ ) divQ <<= absCoef_->field_ref() * sigma * ( pow( temperature, 4 ) - pow( envTemp, 4 ) );
    else              divQ <<=                         sigma * ( pow( temperature, 4 ) - pow( envTemp, 4 ) );
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
    envTempTag_    ( envTempTag     ),
    absCoefTag_    ( absCoefTag     ),
    envTemp_       ( 0.0            )
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
