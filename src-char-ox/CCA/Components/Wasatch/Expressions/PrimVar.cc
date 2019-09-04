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

#ifndef PrimVar_h
#define PrimVar_h

#include <CCA/Components/Wasatch/Expressions/PrimVar.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
PrimVar( const Expr::Tag& rhoPhiTag,
         const Expr::Tag& rhoTag,
         const Expr::Tag& volFracTag)
  : Expr::Expression<FieldT>(),
    hasIntrusion_(volFracTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
  
  rhophi_ = this->template create_field_request<FieldT>(rhoPhiTag);
  rho_    = this->template create_field_request<DensT >(rhoTag   );

  if( hasIntrusion_ ) volfrac_ = this->template create_field_request<FieldT>(volFracTag);
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
~PrimVar()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template<>
void
PrimVar<SpatialOps::SVolField,SpatialOps::SVolField>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  const DensT& rho = rho_->field_ref();
  const FieldT& rhophi = rhophi_->field_ref();
  
  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( phi );
  *tmp <<= 1.0; // we need to set this to 1.0 so that we don't get random values in out-of-domain faces
  *tmp <<= (*interpOp_)( rho );

  if( hasIntrusion_ ) phi <<= volfrac_->field_ref() * rhophi / *tmp;
  else                phi <<= rhophi / *tmp;
}

//--------------------------------------------------------------------

template<>
void
PrimVar<SpatialOps::SVolField,SpatialOps::SVolField>::
evaluate()
{
  using namespace SpatialOps;
  typedef SpatialOps::SVolField FieldT;
  FieldT& phi = this->value();
  const FieldT& rho    = rho_   ->field_ref();
  const FieldT& rhophi = rhophi_->field_ref();
  if( hasIntrusion_ ) phi <<= volfrac_->field_ref() * rhophi / rho;
  else                phi <<= rhophi / rho;
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& rhoPhiTag,
                  const Expr::Tag& rhoTag,
                  const Expr::Tag& volFracTag )
  : ExpressionBuilder(result),
    rhophit_ ( rhoPhiTag  ),
    rhot_    ( rhoTag     ),
    volfract_( volFracTag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
Expr::ExpressionBase*
PrimVar<FieldT,DensT>::Builder::build() const
{
  return new PrimVar<FieldT,DensT>( rhophit_, rhot_, volfract_ );
}

//====================================================================
//  Explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class PrimVar< SpatialOps::SVolField, SpatialOps::SVolField >;
template class PrimVar< SpatialOps::XVolField, SpatialOps::SVolField >;
template class PrimVar< SpatialOps::YVolField, SpatialOps::SVolField >;
template class PrimVar< SpatialOps::ZVolField, SpatialOps::SVolField >;
//====================================================================

#endif // PrimVar_h
