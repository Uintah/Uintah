/*
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

#include "ScalabilityTestSrc.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT >
ScalabilityTestSrc<FieldT>::
ScalabilityTestSrc( const Expr::Tag& varTag,
                    const int nvar )
: Expr::Expression<FieldT>(),
  phiTag_( varTag ),
  nvar_  ( nvar   )
{
  tmpVec_.resize( nvar, 0.0 );
}

//------------------------------------------------------------------

template< typename FieldT >
ScalabilityTestSrc<FieldT>::~ScalabilityTestSrc()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
ScalabilityTestSrc<FieldT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  for( int i=0; i!=nvar_; ++i ){
    std::ostringstream nam;
    nam << phiTag_.name() << i;
    exprDeps.requires_expression( Expr::Tag( nam.str(), phiTag_.context() ) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ScalabilityTestSrc<FieldT>::bind_fields( const Expr::FieldManagerList& fml )
{
  phi_.clear();
  for( int i=0; i!=nvar_; ++i ){
    std::ostringstream nam;
    nam << phiTag_.name() << i;
    phi_.push_back( &fml.field_manager<FieldT>().field_ref( Expr::Tag(nam.str(),phiTag_.context()) ) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ScalabilityTestSrc<FieldT>::evaluate()
{
  FieldT& val = this->value();

  // pack iterators into a vector
  iterVec_.clear();
  for( typename FieldVecT::const_iterator ifld=phi_.begin(); ifld!=phi_.end(); ++ifld ){
    iterVec_.push_back( (*ifld)->begin() );
  }

  for( typename FieldT::iterator ival=val.begin(); ival!=val.end(); ++ival ){
    // unpack into temporary
    tmpVec_.clear();
    for( typename IterVec::const_iterator ii=iterVec_.begin(); ii!=iterVec_.end(); ++ii ){
      tmpVec_.push_back( **ii );
    }

    double src=1.0;
    for( std::vector<double>::const_iterator isrc=tmpVec_.begin(); isrc!=tmpVec_.end(); ++isrc ){
      src += exp(*isrc);
    }

    *ival = src;

    // advance iterators to next point
    for( typename IterVec::iterator ii=iterVec_.begin(); ii!=iterVec_.end(); ++ii ){
      ++(*ii);
    }
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ScalabilityTestSrc<FieldT>::Builder::build() const
{
  return new ScalabilityTestSrc( tag_, nvar_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
ScalabilityTestSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& phiTag,
         const int nvar )
: ExpressionBuilder(result),
  tag_ ( phiTag ),
  nvar_( nvar   )
{}

//--------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class ScalabilityTestSrc< SpatialOps::structured::SVolField >;
template class ScalabilityTestSrc< SpatialOps::structured::XVolField >;
template class ScalabilityTestSrc< SpatialOps::structured::YVolField >;
template class ScalabilityTestSrc< SpatialOps::structured::ZVolField >;
//==========================================================================
