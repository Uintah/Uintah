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

#ifndef Multiplier_h
#define Multiplier_h

#include "Multiplier.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename Field1T, typename Field2T >
Multiplier<Field1T,Field2T>::
Multiplier( const Expr::Tag& var1Tag,
            const Expr::Tag& var2Tag )
  : Expr::Expression<Field1T>(),
    var1t_( var1Tag ),
    var2t_( var2Tag )
{}

template< typename FieldT >
Multiplier<FieldT,FieldT>::
Multiplier( const Expr::Tag& var1Tag,
            const Expr::Tag& var2Tag )
  : Expr::Expression<FieldT>(),
    var1t_( var1Tag ),
    var2t_( var2Tag )
{}

//--------------------------------------------------------------------

template< typename Field1T, typename Field2T >
Multiplier<Field1T,Field2T>::
~Multiplier()
{}

template< typename FieldT >
Multiplier<FieldT,FieldT>::
~Multiplier()
{}

//--------------------------------------------------------------------

template< typename Field1T, typename Field2T >
void
Multiplier<Field1T,Field2T>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( var1t_ );
  exprDeps.requires_expression( var2t_ );
}

template< typename FieldT >
void
Multiplier<FieldT,FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( var1t_ );
  exprDeps.requires_expression( var2t_ );
}

//--------------------------------------------------------------------

template< typename Field1T, typename Field2T >
void
Multiplier<Field1T,Field2T>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<Field1T>& v1v2fm = fml.template field_manager<Field1T>();
  const Expr::FieldManager<Field2T>& var2fm = fml.template field_manager<Field2T>();

  var1_ = &v1v2fm.field_ref( var1t_ );
  var2_ = &var2fm.field_ref( var2t_ );
}

template< typename FieldT >
void
Multiplier<FieldT,FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& v1v2fm = fml.template field_manager<FieldT>();
  var1_ = &v1v2fm.field_ref( var1t_ );
  var2_ = &v1v2fm.field_ref( var2t_ );
}

//--------------------------------------------------------------------

template< typename Field1T, typename Field2T >
void
Multiplier<Field1T,Field2T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename Field1T, typename Field2T >
void
Multiplier<Field1T,Field2T>::
evaluate()
{
  using namespace SpatialOps;
  Field1T& v1v2 = this->value();
  SpatialOps::SpatFldPtr<Field1T> tmp = SpatialOps::SpatialFieldStore<Field1T>::self().get( v1v2 );
  interpOp_->apply_to_field( *var2_, *tmp );
  v1v2 <<= *var1_ * *tmp;
}

template< typename FieldT >
void
Multiplier<FieldT,FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& v1v2 = this->value();
  v1v2 <<= *var1_ * *var2_;
}

//--------------------------------------------------------------------

template< typename Field1T, typename Field2T >
Multiplier<Field1T,Field2T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& var1Tag,
                  const Expr::Tag& var2Tag )
  : ExpressionBuilder(result),
    var1t_( var1Tag ),
    var2t_( var2Tag )
{}

template< typename FieldT >
Multiplier<FieldT,FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& var1Tag,
                  const Expr::Tag& var2Tag )
  : ExpressionBuilder(result),
    var1t_( var1Tag ),
    var2t_( var2Tag )
{}

//--------------------------------------------------------------------

template< typename Field1T, typename Field2T >
Expr::ExpressionBase*
Multiplier<Field1T,Field2T>::Builder::build() const
{
  return new Multiplier<Field1T,Field2T>( var1t_, var2t_ );
}

template< typename FieldT >
Expr::ExpressionBase*
Multiplier<FieldT,FieldT>::Builder::build() const
{
  return new Multiplier<FieldT,FieldT>( var1t_,var2t_ );
}

//====================================================================
//  Explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class Multiplier< SpatialOps::structured::SVolField, SpatialOps::structured::SVolField >;
template class Multiplier< SpatialOps::structured::XVolField, SpatialOps::structured::SVolField >;
template class Multiplier< SpatialOps::structured::YVolField, SpatialOps::structured::SVolField >;
template class Multiplier< SpatialOps::structured::ZVolField, SpatialOps::structured::SVolField >;
//====================================================================


#endif // Multiplier_h
