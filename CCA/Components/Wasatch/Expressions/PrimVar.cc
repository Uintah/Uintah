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

#ifndef PrimVar_h
#define PrimVar_h

#include "PrimVar.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
PrimVar( const Expr::Tag& rhoPhiTag,
         const Expr::Tag& rhoTag )
  : Expr::Expression<FieldT>(),
    rhophit_( rhoPhiTag ),
    rhot_   ( rhoTag    )
{}

template< typename FieldT >
PrimVar<FieldT,FieldT>::
PrimVar( const Expr::Tag& rhoPhiTag,
         const Expr::Tag& rhoTag )
  : Expr::Expression<FieldT>(),
    rhophit_( rhoPhiTag ),
    rhot_   ( rhoTag    )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
~PrimVar()
{}

template< typename FieldT >
PrimVar<FieldT,FieldT>::
~PrimVar()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhophit_ );
  exprDeps.requires_expression( rhot_    );
}

template< typename FieldT >
void
PrimVar<FieldT,FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhophit_ );
  exprDeps.requires_expression( rhot_    );
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& phifm = fml.template field_manager<FieldT>();
  const Expr::FieldManager<DensT >& denfm = fml.template field_manager<DensT >();

  rhophi_ = &phifm.field_ref( rhophit_ );
  rho_    = &denfm.field_ref( rhot_    );
}

template< typename FieldT >
void
PrimVar<FieldT,FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& phifm = fml.template field_manager<FieldT>();
  rhophi_ = &phifm.field_ref( rhophit_ );
  rho_    = &phifm.field_ref( rhot_    );
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( phi );
  interpOp_->apply_to_field( *rho_, *tmp );
  phi <<= *rhophi_ / *tmp;
}

template< typename FieldT >
void
PrimVar<FieldT,FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  phi <<= *rhophi_ / *rho_;
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& rhoPhiTag,
                  const Expr::Tag& rhoTag )
  : ExpressionBuilder(result),
    rhophit_( rhoPhiTag ),
    rhot_   ( rhoTag    )
{}

template< typename FieldT >
PrimVar<FieldT,FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& rhoPhiTag,
                  const Expr::Tag& rhoTag )
  : ExpressionBuilder(result),
    rhophit_( rhoPhiTag ),
    rhot_   ( rhoTag    )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
Expr::ExpressionBase*
PrimVar<FieldT,DensT>::Builder::build() const
{
  return new PrimVar<FieldT,DensT>( rhophit_, rhot_ );
}

template< typename FieldT >
Expr::ExpressionBase*
PrimVar<FieldT,FieldT>::Builder::build() const
{
  return new PrimVar<FieldT,FieldT>( rhophit_, rhot_ );
}

//====================================================================
//  Explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class PrimVar< SpatialOps::structured::SVolField, SpatialOps::structured::SVolField >;
template class PrimVar< SpatialOps::structured::XVolField, SpatialOps::structured::SVolField >;
template class PrimVar< SpatialOps::structured::YVolField, SpatialOps::structured::SVolField >;
template class PrimVar< SpatialOps::structured::ZVolField, SpatialOps::structured::SVolField >;
//====================================================================


#endif // PrimVar_h
