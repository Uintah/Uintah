/*
 * The MIT License
 *
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

#include <CCA/Components/Wasatch/Expressions/TimeAdvance.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <CCA/Components/Wasatch/TagNames.h>


template< typename FieldT >
TimeAdvance<FieldT>::
TimeAdvance( const std::string& solnVarName,
            const Expr::Tag& phiOldTag,
            const Expr::Tag& rhsTag,
            const Wasatch::TimeIntegrator timeIntInfo )
: Expr::Expression<FieldT>(),
phioldt_        ( phiOldTag                                   ),
rhst_        ( rhsTag                                      ),
dtt_   ( Wasatch::TagNames::self().dt                ),
rkstaget_    ( Wasatch::TagNames::self().rkstage           ),
timeIntInfo_ ( timeIntInfo                                 ),
rkStage_     ( 1                                           )
{
  this->set_gpu_runnable( true );
  a_ = timeIntInfo_.alpha[rkStage_-1];
  b_ = timeIntInfo_.beta[rkStage_-1];
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
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phioldt_    );
  exprDeps.requires_expression( rhst_    );
  
  exprDeps.requires_expression( dtt_ );
  exprDeps.requires_expression( rkstaget_    );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TimeAdvance<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& svfm = fml.template field_manager<FieldT>();
  phiOld_     = &svfm.field_ref( phioldt_    );
  rhs_     = &svfm.field_ref( rhst_ );
  
  const Expr::FieldMgrSelector<SingleValue>::type& doublefm = fml.field_manager<SingleValue>();
  dt_ = &doublefm.field_ref(dtt_);
  rks_= &doublefm.field_ref(rkstaget_);
  rkStage_ =(int) (*rks_)[0];
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TimeAdvance<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
TimeAdvance<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  if (rkStage_ == 1)  {
    phi <<= *phiOld_ + *dt_ * *rhs_;
  } else {
    a_ = timeIntInfo_.alpha[rkStage_-1];
    b_ = timeIntInfo_.beta[rkStage_-1];
    phi <<= a_ * *phiOld_ + b_ * (phi + *dt_ * *rhs_);
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
TimeAdvance<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& rhsTag,
                  const Wasatch::TimeIntegrator timeIntInfo )
  : ExpressionBuilder(result),
    solnVarName_(result.name()),
    phioldt_(Expr::Tag(solnVarName_, Expr::STATE_N)),
    rhst_(rhsTag),
    timeIntInfo_(timeIntInfo)
{}

//--------------------------------------------------------------------

template< typename FieldT >
TimeAdvance<FieldT>::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& phiOldTag,
                 const Expr::Tag& rhsTag,
                 const Wasatch::TimeIntegrator timeIntInfo )
: ExpressionBuilder(result),
  solnVarName_(result.name()),
  phioldt_(phiOldTag),
  rhst_(rhsTag),
  timeIntInfo_(timeIntInfo)
{}


//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TimeAdvance<FieldT>::Builder::build() const
{
  return new TimeAdvance<FieldT>( solnVarName_, phioldt_, rhst_, timeIntInfo_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class TimeAdvance< SpatialOps::structured::SVolField >;
template class TimeAdvance< SpatialOps::structured::XVolField >;
template class TimeAdvance< SpatialOps::structured::YVolField >;
template class TimeAdvance< SpatialOps::structured::ZVolField >;
template class TimeAdvance< SpatialOps::Particle::ParticleField >;
//==========================================================================
