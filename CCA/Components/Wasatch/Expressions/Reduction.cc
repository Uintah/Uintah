
#include "Reduction.h"
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Variables/VarLabel.h>

template< typename ReductionOpT >
ReductionEnum get_reduction_name();

template<>
ReductionEnum get_reduction_name<Uintah::Reductions::Min<double> >() {
  return ReduceMin;
}

template<>
ReductionEnum get_reduction_name<Uintah::Reductions::Max<double> >() {
  return ReduceMax;
}

template<>
ReductionEnum get_reduction_name<Uintah::Reductions::Sum<double> >() {
  return ReduceSum;
}

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename SrcFieldT, typename ReductionOpT >
Reduction<SrcFieldT,ReductionOpT>::
Reduction( const Expr::Tag& resultTag,
           const Expr::Tag& srcTag,
           bool printVar )
: ReductionBase( resultTag,
                 srcTag,
                 Uintah::VarLabel::create( resultTag.name() + "_uintah_reduction_var",
                                                            Uintah::ReductionVariable<double, ReductionOpT >::getTypeDescription()
                                                            ),
                 get_reduction_name<ReductionOpT>(),
                 printVar )

{}

//--------------------------------------------------------------------

template< typename SrcFieldT, typename ReductionOpT >
Reduction<SrcFieldT,ReductionOpT>::
~Reduction()
{}

//--------------------------------------------------------------------

template< typename SrcFieldT,
          typename ReductionOpT >
void
Reduction<SrcFieldT,ReductionOpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( this->srcTag_ );
}

//--------------------------------------------------------------------

template< typename SrcFieldT,
          typename ReductionOpT >
void
Reduction<SrcFieldT,ReductionOpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  src_ = &fml.template field_ref< SrcFieldT >( this->srcTag_ );
}

//--------------------------------------------------------------------

template< typename SrcFieldT,
          typename ReductionOpT >
void
Reduction<SrcFieldT,ReductionOpT>::
evaluate()
{
  double& result = this->value();
  
  using namespace SpatialOps;
  
  switch (reductionName_) {
      
    case ReduceMin:
      result = field_min_interior(*src_);
      break;
      
    case ReduceMax:
      result = field_max_interior(*src_);
      break;

    case ReduceSum:
      result = field_sum_interior(*src_);
      break;
      
    default:
      result = 0.0;
      break;
  }
}

//--------------------------------------------------------------------

template< typename SrcFieldT,
          typename ReductionOpT >
Reduction<SrcFieldT,ReductionOpT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& srcTag,
                  bool printVar )
  : ExpressionBuilder( resultTag ),
    resultTag_       (resultTag  ),
    srcTag_          ( srcTag    ),
    printVar_        (printVar   )
{}

//--------------------------------------------------------------------

template< typename SrcFieldT, typename ReductionOpT >
Expr::ExpressionBase*
Reduction<SrcFieldT,ReductionOpT>::
Builder::build() const
{
  return new Reduction<SrcFieldT, ReductionOpT>( resultTag_, srcTag_, printVar_ );
}

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
#define DECLARE_MIN_REDUCTION(VOL) template class Reduction< VOL, Uintah::Reductions::Min<double> >;
#define DECLARE_MAX_REDUCTION(VOL) template class Reduction< VOL, Uintah::Reductions::Max<double> >;
#define DECLARE_SUM_REDUCTION(VOL) template class Reduction< VOL, Uintah::Reductions::Sum<double> >;
DECLARE_MIN_REDUCTION(SVolField);
DECLARE_MIN_REDUCTION(XVolField);
DECLARE_MIN_REDUCTION(YVolField);
DECLARE_MIN_REDUCTION(ZVolField);

DECLARE_MAX_REDUCTION(SVolField);
DECLARE_MAX_REDUCTION(XVolField);
DECLARE_MAX_REDUCTION(YVolField);
DECLARE_MAX_REDUCTION(ZVolField);

DECLARE_SUM_REDUCTION(SVolField);
DECLARE_SUM_REDUCTION(XVolField);
DECLARE_SUM_REDUCTION(YVolField);
DECLARE_SUM_REDUCTION(ZVolField);

//==========================================================================
