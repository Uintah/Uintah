
#include "Reduction.h"
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <Core/Disclosure/TypeDescription.h>

template< typename ReductionOpT > ReductionEnum get_reduction_name();
template<> ReductionEnum get_reduction_name<ReductionMinOpT>() { return ReduceMin; }
template<> ReductionEnum get_reduction_name<ReductionMaxOpT>() { return ReduceMax; }
template<> ReductionEnum get_reduction_name<ReductionSumOpT>() { return ReduceSum; }

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename SrcFieldT, typename ReductionOpT >
Reduction<SrcFieldT,ReductionOpT>::
Reduction( const Expr::Tag& resultTag,
           const Expr::Tag& srcTag,
           const bool printVar )
: ReductionBase( resultTag,
                 srcTag,
                 get_reduction_name<ReductionOpT>(),
                 printVar )
{
   src_ = this->template create_field_request<SrcFieldT>(srcTag);
}

//--------------------------------------------------------------------

template< typename SrcFieldT, typename ReductionOpT >
Reduction<SrcFieldT,ReductionOpT>::
~Reduction()
{}

//--------------------------------------------------------------------

template<typename FieldT, typename ReductionOpT>
struct populate_result{
  static inline void doit( const FieldT& src,
                           const ReductionEnum reduction,
                           SpatialOps::SingleValueField& result )
  {
    using namespace SpatialOps;
     switch( reduction ){
       case ReduceMin: result <<= field_min_interior(src); break;
       case ReduceMax: result <<= field_max_interior(src); break;
       case ReduceSum: result <<= field_sum_interior(src); break;
       default:        result <<= 0.0;                     break;
     }
  }
};

template<typename ReductionOpT>
struct populate_result<SpatialOps::SingleValueField,ReductionOpT>{
  static inline void doit( const SpatialOps::SingleValueField& src,
                           const ReductionEnum reduction,
                           SpatialOps::SingleValueField& result )
  {
    using namespace SpatialOps;
    result <<= src;
  }
};

//--------------------------------------------------------------------

template< typename SrcFieldT, typename ReductionOpT >
void
Reduction<SrcFieldT,ReductionOpT>::
evaluate()
{
  populate_result<SrcFieldT,ReductionOpT>::doit( src_->field_ref(), reductionName_, this->value() );
}

//--------------------------------------------------------------------

template< typename SrcFieldT, typename ReductionOpT >
Reduction<SrcFieldT,ReductionOpT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& srcTag,
                  const bool printVar )
  : ExpressionBuilder( resultTag ),
    resultTag_       ( resultTag ),
    srcTag_          ( srcTag    ),
    printVar_        ( printVar  )
{}

//--------------------------------------------------------------------

template< typename SrcFieldT, typename ReductionOpT >
Expr::ExpressionBase*
Reduction<SrcFieldT,ReductionOpT>::
Builder::build() const
{
  return new Reduction<SrcFieldT,ReductionOpT>( resultTag_, srcTag_, printVar_ );
}

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>

#define DECLARE( T )                           \
  template class Reduction<T,ReductionMinOpT>; \
  template class Reduction<T,ReductionMaxOpT>; \
  template class Reduction<T,ReductionSumOpT>;

DECLARE( SpatialOps::SingleValueField )
DECLARE( SVolField                                )
DECLARE( XVolField                                )
DECLARE( YVolField                                )
DECLARE( ZVolField                                )
//==========================================================================
