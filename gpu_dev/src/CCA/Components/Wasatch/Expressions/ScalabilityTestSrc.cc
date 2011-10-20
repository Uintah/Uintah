#include "ScalabilityTestSrc.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT >
ScalabilityTestSrc<FieldT>::
ScalabilityTestSrc( const Expr::Tag varTag,
                    const int nvar,
                    const Expr::ExpressionID& id,
                    const Expr::ExpressionRegistry& reg )
: Expr::Expression<FieldT>( id, reg ),
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
ScalabilityTestSrc<FieldT>::Builder::
build( const Expr::ExpressionID& id,
      const Expr::ExpressionRegistry& reg ) const
{
  return new ScalabilityTestSrc( tag_, nvar_, id, reg );
}

//--------------------------------------------------------------------

template< typename FieldT >
ScalabilityTestSrc<FieldT>::Builder::
Builder( const Expr::Tag phiTag,
        const int nvar )
: tag_ ( phiTag ),
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
