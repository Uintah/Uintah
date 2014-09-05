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

#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <Core/Exceptions/InvalidValue.h>

template< typename FieldT >
Expr::Tag ScalarRHS<FieldT>::resolve_field_tag( const FieldSelector field,
                                                const FieldTagInfo& info )
{
  Expr::Tag tag;
  const FieldTagInfo::const_iterator ifld = info.find( field );
  if( ifld != info.end() ) tag = ifld->second;
  return tag;
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::ScalarRHS( const FieldTagInfo& fieldTags,
                              const std::vector<Expr::Tag>& srcTags,
                              const Expr::Tag densityTag,
                              const Expr::Tag volFracTag,
                              const Expr::Tag xAreaFracTag,
                              const Expr::Tag yAreaFracTag,
                              const Expr::Tag zAreaFracTag,
                              const bool isConstDensity )
  : Expr::Expression<FieldT>(),

    convTagX_( ScalarRHS<FieldT>::resolve_field_tag( CONVECTIVE_FLUX_X, fieldTags ) ),
    convTagY_( ScalarRHS<FieldT>::resolve_field_tag( CONVECTIVE_FLUX_Y, fieldTags ) ),
    convTagZ_( ScalarRHS<FieldT>::resolve_field_tag( CONVECTIVE_FLUX_Z, fieldTags ) ),

    diffTagX_( ScalarRHS<FieldT>::resolve_field_tag( DIFFUSIVE_FLUX_X, fieldTags ) ),
    diffTagY_( ScalarRHS<FieldT>::resolve_field_tag( DIFFUSIVE_FLUX_Y, fieldTags ) ),
    diffTagZ_( ScalarRHS<FieldT>::resolve_field_tag( DIFFUSIVE_FLUX_Z, fieldTags ) ),

    haveConvection_( convTagX_ != Expr::Tag() || convTagY_ != Expr::Tag() || convTagZ_ != Expr::Tag() ),
    haveDiffusion_ ( diffTagX_ != Expr::Tag() || diffTagY_ != Expr::Tag() || diffTagZ_ != Expr::Tag() ),

    doXDir_( convTagX_ != Expr::Tag() || diffTagX_ != Expr::Tag() ),
    doYDir_( convTagY_ != Expr::Tag() || diffTagY_ != Expr::Tag() ),
    doZDir_( convTagZ_ != Expr::Tag() || diffTagZ_ != Expr::Tag() ),

    volFracTag_( volFracTag ),

    xAreaFracTag_( xAreaFracTag ),
    yAreaFracTag_( yAreaFracTag ),
    zAreaFracTag_( zAreaFracTag ),

    haveVolFrac_( volFracTag_ != Expr::Tag() ),
    haveXAreaFrac_( xAreaFracTag_ != Expr::Tag() ),
    haveYAreaFrac_( yAreaFracTag_ != Expr::Tag() ),
    haveZAreaFrac_( zAreaFracTag_ != Expr::Tag() ),

    densityTag_    ( densityTag     ),
    isConstDensity_( isConstDensity ),
    srcTags_       ( srcTags        )
{
  srcTags_.push_back( ScalarRHS<FieldT>::resolve_field_tag( SOURCE_TERM, fieldTags ) );
  nullify_fields();

  const bool is3d = doXDir_ && doYDir_ && doZDir_;
  const bool haveAreaFrac = haveXAreaFrac_ || haveYAreaFrac_ || haveZAreaFrac_;
  if( is3d && haveAreaFrac ){
    if( !( haveXAreaFrac_ && haveYAreaFrac_ && haveZAreaFrac_ ) ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "In 3D, it is expected that if one area fraction is provided, they all are..."
          << std::endl << std::endl;
      throw std::invalid_argument( msg.str() );
    }
  }

  this->set_gpu_runnable( true );
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::~ScalarRHS()
{}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::nullify_fields()
{
  xConvFlux_ = NULL;  yConvFlux_ = NULL;  zConvFlux_ = NULL;
  xDiffFlux_ = NULL;  yDiffFlux_ = NULL;  zDiffFlux_ = NULL;
  divOpX_    = NULL;  divOpY_    = NULL;  divOpZ_    = NULL;
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<XFluxT   >::type& xFluxFM  = fml.field_manager<XFluxT   >();
  const typename Expr::FieldMgrSelector<YFluxT   >::type& yFluxFM  = fml.field_manager<YFluxT   >();
  const typename Expr::FieldMgrSelector<ZFluxT   >::type& zFluxFM  = fml.field_manager<ZFluxT   >();
  const typename Expr::FieldMgrSelector<FieldT   >::type& scalarFM = fml.field_manager<FieldT   >();
  const typename Expr::FieldMgrSelector<XVolField>::type& xVolFM   = fml.field_manager<XVolField>();
  const typename Expr::FieldMgrSelector<YVolField>::type& yVolFM   = fml.field_manager<YVolField>();
  const typename Expr::FieldMgrSelector<ZVolField>::type& zVolFM   = fml.field_manager<ZVolField>();
  const typename Expr::FieldMgrSelector<SVolField>::type& sVolFM   = fml.field_manager<SVolField>();

  if( haveConvection_ ){
    if( doXDir_ )  xConvFlux_ = &xFluxFM.field_ref( convTagX_ );
    if( doYDir_ )  yConvFlux_ = &yFluxFM.field_ref( convTagY_ );
    if( doZDir_ )  zConvFlux_ = &zFluxFM.field_ref( convTagZ_ );
  }

  if( haveDiffusion_ ){
    if( doXDir_ )  xDiffFlux_ = &xFluxFM.field_ref( diffTagX_ );
    if( doYDir_ )  yDiffFlux_ = &yFluxFM.field_ref( diffTagY_ );
    if( doZDir_ )  zDiffFlux_ = &zFluxFM.field_ref( diffTagZ_ );
  }

  if ( haveVolFrac_ ) {
    volfrac_ = &sVolFM.field_ref( volFracTag_ );
  }

  if ( doXDir_ && haveXAreaFrac_ ) xareafrac_ = &xVolFM.field_ref( xAreaFracTag_ );
  if ( doYDir_ && haveYAreaFrac_ ) yareafrac_ = &yVolFM.field_ref( yAreaFracTag_ );
  if ( doZDir_ && haveZAreaFrac_ ) zareafrac_ = &zVolFM.field_ref( zAreaFracTag_ );

  srcTerm_.clear();
  for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
    if( isrc->context() != Expr::INVALID_CONTEXT ) {
      srcTerm_.push_back( &scalarFM.field_ref( *isrc ) );
      if (isConstDensity_){
        rho_ = &sVolFM.field_ref( densityTag_ );
      }
    }
  }
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( haveConvection_ ){
    if( doXDir_ )  exprDeps.requires_expression( convTagX_ );
    if( doYDir_ )  exprDeps.requires_expression( convTagY_ );
    if( doZDir_ )  exprDeps.requires_expression( convTagZ_ );
  }

  if( haveDiffusion_ ){
    if( doXDir_ )  exprDeps.requires_expression( diffTagX_ );
    if( doYDir_ )  exprDeps.requires_expression( diffTagY_ );
    if( doZDir_ )  exprDeps.requires_expression( diffTagZ_ );
  }

  if( haveVolFrac_   ) exprDeps.requires_expression( volFracTag_   );
  if( doXDir_ && haveXAreaFrac_ ) exprDeps.requires_expression( xAreaFracTag_ );
  if( doYDir_ && haveYAreaFrac_ ) exprDeps.requires_expression( yAreaFracTag_ );
  if( doZDir_ && haveZAreaFrac_ ) exprDeps.requires_expression( zAreaFracTag_ );

  for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
    if( isrc->context() != Expr::INVALID_CONTEXT ){
      exprDeps.requires_expression( *isrc );
      if ( isConstDensity_ )
        exprDeps.requires_expression( densityTag_ );
    }
  }
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doXDir_ )  divOpX_ = opDB.retrieve_operator<DivX>();
  if( doYDir_ )  divOpY_ = opDB.retrieve_operator<DivY>();
  if( doZDir_ )  divOpZ_ = opDB.retrieve_operator<DivZ>();

  for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
    if( (isrc->context() != Expr::INVALID_CONTEXT) && isConstDensity_)
      densityInterpOp_ = opDB.retrieve_operator<DensityInterpT>();
  }
  if( haveVolFrac_   ) volFracInterpOp_   = opDB.retrieve_operator<SVolToFieldTInterpT>();
  if( doXDir_ && haveXAreaFrac_ ) xAreaFracInterpOp_ = opDB.retrieve_operator<XVolToXFluxInterpT >();
  if( doYDir_ && haveYAreaFrac_ ) yAreaFracInterpOp_ = opDB.retrieve_operator<YVolToYFluxInterpT >();
  if( doZDir_ && haveZAreaFrac_ ) zAreaFracInterpOp_ = opDB.retrieve_operator<ZVolToZFluxInterpT >();

}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarRHS<FieldT>::evaluate()
{
  using namespace SpatialOps;

  FieldT& rhs = this->value();

  // note that all of this logic is in place so that when we do the
  // actual calculations we can be more efficient (eliminate temporaries,
  // inline things, etc.)

  if( doXDir_ && doYDir_ && doZDir_ && haveConvection_ && haveDiffusion_ ){
    // inline everything
    if( haveXAreaFrac_ ){ // previous error checking enforces that y and z area fractions are also present
      rhs <<= -(*divOpX_)( (*xAreaFracInterpOp_)(*xareafrac_) * ( *xConvFlux_ + *xDiffFlux_ ) )
              -(*divOpY_)( (*yAreaFracInterpOp_)(*yareafrac_) * ( *yConvFlux_ + *yDiffFlux_ ) )
              -(*divOpZ_)( (*zAreaFracInterpOp_)(*zareafrac_) * ( *zConvFlux_ + *zDiffFlux_ ) );
    }
    else{
      rhs <<= -(*divOpX_)( *xConvFlux_ + *xDiffFlux_ )
              -(*divOpY_)( *yConvFlux_ + *yDiffFlux_ )
              -(*divOpZ_)( *zConvFlux_ + *zDiffFlux_ );
    }
  }
  else{
    // handle 2D and 1D cases - not quite as efficient since we won't be
    // running as many production scale calculations in these configurations

    if( doXDir_ ){
      if( haveConvection_ && haveDiffusion_ ){
        if( haveXAreaFrac_ )
          rhs <<= -(*divOpX_)( (*xAreaFracInterpOp_)(*xareafrac_) * (*xConvFlux_ + *xDiffFlux_) );
        else
          rhs <<= -(*divOpX_)( *xConvFlux_ + *xDiffFlux_ );
      }
      else if( haveConvection_ ){
        if( haveXAreaFrac_ )
          rhs <<= -(*divOpX_)( (*xAreaFracInterpOp_)(*xareafrac_) * *xConvFlux_ );
        else
          rhs <<= -(*divOpX_)( *xConvFlux_ );
      }
      else if( haveDiffusion_ ){
        if( haveXAreaFrac_ )
          rhs <<= -(*divOpX_)( (*xAreaFracInterpOp_)(*xareafrac_) * *xDiffFlux_ );
        else
          rhs <<= -(*divOpX_)( *xDiffFlux_ );
      }
    }
    else{
      rhs <<= 0.0; // zero so that we can sum in Y and Z contributions as necessary
    }

    if( doYDir_ ){
      if( haveConvection_ && haveDiffusion_ ){
        if( haveYAreaFrac_ )
          rhs <<= rhs -(*divOpY_)( (*yAreaFracInterpOp_)(*yareafrac_) * (*yConvFlux_ + *yDiffFlux_) );
        else
          rhs <<= rhs -(*divOpY_)( *yConvFlux_ + *yDiffFlux_ );
      }
      else if( haveConvection_ ){
        if( haveYAreaFrac_ )
          rhs <<= rhs -(*divOpY_)( (*yAreaFracInterpOp_)(*yareafrac_) * *yConvFlux_ );
        else
          rhs <<= rhs -(*divOpY_)( *yConvFlux_ );
      }
      else if( haveDiffusion_ ){
        if( haveYAreaFrac_ )
          rhs <<= rhs -(*divOpY_)( (*yAreaFracInterpOp_)(*yareafrac_) * *yDiffFlux_ );
        else
          rhs <<= rhs -(*divOpY_)( *yDiffFlux_ );
      }
    }

    if( doZDir_ ){
      if( haveConvection_ && haveDiffusion_ ){
        if( haveZAreaFrac_ )
          rhs <<= rhs -(*divOpZ_)( (*zAreaFracInterpOp_)(*zareafrac_) * (*zConvFlux_ + *zDiffFlux_) );
        else
          rhs <<= rhs -(*divOpZ_)( *zConvFlux_ + *zDiffFlux_ );
      }
      else if( haveConvection_ ){
        if( haveZAreaFrac_ )
          rhs <<= rhs -(*divOpZ_)( (*zAreaFracInterpOp_)(*zareafrac_) * *zConvFlux_ );
        else
          rhs <<= rhs -(*divOpZ_)( *zConvFlux_ );
      }
      else if( haveDiffusion_ ){
        if( haveZAreaFrac_ )
          rhs <<= rhs -(*divOpZ_)( (*zAreaFracInterpOp_)(*zareafrac_) * *zDiffFlux_ );
        else
          rhs <<= rhs -(*divOpZ_)( *zDiffFlux_ );
      }
    }
  } // 2D and 1D cases

  // accumulate source terms in.  This isn't quite as efficient
  // because we don't have a great way to inline all of this yet.
  typename SrcVec::const_iterator isrc;
  for( isrc=srcTerm_.begin(); isrc!=srcTerm_.end(); ++isrc ) {
    if (isConstDensity_) {
      const double densVal = (*rho_)[0];
      if (haveVolFrac_) rhs <<= rhs + (**isrc / densVal ) * (*volFracInterpOp_)(*volfrac_);
      else              rhs <<= rhs + (**isrc / densVal );
    }
    else {
      if ( haveVolFrac_ )  rhs <<= rhs + **isrc * (*volFracInterpOp_)(*volfrac_);
      else                 rhs <<= rhs + **isrc;
    }
  }
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::Builder::Builder( const Expr::Tag& result,
                                     const FieldTagInfo& fieldInfo,
                                     const std::vector<Expr::Tag>& sources,
                                     const Expr::Tag& densityTag,
                                     const Expr::Tag& volFracTag,
                                     const Expr::Tag& xAreaFracTag,
                                     const Expr::Tag& yAreaFracTag,
                                     const Expr::Tag& zAreaFracTag,
                                     const bool isConstDensity )
  : ExpressionBuilder(result),
    info_          ( fieldInfo ),
    srcT_          ( sources      ),
    volfracT_      ( volFracTag   ),
    xareafracT_    ( xAreaFracTag ),
    yareafracT_    ( yAreaFracTag ),
    zareafracT_    ( zAreaFracTag ),
    densityT_      ( densityTag   ),
    isConstDensity_( isConstDensity )
{}

//------------------------------------------------------------------

template< typename FieldT >
ScalarRHS<FieldT>::Builder::Builder( const Expr::Tag& result,
                                     const FieldTagInfo& fieldInfo,
                                     const Expr::Tag& densityTag,
                                     const Expr::Tag& volFracTag,
                                     const Expr::Tag& xAreaFracTag,
                                     const Expr::Tag& yAreaFracTag,
                                     const Expr::Tag& zAreaFracTag,
                                     const bool isConstDensity )
  : ExpressionBuilder(result),
    info_          ( fieldInfo ),
    volfracT_      ( volFracTag   ),
    xareafracT_    ( xAreaFracTag ),
    yareafracT_    ( yAreaFracTag ),
    zareafracT_    ( zAreaFracTag ),
    densityT_      ( densityTag   ),
    isConstDensity_( isConstDensity )
{}

//------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ScalarRHS<FieldT>::Builder::build() const
{
  return new ScalarRHS<FieldT>( info_, srcT_, densityT_, volfracT_, xareafracT_, yareafracT_, zareafracT_, isConstDensity_ );
}
//------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class ScalarRHS< SpatialOps::structured::SVolField >;
//==========================================================================
