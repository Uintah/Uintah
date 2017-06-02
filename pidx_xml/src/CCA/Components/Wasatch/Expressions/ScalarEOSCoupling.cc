/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

//-- Uintah Includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Wasatch/Expressions/ScalarEOSCoupling.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


Expr::Tag
resolve_tag( const FieldSelector field,
                   const FieldTagInfo& info )
{
  Expr::Tag tag;
  const FieldTagInfo::const_iterator ifld = info.find( field );
  if( ifld != info.end() ) tag = ifld->second;
  return tag;
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarEOSCoupling<FieldT>::ScalarEOSCoupling( const FieldTagInfo& fieldTags,
                                              const Expr::TagList& srcTags,
                                              const Expr::Tag& rhoStarTag,
                                              const Expr::Tag& dRhoDPhiTag,
                                              const bool isStrongForm )
  : Expr::Expression<FieldT>(),

    diffTagX_( resolve_tag( DIFFUSIVE_FLUX_X, fieldTags ) ),
    diffTagY_( resolve_tag( DIFFUSIVE_FLUX_Y, fieldTags ) ),
    diffTagZ_( resolve_tag( DIFFUSIVE_FLUX_Z, fieldTags ) ),

    dRhoDPhiTag_( dRhoDPhiTag ),
    srcTags_( srcTags ),

    haveDiffusion_ ( diffTagX_ != Expr::Tag() || diffTagY_ != Expr::Tag() || diffTagZ_ != Expr::Tag() ),

    doXDiff_( diffTagX_ != Expr::Tag() ),
    doYDiff_( diffTagY_ != Expr::Tag() ),
    doZDiff_( diffTagZ_ != Expr::Tag() ),

    doXDir_( diffTagX_ != Expr::Tag() ),
    doYDir_( diffTagY_ != Expr::Tag() ),
    doZDir_( diffTagZ_ != Expr::Tag() ),

    volFracTag_  ( resolve_tag( VOLUME_FRAC, fieldTags ) ),
    xAreaFracTag_( resolve_tag( AREA_FRAC_X, fieldTags ) ),
    yAreaFracTag_( resolve_tag( AREA_FRAC_Y, fieldTags ) ),
    zAreaFracTag_( resolve_tag( AREA_FRAC_Z, fieldTags ) ),

    haveVolFrac_  ( volFracTag_   != Expr::Tag() ),
    haveXAreaFrac_( xAreaFracTag_ != Expr::Tag() ),
    haveYAreaFrac_( yAreaFracTag_ != Expr::Tag() ),
    haveZAreaFrac_( zAreaFracTag_ != Expr::Tag() ),

    isStrongForm_( isStrongForm )
{
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
  rhoStar_ = this->template create_field_request<FieldT>(rhoStarTag);
  dRhoDPhi_ = this->template create_field_request<FieldT>(dRhoDPhiTag);
  
  if( doXDiff_ )  xDiffFlux_ = this->template create_field_request<XFluxT>( diffTagX_ );
  if( doYDiff_ )  yDiffFlux_ = this->template create_field_request<YFluxT>( diffTagY_ );
  if( doZDiff_ )  zDiffFlux_ = this->template create_field_request<ZFluxT>( diffTagZ_ );
  
  if ( haveVolFrac_ ) {
     volfrac_  = this->template create_field_request<FieldT>( volFracTag_);
  }
  
  if ( doXDir_ && haveXAreaFrac_ )  xareafrac_  = this->template create_field_request<XVolField>( xAreaFracTag_);
  if ( doYDir_ && haveYAreaFrac_ )  yareafrac_  = this->template create_field_request<YVolField>( yAreaFracTag_);
  if ( doZDir_ && haveZAreaFrac_ )  zareafrac_  = this->template create_field_request<ZVolField>( zAreaFracTag_);

  this->template create_field_vector_request<FieldT>(srcTags_, srcTerms_);
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarEOSCoupling<FieldT>::~ScalarEOSCoupling()
{}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarEOSCoupling<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doXDir_ )  divOpX_ = opDB.retrieve_operator<DivX>();
  if( doYDir_ )  divOpY_ = opDB.retrieve_operator<DivY>();
  if( doZDir_ )  divOpZ_ = opDB.retrieve_operator<DivZ>();

  for( std::vector<Expr::Tag>::const_iterator isrc=srcTags_.begin(); isrc!=srcTags_.end(); ++isrc ){
    if( (isrc->context() != Expr::INVALID_CONTEXT))
      densityInterpOp_ = opDB.retrieve_operator<DensityInterpT>();
  }
  if( haveVolFrac_   ) volFracInterpOp_   = opDB.retrieve_operator<SVolToFieldTInterpT>();
  if( doXDir_ && haveXAreaFrac_ ) xAreaFracInterpOp_ = opDB.retrieve_operator<XVolToXFluxInterpT >();
  if( doYDir_ && haveYAreaFrac_ ) yAreaFracInterpOp_ = opDB.retrieve_operator<YVolToYFluxInterpT >();
  if( doZDir_ && haveZAreaFrac_ ) zAreaFracInterpOp_ = opDB.retrieve_operator<ZVolToZFluxInterpT >();
}

//------------------------------------------------------------------

template< typename FieldT >
void ScalarEOSCoupling<FieldT>::evaluate()
{
  using namespace SpatialOps;

  FieldT& rhs = this->value();

  // note that all of this logic is in place so that when we do the
  // actual calculations we can be more efficient (eliminate temporaries,
  // inline things, etc.)

  if( doXDiff_ && doYDiff_ && doZDiff_ ){
    // inline everything
    const XFluxT& xDiffFlux  =  xDiffFlux_->field_ref();
    const YFluxT& yDiffFlux  =  yDiffFlux_->field_ref();
    const ZFluxT& zDiffFlux  =  zDiffFlux_->field_ref();
    
    if( haveXAreaFrac_ ){ // previous error checking enforces that y and z area fractions are also present
      const XVolField& xAreaFrac = xareafrac_->field_ref();
      const YVolField& yAreaFrac = yareafrac_->field_ref();
      const ZVolField& zAreaFrac = zareafrac_->field_ref();
      rhs <<= -(*divOpX_)( (*xAreaFracInterpOp_)(xAreaFrac) * ( xDiffFlux ) )
              -(*divOpY_)( (*yAreaFracInterpOp_)(yAreaFrac) * ( yDiffFlux ) )
              -(*divOpZ_)( (*zAreaFracInterpOp_)(zAreaFrac) * ( zDiffFlux ) );
    }
    else{
        rhs <<= -(*divOpX_)( xDiffFlux )
                -(*divOpY_)( yDiffFlux )
                -(*divOpZ_)( zDiffFlux );
    }
  }
  else{
    // handle 2D and 1D cases - not quite as efficient since we won't be
    // running as many production scale calculations in these configurations
    
    if (doXDiff_) {
      if( haveXAreaFrac_ ) rhs <<= -(*divOpX_)( (*xAreaFracInterpOp_)(xareafrac_->field_ref()) * xDiffFlux_->field_ref() );
      else                 rhs <<= -(*divOpX_)( xDiffFlux_->field_ref() );
    } else{
      rhs <<= 0.0; // zero so that we can sum in Y and Z contributions as necessary
    }

    if (doYDiff_) {
      if( haveYAreaFrac_ ) rhs <<= rhs -(*divOpY_)( (*yAreaFracInterpOp_)(yareafrac_->field_ref()) * yDiffFlux_->field_ref() );
      else                 rhs <<= rhs -(*divOpY_)( yDiffFlux_->field_ref() );
    }

    if (doZDiff_) {
      if( haveZAreaFrac_ ) rhs <<= rhs -(*divOpZ_)( (*zAreaFracInterpOp_)(zareafrac_->field_ref()) * zDiffFlux_->field_ref() );
      else                 rhs <<= rhs -(*divOpZ_)( zDiffFlux_->field_ref() );
    }
  } // 2D and 1D cases

  // accumulate source terms in.  This isn't quite as efficient
  // because we don't have a great way to inline all of this yet.
  typename std::vector<FieldT>::const_iterator isrc;

  for( size_t i=0; i<srcTerms_.size(); ++i ) {
    if( haveVolFrac_ )  rhs <<= rhs + srcTerms_[i]->field_ref() * (*volFracInterpOp_)(volfrac_->field_ref());
    else                rhs <<= rhs + srcTerms_[i]->field_ref();
  }

  const SVolField& rhoStar = rhoStar_->field_ref();
  rhs <<= - 1.0/rhoStar/rhoStar * dRhoDPhi_->field_ref() * rhs;
}

//------------------------------------------------------------------

template< typename FieldT >
ScalarEOSCoupling<FieldT>::Builder::Builder( const Expr::Tag& result,
                                     const FieldTagInfo& fieldInfo,
                                     const Expr::TagList sources,
                                     const Expr::Tag& rhoStarTag,
                                     const Expr::Tag& dRhoDPhiTag,
                                     const bool isStrongForm)

  : ExpressionBuilder(result),
    info_          ( fieldInfo      ),
    srcT_          ( sources        ),
    rhoStarTag_    ( rhoStarTag     ),
    dRhoDPhiTag_      ( dRhoDPhiTag ),
    isStrongForm_  ( isStrongForm   )
{}

//------------------------------------------------------------------

template< typename FieldT >
ScalarEOSCoupling<FieldT>::Builder::Builder( const Expr::Tag& result,
                                     const FieldTagInfo& fieldInfo,
                                     const Expr::Tag& rhoStarTag,
                                     const Expr::Tag& dRhoDPhiTag,
                                     const bool isStrongForm)
  : ExpressionBuilder(result),
    info_          ( fieldInfo      ),
    rhoStarTag_    ( rhoStarTag     ),
    dRhoDPhiTag_   ( dRhoDPhiTag    ),
    isStrongForm_  ( isStrongForm   )
{}

//------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ScalarEOSCoupling<FieldT>::Builder::build() const
{
  return new ScalarEOSCoupling<FieldT>( info_, srcT_, rhoStarTag_, dRhoDPhiTag_, isStrongForm_ );
}
//------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class ScalarEOSCoupling< SpatialOps::SVolField >;
//==========================================================================
