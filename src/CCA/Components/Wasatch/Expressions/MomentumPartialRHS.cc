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

#include <CCA/Components/Wasatch/Expressions/MomentumPartialRHS.h>
#include <CCA/Components/Wasatch/FieldTypes.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT >
MomRHSPart<FieldT>::
MomRHSPart( const Expr::Tag& convFluxX,
            const Expr::Tag& convFluxY,
            const Expr::Tag& convFluxZ,
            const Expr::Tag& viscTag,
            const Expr::Tag& strainX,
            const Expr::Tag& strainY,
            const Expr::Tag& strainZ,
            const Expr::Tag& dilataionTag,
            const Expr::Tag& densityTag,
            const Expr::Tag& bodyForceTag,
            const Expr::TagList& srcTermTags,
            const Expr::Tag& volFracTag )
  : Expr::Expression<FieldT>(),
    doXConv_( convFluxX != Expr::Tag()),
    doYConv_( convFluxY != Expr::Tag()),
    doZConv_( convFluxZ != Expr::Tag()),

    doXTau_( strainX != Expr::Tag()),
    doYTau_( strainY != Expr::Tag()),
    doZTau_( strainZ != Expr::Tag()),

    is3dconvdiff_( doXConv_ && doYConv_ && doZConv_ && doXTau_ && doYTau_ && doZTau_ ),

    hasBodyF_(bodyForceTag != Expr::Tag()),
    hasSrcTerm_(srcTermTags.size()>0),
    hasIntrusion_(volFracTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
  
  divu_ = this->template create_field_request<SVolField>(dilataionTag);
  
  if( doXConv_ ) cFluxX_ = this->template create_field_request<XFluxT>(convFluxX);
  if( doYConv_ ) cFluxY_ = this->template create_field_request<YFluxT>(convFluxY);
  if( doZConv_ ) cFluxZ_ = this->template create_field_request<ZFluxT>(convFluxZ);
  
  if( doXTau_  ) strainX_ = this->template create_field_request<XFluxT>(strainX);
  if( doYTau_  ) strainY_ = this->template create_field_request<YFluxT>(strainY);
  if( doZTau_  ) strainZ_ = this->template create_field_request<ZFluxT>(strainZ);

  if( doXTau_ || doYTau_ || doZTau_ ) visc_ = this->template create_field_request<SVolField>(viscTag);
  
  if( hasBodyF_ ){
    density_   = this->template create_field_request<SVolField>(densityTag  );
    bodyForce_ = this->template create_field_request<FieldT   >(bodyForceTag);
  }
  if( hasSrcTerm_   ) this->template create_field_vector_request<FieldT>( srcTermTags, srcTerms_ );
  if( hasIntrusion_ ) volfrac_ = this->template create_field_request<FieldT>(volFracTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHSPart<FieldT>::
~MomRHSPart()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  densityInterpOp_ = opDB.retrieve_operator<DensityInterpT>();

  if( doXConv_ || doXTau_ )  divXOp_ = opDB.retrieve_operator<DivX>();
  if( doYConv_ || doYTau_ )  divYOp_ = opDB.retrieve_operator<DivY>();
  if( doZConv_ || doZTau_ )  divZOp_ = opDB.retrieve_operator<DivZ>();

  if( doXTau_ ) sVol2XFluxInterpOp_ = opDB.retrieve_operator<SVol2XFluxInterpT>();
  if( doYTau_ ) sVol2YFluxInterpOp_ = opDB.retrieve_operator<SVol2YFluxInterpT>();
  if( doZTau_ ) sVol2ZFluxInterpOp_ = opDB.retrieve_operator<SVol2ZFluxInterpT>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const WasatchCore::Direction stagLoc = WasatchCore::get_staggered_location<FieldT>();
  
  if( is3dconvdiff_ ){ // inline all convective and diffusive contributions
    const XFluxT& cfx = cFluxX_->field_ref();
    const YFluxT& cfy = cFluxY_->field_ref();
    const ZFluxT& cfz = cFluxZ_->field_ref();
    
    const XFluxT& strainX = strainX_->field_ref();
    const YFluxT& strainY = strainY_->field_ref();
    const ZFluxT& strainZ = strainZ_->field_ref();
    
    SpatialOps::SpatFldPtr<XFluxT> strainXDil = SpatialOps::SpatialFieldStore::get<XFluxT>( result );
    *strainXDil <<= strainX;
    SpatialOps::SpatFldPtr<YFluxT> strainYDil = SpatialOps::SpatialFieldStore::get<YFluxT>( result );
    *strainYDil <<= strainY;
    SpatialOps::SpatFldPtr<ZFluxT> strainZDil = SpatialOps::SpatialFieldStore::get<ZFluxT>( result );
    *strainZDil <<= strainZ;
  
    switch (stagLoc) {
      case WasatchCore::XDIR:
        *strainXDil <<= *strainXDil - 1.0/3.0 * (*sVol2XFluxInterpOp_)(divu_->field_ref());
        break;
      case WasatchCore::YDIR:
        *strainYDil <<= *strainYDil - 1.0/3.0 * (*sVol2YFluxInterpOp_)(divu_->field_ref());
        break;
      case WasatchCore::ZDIR:
        *strainZDil <<= *strainZDil - 1.0/3.0 * (*sVol2ZFluxInterpOp_)(divu_->field_ref());
        break;
      default:
        break;
    }

    const SVolField& mu = visc_->field_ref();
    // note: this does not diff, but is slower:
    result <<= (*divXOp_)(-cfx)
              +(*divYOp_)(-cfy)
              +(*divZOp_)(-cfz)
              + 2.0 * (*divXOp_)((*sVol2XFluxInterpOp_)(mu) * *strainXDil )
              + 2.0 * (*divYOp_)((*sVol2YFluxInterpOp_)(mu) * *strainYDil )
              + 2.0 * (*divZOp_)((*sVol2ZFluxInterpOp_)(mu) * *strainZDil );
//    // this is the fully inlined version, which causes diffs on ~9 tests.
//    result <<= (*divXOp_)( -cfx + 2.0 * (*sVol2XFluxInterpOp_)(mu) * strainX ) +
//               (*divYOp_)( -cfy + 2.0 * (*sVol2YFluxInterpOp_)(mu) * strainY ) +
//               (*divZOp_)( -cfz + 2.0 * (*sVol2ZFluxInterpOp_)(mu) * strainZ );
    

  }
  else{ // 1D and 2D cases, or cases with only convection or diffusion - not optimized for these...

    result <<= 0.0;

    if( doXConv_ ) result <<= result - (*divXOp_)(cFluxX_->field_ref());
    if( doYConv_ ) result <<= result - (*divYOp_)(cFluxY_->field_ref());
    if( doZConv_ ) result <<= result - (*divZOp_)(cFluxZ_->field_ref());

    if( doXTau_ ) {
      SpatialOps::SpatFldPtr<XFluxT> strainXDil = SpatialOps::SpatialFieldStore::get<XFluxT>( result );
      *strainXDil <<= strainX_->field_ref();
      if (stagLoc == WasatchCore::XDIR) *strainXDil <<= *strainXDil - 1.0/3.0 * (*sVol2XFluxInterpOp_)(divu_->field_ref());
      result <<= result + 2.0 * (*divXOp_)( (*sVol2XFluxInterpOp_)(visc_->field_ref()) * *strainXDil); // + 2*div(mu*S_xi)
    }
    if( doYTau_ ) {
      SpatialOps::SpatFldPtr<YFluxT> strainYDil = SpatialOps::SpatialFieldStore::get<YFluxT>( result );
      *strainYDil <<= strainY_->field_ref();
      if (stagLoc == WasatchCore::YDIR) *strainYDil <<= *strainYDil - 1.0/3.0 * (*sVol2YFluxInterpOp_)(divu_->field_ref());
      result <<= result + 2.0 * (*divYOp_)( (*sVol2YFluxInterpOp_)(visc_->field_ref()) * *strainYDil); // + 2*div(mu*S_yi)
    }
    if( doZTau_ ) {
      SpatialOps::SpatFldPtr<ZFluxT> strainZDil = SpatialOps::SpatialFieldStore::get<ZFluxT>( result );
      *strainZDil <<= strainZ_->field_ref();
      if (stagLoc == WasatchCore::ZDIR) *strainZDil <<= *strainZDil - 1.0/3.0 * (*sVol2ZFluxInterpOp_)(divu_->field_ref());
      result <<= result + 2.0 * (*divZOp_)( (*sVol2ZFluxInterpOp_)(visc_->field_ref()) * *strainZDil); // + 2*div(mu*S_zi)
    }
  }
  
  // sum in other terms as required
  if( hasBodyF_     ) result <<= result + (*densityInterpOp_)(density_->field_ref()) * bodyForce_->field_ref();
//  if( hasSrcTerm_   ) result <<= result + srcTerm_->field_ref();
  // accumulate source terms in.  This isn't quite as efficient
  // because we don't have a great way to inline all of this yet.
  typename std::vector<FieldT>::const_iterator isrc;
  for( size_t i=0; i<srcTerms_.size(); ++i ) {
    result <<= result + srcTerms_[i]->field_ref();
  }

  if( hasIntrusion_ ) result <<= result * volfrac_->field_ref();
}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHSPart<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& convFluxX,
                  const Expr::Tag& convFluxY,
                  const Expr::Tag& convFluxZ,
                  const Expr::Tag& viscTag,
                  const Expr::Tag& strainX,
                  const Expr::Tag& strainY,
                  const Expr::Tag& strainZ,
                  const Expr::Tag& dilataionTag,
                  const Expr::Tag& densityTag,
                  const Expr::Tag& bodyForceTag,
                  const Expr::TagList& srcTermTags,
                  const Expr::Tag& volFracTag )
  : ExpressionBuilder(result),
    cfluxXt_    ( convFluxX    ),
    cfluxYt_    ( convFluxY    ),
    cfluxZt_    ( convFluxZ    ),
    viscTag_    ( viscTag      ),
    strainXt_   ( strainX      ),
    strainYt_   ( strainY      ),
    strainZt_   ( strainZ      ),
    dilataiont_ ( dilataionTag ),
    densityt_   ( densityTag   ),
    bodyForcet_ ( bodyForceTag ),
    srcTermtags_( srcTermTags   ),
    volfract_   ( volFracTag   )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MomRHSPart<FieldT>::Builder::build() const
{
  return new MomRHSPart<FieldT>( cfluxXt_, cfluxYt_, cfluxZt_,
                                 viscTag_, strainXt_, strainYt_, strainZt_,
                                 dilataiont_, densityt_, bodyForcet_, srcTermtags_,
                                 volfract_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHSPart< SpatialOps::SVolField >;
template class MomRHSPart< SpatialOps::XVolField >;
template class MomRHSPart< SpatialOps::YVolField >;
template class MomRHSPart< SpatialOps::ZVolField >;
//==================================================================
