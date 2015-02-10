/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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
            const Expr::Tag& tauX,
            const Expr::Tag& tauY,
            const Expr::Tag& tauZ,
            const Expr::Tag& densityTag,
            const Expr::Tag& bodyForceTag,
            const Expr::Tag& srcTermTag,
            const Expr::Tag& volFracTag )
  : Expr::Expression<FieldT>(),
    doXConv_( convFluxX != Expr::Tag()),
    doYConv_( convFluxY != Expr::Tag()),
    doZConv_( convFluxZ != Expr::Tag()),

    doXTau_( tauX != Expr::Tag()),
    doYTau_( tauY != Expr::Tag()),
    doZTau_( tauZ != Expr::Tag()),

    is3dconvdiff_( doXConv_ && doYConv_ && doZConv_ && doXTau_ && doYTau_ && doZTau_ ),

    hasBodyF_(bodyForceTag != Expr::Tag()),
    hasSrcTerm_(srcTermTag != Expr::Tag()),
    hasIntrusion_(volFracTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
  
  if( doXConv_ )  this->template create_field_request(convFluxX, cFluxX_);
  if( doYConv_ )  this->template create_field_request(convFluxY, cFluxY_);
  if( doZConv_ )  this->template create_field_request(convFluxZ, cFluxZ_);
  
  if( doXTau_ )  this->template create_field_request(tauX, tauX_);
  if( doYTau_ )  this->template create_field_request(tauY, tauY_);
  if( doZTau_ )  this->template create_field_request(tauZ, tauZ_);

  if(doXTau_ || doYTau_ || doZTau_) this->template create_field_request(viscTag, visc_);
  this->template create_field_request(densityTag, density_);
  
  if( hasBodyF_ )  this->template create_field_request(bodyForceTag, bodyForce_);
  if( hasSrcTerm_ )  this->template create_field_request(srcTermTag, srcTerm_);
  if( hasIntrusion_ )  this->template create_field_request(volFracTag, volfrac_);
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

  result <<= 0.0;

  if( is3dconvdiff_ ){ // inline all convective and diffusive contributions
    const XFluxT& cfx = cFluxX_->field_ref();
    const YFluxT& cfy = cFluxY_->field_ref();
    const ZFluxT& cfz = cFluxZ_->field_ref();
    
    const XFluxT& tx = tauX_->field_ref();
    const YFluxT& ty = tauY_->field_ref();
    const ZFluxT& tz = tauZ_->field_ref();

    const SVolField& mu = visc_->field_ref();
    // note: this does not diff, but is slow:
    result <<= (*divXOp_)(-cfx)
              +(*divYOp_)(-cfy)
              +(*divZOp_)(-cfz)
              + 2.0 * (*divXOp_)((*sVol2XFluxInterpOp_)(mu) * tx )
              + 2.0 * (*divYOp_)((*sVol2YFluxInterpOp_)(mu) * ty )
              + 2.0 * (*divZOp_)((*sVol2ZFluxInterpOp_)(mu) * tz );
//    // this is the fully inlined version, which causes diffs on ~9 tests.
//    result <<= (*divXOp_)( -cfx + 2.0 * (*sVol2XFluxInterpOp_)(mu) * tx ) +
//               (*divYOp_)( -cfy + 2.0 * (*sVol2YFluxInterpOp_)(mu) * ty ) +
//               (*divZOp_)( -cfz + 2.0 * (*sVol2ZFluxInterpOp_)(mu) * tz );
  }
  else{ // 1D and 2D cases, or cases with only convection or diffusion - not optimized for these...
    if( doXConv_ ) result <<= result - (*divXOp_)(cFluxX_->field_ref());
    if( doYConv_ ) result <<= result - (*divYOp_)(cFluxY_->field_ref());
    if( doZConv_ ) result <<= result - (*divZOp_)(cFluxZ_->field_ref());

    if( doXTau_ ) result <<= result + 2.0 * (*divXOp_)( (*sVol2XFluxInterpOp_)(visc_->field_ref()) * tauX_->field_ref()); // + 2*div(mu*S_xi)
    if( doYTau_ ) result <<= result + 2.0 * (*divYOp_)( (*sVol2YFluxInterpOp_)(visc_->field_ref()) * tauY_->field_ref()); // + 2*div(mu*S_yi)
    if( doZTau_ ) result <<= result + 2.0 * (*divZOp_)( (*sVol2ZFluxInterpOp_)(visc_->field_ref()) * tauZ_->field_ref()); // + 2*div(mu*S_zi)
  }
  
  // sum in other terms as required
  if( hasBodyF_ ) result <<= result + (*densityInterpOp_)(density_->field_ref()) * bodyForce_->field_ref();
  if( hasSrcTerm_ ) result <<= result + srcTerm_->field_ref();
  if ( hasIntrusion_ ) result <<= result * volfrac_->field_ref();
}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHSPart<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& convFluxX,
                  const Expr::Tag& convFluxY,
                  const Expr::Tag& convFluxZ,
                  const Expr::Tag& viscTag,
                  const Expr::Tag& tauX,
                  const Expr::Tag& tauY,
                  const Expr::Tag& tauZ,
                  const Expr::Tag& densityTag,
                  const Expr::Tag& bodyForceTag,
                  const Expr::Tag& srcTermTag,
                  const Expr::Tag& volFracTag )
  : ExpressionBuilder(result),
    cfluxXt_   ( convFluxX    ),
    cfluxYt_   ( convFluxY    ),
    cfluxZt_   ( convFluxZ    ),
    viscTag_   ( viscTag      ),
    tauXt_     ( tauX         ),
    tauYt_     ( tauY         ),
    tauZt_     ( tauZ         ),
    densityt_  ( densityTag   ),
    bodyForcet_( bodyForceTag ),
    srcTermt_  ( srcTermTag   ),
    volfract_  ( volFracTag   )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MomRHSPart<FieldT>::Builder::build() const
{
  return new MomRHSPart<FieldT>( cfluxXt_, cfluxYt_, cfluxZt_,
                                 viscTag_, tauXt_, tauYt_, tauZt_,
                                 densityt_, bodyForcet_, srcTermt_,
                                 volfract_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHSPart< SpatialOps::XVolField >;
template class MomRHSPart< SpatialOps::YVolField >;
template class MomRHSPart< SpatialOps::ZVolField >;
//==================================================================
